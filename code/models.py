# Copyright 2019 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Model graph definitions and other functions for training and testing."""

import functools
import math
import operator
import os
import random
import re
import numpy as np
import tensorflow as tf


def get_model(config, gpuid):
  """Make model instance and pin to one gpu.

  Args:
    config: arguments.
    gpuid: gpu id to use
  Returns:
    Model instance.
  """
  with tf.name_scope(config.modelname), tf.device('/gpu:%d' % gpuid):
    model = Model(config, '%s' % config.modelname)
  return model


class Model(object):
  """Model graph definitions.
  """

  def __init__(self, config, scope):
    self.scope = scope
    self.config = config

    self.global_step = tf.get_variable('global_step', shape=[],
                                       dtype='int32',
                                       initializer=tf.constant_initializer(0),
                                       trainable=False)

    # get all the dimension here
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N = config.batch_size

    KP = self.KP = config.kp_size

    SH = self.SH = config.scene_h
    SW = self.SW = config.scene_w
    SC = self.SC = config.scene_class

    K = self.K = config.max_other

    self.P = P = 2  # traj coordinate dimension

    # all the inputs

    # the trajactory sequence,
    # in training, it is the obs+pred combined,
    # in testing, only obs is fed and the rest is zeros
    # [N,T1,2] # T1 is the obs_len
    # mask is used for variable length input extension
    self.traj_obs_gt = tf.placeholder(
        'float', [N, None, P], name='traj_obs_gt')
    self.traj_obs_gt_mask = tf.placeholder(
        'bool', [N, None], name='traj_obs_gt_mask')

    # [N,T2,2]
    self.traj_pred_gt = tf.placeholder(
        'float', [N, None, P], name='traj_pred_gt')
    self.traj_pred_gt_mask = tf.placeholder(
        'bool', [N, None], name='traj_pred_gt_mask')

    self.obs_kp = tf.placeholder('float', [N, None, KP, 2], name='obs_kp')

    # used for drop out switch
    self.is_train = tf.placeholder('bool', [], name='is_train')

    # scene semantic segmentation features
    # the index to the feature
    self.obs_scene = tf.placeholder('int32', [N, None], name='obs_scene')
    self.obs_scene_mask = tf.placeholder(
        'bool', [N, None], name='obs_scene_mask')
    # the actual feature
    self.scene_feat = tf.placeholder(
        'float32', [None, SH, SW, SC], name='scene_feat')

    # [N, obs_len, 5, 9, 2048]
    self.obs_person_features = tf.placeholder('float32', [
        N, None, config.person_h, config.person_w,
        config.person_feat_dim], name='obs_boxes_features')

    # other box
    # the box input is the relative coordinates
    # [N,obs_len, K, 4]
    self.obs_other_boxes = tf.placeholder(
        'float32', [N, None, K, 4], name='other_boxes')
    # [N,obs_len, K, num_class]
    self.obs_other_boxes_class = tf.placeholder(
        'float32', [N, None, K, config.num_box_class], name='other_boxes_class')
    # [N,obs_len, K]
    self.obs_other_boxes_mask = tf.placeholder(
        'bool', [N, None, K], name='other_boxes_mask')

    # grid loss
    self.grid_pred_labels = []
    self.grid_pred_targets = []
    self.grid_obs_labels = []
    self.grid_obs_targets = []
    for _ in config.scene_grids:
      # [N, seq_len]
      # currently only the destination
      self.grid_pred_labels.append(
          tf.placeholder('int32', [N]))  # grid class
      self.grid_pred_targets.append(tf.placeholder('float32', [N, 2]))

      self.grid_obs_labels.append(
          tf.placeholder('int32', [N, None]))  # grid class
      self.grid_obs_targets.append(
          tf.placeholder('float32', [N, None, 2]))

    # traj class loss
    self.traj_class_gt = tf.placeholder('int64', [N], name='traj_class')

    self.future_act_label = tf.placeholder(
        'uint8', [N, config.num_act], name='future_act')

    self.loss = None
    self.build_forward()
    self.build_loss()

  def build_forward(self):
    """Build the forward model graph."""
    config = self.config
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N
    KP = self.KP

    # add dropout
    keep_prob = tf.cond(self.is_train,
                        lambda: tf.constant(config.keep_prob),
                        lambda: tf.constant(1.0))

    # ------------------------- encoder ------------------------
    enc_cell_traj = tf.nn.rnn_cell.LSTMCell(
        config.enc_hidden_size, state_is_tuple=True, name='enc_traj')
    enc_cell_traj = tf.nn.rnn_cell.DropoutWrapper(enc_cell_traj, keep_prob)

    # scene encoder
    enc_cell_personscene = tf.nn.rnn_cell.LSTMCell(
        config.enc_hidden_size, state_is_tuple=True, name='enc_scene')
    enc_cell_personscene = tf.nn.rnn_cell.DropoutWrapper(
        enc_cell_personscene, keep_prob)

    # person pose encoder
    if config.add_kp:
      enc_cell_kp = tf.nn.rnn_cell.LSTMCell(
          config.enc_hidden_size, state_is_tuple=True, name='enc_kp')
      enc_cell_kp = tf.nn.rnn_cell.DropoutWrapper(enc_cell_kp, keep_prob)

    # person appearance encoder
    enc_cell_person = tf.nn.rnn_cell.LSTMCell(
        config.enc_hidden_size, state_is_tuple=True, name='enc_person')
    # enc_cell_person = tf.contrib.rnn.ConvLSTMCell(conv_ndims=2,
    #     input_shape=[person_h, person_w, person_feat_dim],
    #     output_channels=config.enc_hidden_size, kernel_shape=[3,3])
    enc_cell_person = tf.nn.rnn_cell.DropoutWrapper(
        enc_cell_person, keep_prob)

    # other box encoder
    enc_cell_other = tf.nn.rnn_cell.LSTMCell(
        config.enc_hidden_size, state_is_tuple=True, name='enc_other')
    enc_cell_other = tf.nn.rnn_cell.DropoutWrapper(
        enc_cell_other, keep_prob)

    # activity location/grid loss
    enc_cell_gridclass = []
    for i, _ in enumerate(config.scene_grids):
      enc_cell_gridclass_this = tf.nn.rnn_cell.LSTMCell(
          config.enc_hidden_size, state_is_tuple=True,
          name='enc_gridclass_%s' % i)
      enc_cell_gridclass_this = tf.nn.rnn_cell.DropoutWrapper(
          enc_cell_gridclass_this, keep_prob)
      enc_cell_gridclass.append(enc_cell_gridclass_this)

    # ------------------------ decoder

    if config.multi_decoder:
      dec_cell_traj = [tf.nn.rnn_cell.LSTMCell(
          config.dec_hidden_size, state_is_tuple=True, name='dec_traj_%s' % i)
                       for i in range(len(config.traj_cats))]
      dec_cell_traj = [tf.nn.rnn_cell.DropoutWrapper(
          one, keep_prob) for one in dec_cell_traj]
    else:
      dec_cell_traj = tf.nn.rnn_cell.LSTMCell(
          config.dec_hidden_size, state_is_tuple=True, name='dec_traj')
      dec_cell_traj = tf.nn.rnn_cell.DropoutWrapper(
          dec_cell_traj, keep_prob)

    # ----------------------------------------------------------
    # the obs part is the same for training and testing
    # obs_out is only used in training

    # encoder, decoder
    # top_scope is used for variable inside
    # encode and decode if want to share variable across
    with tf.variable_scope('person_pred') as top_scope:

      # [N,T1,h_dim]
      # xy encoder
      obs_length = tf.reduce_sum(
          tf.cast(self.traj_obs_gt_mask, 'int32'), 1)

      traj_xy_emb_enc = linear(self.traj_obs_gt,
                               output_size=config.emb_size,
                               activation=config.activation_func,
                               add_bias=True,
                               scope='enc_xy_emb')
      traj_obs_enc_h, traj_obs_enc_last_state = tf.nn.dynamic_rnn(
          enc_cell_traj, traj_xy_emb_enc, sequence_length=obs_length,
          dtype='float', scope='encoder_traj')

      enc_h_list = [traj_obs_enc_h]

      enc_last_state_list = [traj_obs_enc_last_state]

      # grid class and grid regression encoder
      # multi-scale
      grid_obs_enc_h = []
      grid_obs_enc_last_state = []

      for i, (h, w) in enumerate(config.scene_grids):
        #  [N, T] -> [N, T, h*w]
        obs_gridclass_onehot = tf.one_hot(self.grid_obs_labels[i], h*w)
        obs_gridclass_encode_h, obs_gridclass_encode_last_state = \
            tf.nn.dynamic_rnn(enc_cell_gridclass[i], obs_gridclass_onehot,
                              sequence_length=obs_length, dtype='float',
                              scope='encoder_gridclass_%s' % i)
        grid_obs_enc_h.append(obs_gridclass_encode_h)
        grid_obs_enc_last_state.append(obs_gridclass_encode_last_state)

      enc_h_list.extend(grid_obs_enc_h)

      enc_last_state_list.extend(grid_obs_enc_last_state)

      # gather all visual observation encoder
      # ------------------------------------------------------------
      with tf.variable_scope('scene'):
        # [N,obs_len, SH, SW, SC]
        obs_scene = tf.nn.embedding_lookup(
            self.scene_feat, self.obs_scene)
        obs_scene = tf.reduce_mean(obs_scene, axis=1)  # [N,SH,SW,SC]

        with tf.variable_scope('scene_conv'):
          # [N, SH, SW, dim]
          # resnet structure?
          conv_dim = config.scene_conv_dim

          scene_conv1 = obs_scene

          # [N, SH/2, SW/2, dim]
          scene_conv2 = conv2d(scene_conv1, out_channel=conv_dim,
                               kernel=config.scene_conv_kernel,
                               stride=2, activation=config.activation_func,
                               add_bias=True, scope='conv2')
          # [N, SH/4, SW/4, dim]
          scene_conv3 = conv2d(scene_conv2, out_channel=conv_dim,
                               kernel=config.scene_conv_kernel,
                               stride=2, activation=config.activation_func,
                               add_bias=True, scope='conv3')
          self.scene_convs = [scene_conv2, scene_conv3]

        # pool the scene features for each trajectory, for different scale
        # currently only used single scale conv
        pool_scale_idx = config.pool_scale_idx

        scene_h, scene_w = config.scene_grids[pool_scale_idx]

        # [N, num_grid_class, conv_dim]
        scene_conv_full = tf.reshape(
            self.scene_convs[pool_scale_idx], (N, scene_h*scene_w, conv_dim))

        # [N, seq_len]
        obs_grid = self.grid_obs_labels[pool_scale_idx]

        obs_grid = tf.reshape(obs_grid, [-1])  # [N*seq_len]
        # [N*seq_len, 2]
        indices = tf.stack(
            [tf.range(tf.shape(obs_grid)[0]), tf.to_int32(obs_grid)], axis=-1)

        # [N, seq_len, num_grid_class, conv_dim]
        scene_conv_full_tile = tf.tile(tf.expand_dims(
            scene_conv_full, 1), [1, config.obs_len, 1, 1])
        # [N*seq_len, num_grid_class, conv_dim]
        scene_conv_full_tile = tf.reshape(
            scene_conv_full_tile, (-1, scene_h*scene_w, conv_dim))

        # [N*seq_len, h*w, feat_dim] + [N*seq_len,2] -> # [N*seq_len, feat_dim]
        obs_personscene = tf.gather_nd(scene_conv_full_tile, indices)
        obs_personscene = tf.reshape(
            obs_personscene, (N, config.obs_len, conv_dim))

        # obs_personscene [N, seq_len, conv_dim]
        personscene_obs_enc_h, personscene_obs_enc_last_state = \
            tf.nn.dynamic_rnn(enc_cell_personscene, obs_personscene,
                              sequence_length=obs_length, dtype='float',
                              scope='encoder_personscene')

        enc_h_list.append(personscene_obs_enc_h)
        enc_last_state_list.append(personscene_obs_enc_last_state)

      # person pose
      if config.add_kp:
        obs_kp = tf.reshape(self.obs_kp, [N, -1, KP*2])
        obs_kp = linear(obs_kp, output_size=config.emb_size, add_bias=True,
                        activation=config.activation_func, scope='kp_emb')

        kp_obs_enc_h, kp_obs_enc_last_state = tf.nn.dynamic_rnn(
            enc_cell_kp, obs_kp, sequence_length=obs_length, dtype='float',
            scope='encoder_kp')

        enc_h_list.append(kp_obs_enc_h)
        enc_last_state_list.append(kp_obs_enc_last_state)

      # person appearance
      # average and then normal lstm
      obs_person_features = tf.reduce_mean(
          self.obs_person_features, axis=[2, 3])
      # [N,T,hdim]
      person_obs_enc_h, person_obs_enc_last_state = tf.nn.dynamic_rnn(
          enc_cell_person, obs_person_features, sequence_length=obs_length,
          dtype='float', scope='encoder_person')
      enc_h_list.append(person_obs_enc_h)
      enc_last_state_list.append(person_obs_enc_last_state)

      # extract features from other boxes
      # obs_other_boxes [N, obs_len, K, 4]
      # obs_other_boxes_class [N, obs_len, K, num_class]
      # obs_other_boxes_mask [N, obs_len, K]

      with tf.variable_scope('other_box'):
        # [N, obs_len, K, box_emb_size]
        obs_other_boxes_geo_features = linear(
            self.obs_other_boxes, add_bias=True,
            activation=config.activation_func, output_size=config.box_emb_size,
            scope='other_box_geo_emb')
        obs_other_boxes_class_features = linear(
            self.obs_other_boxes_class, add_bias=True,
            activation=config.activation_func, output_size=config.box_emb_size,
            scope='other_box_class_emb')

        obs_other_boxes_features = tf.concat(
            [obs_other_boxes_geo_features, obs_other_boxes_class_features],
            axis=3)

        # cosine simi
        obs_other_boxes_geo_features = tf.nn.l2_normalize(
            obs_other_boxes_geo_features, -1)
        obs_other_boxes_class_features = tf.nn.l2_normalize(
            obs_other_boxes_class_features, -1)
        # [N, T,K]
        other_attention = tf.reduce_sum(tf.multiply(
            obs_other_boxes_geo_features, obs_other_boxes_class_features), 3)

        other_attention = exp_mask(
            other_attention, self.obs_other_boxes_mask)

        other_attention = tf.nn.softmax(other_attention)

        # [N, obs_len, K, 1] * [N, obs_len, K, feat_dim]
        # -> [N, obs_len, feat_dim]
        other_box_features_attended = tf.reduce_sum(tf.expand_dims(
            other_attention, -1)*obs_other_boxes_features, axis=2)

        other_obs_enc_h, other_obs_enc_last_state = tf.nn.dynamic_rnn(
            enc_cell_other, other_box_features_attended,
            sequence_length=obs_length, dtype='float', scope='encoder_other')

      enc_h_list.append(other_obs_enc_h)
      enc_last_state_list.append(other_obs_enc_last_state)

      # pack all observed hidden states
      obs_enc_h = tf.stack(enc_h_list, axis=1)
      # .h is [N,h_dim*k]
      obs_enc_last_state = concat_states(enc_last_state_list, axis=1)

      # -------------------------------------------------- xy decoder
      traj_obs_last = self.traj_obs_gt[:, -1]

      pred_length = tf.reduce_sum(
          tf.cast(self.traj_pred_gt_mask, 'int32'), 1)  # N

      if config.multi_decoder:

        # [N, num_traj_cat] # each is num_traj_cat classification
        self.traj_class_logits = self.traj_class_head(
            obs_enc_h, obs_enc_last_state, scope='traj_class_predict')

        # [N]
        traj_class = tf.argmax(self.traj_class_logits, axis=1)

        traj_class_gated = tf.cond(
            self.is_train,
            lambda: self.traj_class_gt,
            lambda: traj_class,
        )

        traj_pred_outs = [
            self.decoder(
                traj_obs_last,
                traj_obs_enc_last_state,
                obs_enc_h,
                pred_length,
                dec_cell_traj[traj_cat],
                top_scope=top_scope,
                scope='decoder_%s' % traj_cat)
            for _, traj_cat in config.traj_cats
        ]

        # [N, num_decoder, T, 2]
        self.traj_pred_outs = tf.stack(traj_pred_outs, axis=1)

        # [N, 2]
        indices = tf.stack(
            [tf.range(N), tf.to_int32(traj_class_gated)], axis=1)

        # [N, T, 2]
        traj_pred_out = tf.gather_nd(self.traj_pred_outs, indices)

      else:
        traj_pred_out = self.decoder(traj_obs_last, traj_obs_enc_last_state,
                                     obs_enc_h, pred_length, dec_cell_traj,
                                     top_scope=top_scope, scope='decoder')

      if config.add_activity:
        # activity decoder
        self.future_act_logits = self.activity_head(
            obs_enc_h, obs_enc_last_state, scope='activity_predict')

      # predict the activity destination
      with tf.variable_scope('grid_head', reuse=tf.AUTO_REUSE):
        conv_dim = config.scene_conv_dim

        assert len(config.scene_grids) == 2
        # grid class and grid target output
        self.grid_class_logits = []
        self.grid_target_logits = []
        for i, (h, w) in enumerate(config.scene_grids):
          # [h,w,c]
          this_scene_conv = self.scene_convs[i]
          this_scene_conv = tf.reshape(
              this_scene_conv, [N, h*w, conv_dim])

          # tile
          # [N, h*w, h_dim*k]
          h_tile = tf.tile(tf.expand_dims(
              obs_enc_last_state.h, axis=1), [1, h*w, 1])

          # [N, h*w, conv_dim + h_dim + emb]

          scene_feature = tf.concat(
              [h_tile, this_scene_conv], axis=-1)

          # add the occupation map, grid obs input is already in the h_tile
          # [N, T, h*w]
          obs_gridclass_onehot = tf.one_hot(
              self.grid_obs_labels[i], h*w)
          obs_gridclass_occupy = tf.reduce_sum(
              obs_gridclass_onehot, axis=1)
          obs_gridclass = tf.cast(
              obs_gridclass_occupy, 'float32')  # [N,h*w]
          obs_gridclass = tf.reshape(obs_gridclass, [N, h*w, 1])

          # [N, h*w, 1] -> [N, h*w, emb]
          obs_grid_class_emb = linear(obs_gridclass,
                                      output_size=config.emb_size,
                                      activation=config.activation_func,
                                      add_bias=True,
                                      scope='obs_grid_class_emb_%d' % i)
          scene_feature = tf.concat(
              [scene_feature, obs_grid_class_emb], axis=-1)

          grid_class_logit = conv2d(tf.reshape(scene_feature, [N, h, w, -1]),
                                    out_channel=1, kernel=1, stride=1,
                                    activation=config.activation_func,
                                    add_bias=True, scope='grid_class_%d' % i)
          grid_target_logit_all = conv2d(tf.reshape(scene_feature,
                                                    [N, h, w, -1]),
                                         out_channel=2, kernel=1, stride=1,
                                         activation=config.activation_func,
                                         add_bias=True,
                                         scope='grid_target_%d' % i)
          grid_class_logit = tf.reshape(
              grid_class_logit, [N, h*w, 1])
          grid_target_logit_all = tf.reshape(
              grid_target_logit_all, [N, h*w, 2])

          grid_class_logit = tf.squeeze(grid_class_logit, axis=-1)

          # [N]
          target_class = tf.argmax(grid_class_logit, axis=-1)

          # [N,2]
          indices = tf.stack(
              [tf.range(N), tf.to_int32(target_class)], axis=-1)
          # [N,h*w,2] + [N,2] -> # [N,2]
          grid_target_logit = tf.gather_nd(
              grid_target_logit_all, indices)

          self.grid_class_logits.append(grid_class_logit)
          self.grid_target_logits.append(grid_target_logit)

    # for loss and forward
    self.traj_pred_out = traj_pred_out

  # output [N, num_decoder]
  # enc_h for future extension, so pylint: disable=unused-argument
  def traj_class_head(self, enc_h, enc_last_state, scope='predict_traj_cat'):
    """Trajectory classification branch."""
    config = self.config
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

      # [N, hdim*num_enc]
      feature = enc_last_state.h

      # [N, num_traj_class]
      logits = linear(feature, output_size=len(config.traj_cats),
                      add_bias=False, activation=tf.identity,
                      scope='traj_cat_logits')

      return logits

  def activity_head(self, enc_h, enc_last_state, scope='activity_predict'):
    """Activity prediction branch."""
    config = self.config

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

      feature = enc_last_state.h

      future_act = linear(feature, output_size=config.num_act, add_bias=False,
                          activation=tf.identity, scope='future_act')

      return future_act

  def decoder(self, first_input, enc_last_state, enc_h, pred_length, rnn_cell,
              top_scope, scope):
    """Decoder definition."""
    config = self.config
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N
    P = self.P

    with tf.variable_scope(scope):
      # this is only used for training
      with tf.name_scope('prepare_pred_gt_training'):
        # these input only used during training
        time_1st_traj_pred = tf.transpose(
            self.traj_pred_gt, perm=[1, 0, 2])  # [N,T2,W] -> [T2,N,W]
        T2 = tf.shape(time_1st_traj_pred)[0]  # T2
        traj_pred_gt = tf.TensorArray(size=T2, dtype='float')
        traj_pred_gt = traj_pred_gt.unstack(
            time_1st_traj_pred)  # [T2] , [N,W]

      # all None for first call
      with tf.name_scope('decoder_rnn'):
        def decoder_loop_fn(time, cell_output, cell_state, loop_state):
          """RNN loop function for the decoder."""
          emit_output = cell_output  # == None for time==0

          elements_finished = time >= pred_length
          finished = tf.reduce_all(elements_finished)

          # h_{t-1}
          with tf.name_scope('prepare_next_cell_state'):

            if cell_output is None:
              next_cell_state = enc_last_state
            else:
              next_cell_state = cell_state

          # x_t
          with tf.name_scope('prepare_next_input'):
            if cell_output is None:  # first time
              next_input_xy = first_input  # the last observed x,y as input
            else:
              # for testing, construct from this output to be next input
              next_input_xy = tf.cond(
                  # first check the sequence finished or not
                  finished,
                  lambda: tf.zeros([N, P], dtype='float'),
                  # pylint: disable=g-long-lambda
                  lambda: tf.cond(
                      self.is_train,
                      # this will make training faster than testing
                      lambda: traj_pred_gt.read(time),
                      # hidden vector from last step to coordinates
                      lambda: self.hidden2xy(cell_output, scope=top_scope,
                                             additional_scope='hidden2xy'))
              )

            # spatial embedding
            # [N,emb]
            xy_emb = linear(next_input_xy, output_size=config.emb_size,
                            activation=config.activation_func, add_bias=True,
                            scope='xy_emb_dec')

            next_input = xy_emb

            with tf.name_scope('attend_enc'):
              # [N,h_dim]

              attended_encode_states = focal_attention(
                  next_cell_state.h, enc_h, use_sigmoid=False,
                  scope='decoder_attend_encoders')

            next_input = tf.concat(
                [xy_emb, attended_encode_states], axis=1)

          return elements_finished, next_input, next_cell_state, \
              emit_output, None  # next_loop_state

        decoder_out_ta, _, _ = tf.nn.raw_rnn(
            rnn_cell, decoder_loop_fn, scope='decoder_rnn')

      with tf.name_scope('reconstruct_output'):
        decoder_out_h = decoder_out_ta.stack()  # [T2,N,h_dim]
        # [N,T2,h_dim]
        decoder_out_h = tf.transpose(decoder_out_h, perm=[1, 0, 2])

      # recompute the output;
      # if use loop_state to save the output, will 10x slower

      # use the same hidden2xy for different decoder
      decoder_out = self.hidden2xy(
          decoder_out_h, scope=top_scope, additional_scope='hidden2xy')

    return decoder_out

  def hidden2xy(self, lstm_h, return_scope=False, scope='hidden2xy',
                additional_scope=None):
    """Hiddent states to xy coordinates."""
    # Tensor dimensions, so pylint: disable=g-bad-name
    P = self.P
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE) as this_scope:
      if additional_scope is not None:
        return self.hidden2xy(lstm_h, return_scope=return_scope,
                              scope=additional_scope, additional_scope=None)

      out_xy = linear(lstm_h, output_size=P, activation=tf.identity,
                      add_bias=False, scope='out_xy_mlp2')

      if return_scope:
        return out_xy, this_scope
      return out_xy

  def build_loss(self):
    """Model loss."""
    config = self.config
    losses = []
    # N,T,W
    # L2 loss
    # [N,T2,W]
    traj_pred_out = self.traj_pred_out

    traj_pred_gt = self.traj_pred_gt

    diff = traj_pred_out - traj_pred_gt

    xyloss = tf.pow(diff, 2)  # [N,T2,2]
    xyloss = tf.reduce_mean(xyloss)

    self.xyloss = xyloss

    losses.append(xyloss)

    # trajectory classification loss
    if config.multi_decoder:
      traj_class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=self.traj_class_gt, logits=self.traj_class_logits)
      traj_class_loss = tf.reduce_mean(
          traj_class_loss)*tf.constant(config.traj_class_loss_weight,
                                       dtype='float')

      self.traj_class_loss = traj_class_loss
      losses.append(traj_class_loss)

    # ------------------------ activity destination loss
    self.grid_loss = []
    grid_loss_weight = config.grid_loss_weight
    for i, _ in enumerate(config.scene_grids):
      grid_pred_label = self.grid_pred_labels[i]  # [N]
      grid_pred_target = self.grid_pred_targets[i]  # [N,2]

      grid_class_logit = self.grid_class_logits[i]  # [N,h*w]
      grid_target_logit = self.grid_target_logits[i]  # [N,2]

      # classification loss
      class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
          labels=grid_pred_label, logits=grid_class_logit)
      class_loss = tf.reduce_mean(class_loss)

      # regression loss
      regression_loss = tf.losses.huber_loss(
          labels=grid_pred_target, predictions=grid_target_logit,
          reduction=tf.losses.Reduction.MEAN)

      class_loss = class_loss * \
        tf.constant(grid_loss_weight, dtype='float')
      regression_loss = regression_loss * \
        tf.constant(grid_loss_weight, dtype='float')

      self.grid_loss.extend([class_loss, regression_loss])

      losses.extend([class_loss, regression_loss])

    # --------- activity class loss
    if config.add_activity:
      act_loss_weight = config.act_loss_weight
      future_act_logits = self.future_act_logits  # [N,num_act]
      future_act_label = self.future_act_label  # [N,num_act]

      activity_loss = tf.nn.sigmoid_cross_entropy_with_logits(
          labels=tf.cast(future_act_label, 'float32'), logits=future_act_logits)
      activity_loss = tf.reduce_mean(activity_loss)

      activity_loss = activity_loss * \
        tf.constant(act_loss_weight, dtype='float')

      self.activity_loss = activity_loss
      losses.extend([activity_loss])

    if config.wd is not None:
      wd = wd_cost('.*/W', config.wd, scope='wd_cost')
      if wd:
        wd = tf.add_n(wd)
        losses.append(wd)

    # there might be l2 weight loss in some layer
    self.loss = tf.add_n(losses, name='total_losses')

  def get_feed_dict(self, batch, is_train=False):
    """Givng a batch of data, construct the feed dict."""
    # get the cap for each kind of step first
    config = self.config
    # Tensor dimensions, so pylint: disable=g-bad-name
    N = self.N
    P = self.P
    KP = self.KP

    T_in = config.obs_len
    T_pred = config.pred_len

    feed_dict = {}

    # initial all the placeholder

    traj_obs_gt = np.zeros([N, T_in, P], dtype='float')
    traj_obs_gt_mask = np.zeros([N, T_in], dtype='bool')

    # link the feed_dict
    feed_dict[self.traj_obs_gt] = traj_obs_gt
    feed_dict[self.traj_obs_gt_mask] = traj_obs_gt_mask

    # for getting pred length during test time
    traj_pred_gt_mask = np.zeros([N, T_pred], dtype='bool')
    feed_dict[self.traj_pred_gt_mask] = traj_pred_gt_mask

    # this is needed since it is in tf.conf?
    traj_pred_gt = np.zeros([N, T_pred, P], dtype='float')
    feed_dict[self.traj_pred_gt] = traj_pred_gt  # all zero when testing,

    feed_dict[self.is_train] = is_train

    data = batch.data
    # encoder features
    # ------------------------------------- xy input

    assert len(data['obs_traj_rel']) == N

    for i, (obs_data, pred_data) in enumerate(zip(data['obs_traj_rel'],
                                                  data['pred_traj_rel'])):
      for j, xy in enumerate(obs_data):
        traj_obs_gt[i, j, :] = xy
        traj_obs_gt_mask[i, j] = True
      for j in range(config.pred_len):
        # used in testing to get the prediction length
        traj_pred_gt_mask[i, j] = True
    # ---------------------------------------

    # scene input
    obs_scene = np.zeros((N, T_in), dtype='int32')
    obs_scene_mask = np.zeros((N, T_in), dtype='bool')

    feed_dict[self.obs_scene] = obs_scene
    feed_dict[self.obs_scene_mask] = obs_scene_mask
    feed_dict[self.scene_feat] = data['batch_scene_feat']

    # each bacth
    for i in range(len(data['batch_obs_scene'])):
      for j in range(len(data['batch_obs_scene'][i])):
        # it was (1) shaped
        obs_scene[i, j] = data['batch_obs_scene'][i][j][0]
        obs_scene_mask[i, j] = True

    # [N,num_scale, T] # each is int to num_grid_class
    for j, _ in enumerate(config.scene_grids):
      this_grid_label = np.zeros([N, T_in], dtype='int32')
      for i in range(len(data['obs_grid_class'])):
        this_grid_label[i, :] = data['obs_grid_class'][i][j, :]

      feed_dict[self.grid_obs_labels[j]] = this_grid_label

    # person pose input
    if config.add_kp:
      obs_kp = np.zeros((N, T_in, KP, 2), dtype='float')

      feed_dict[self.obs_kp] = obs_kp

      # each bacth
      for i, obs_kp_rel in enumerate(data['obs_kp_rel']):
        for j, obs_kp_step in enumerate(obs_kp_rel):
          obs_kp[i, j, :, :] = obs_kp_step

    split = 'train'
    if not is_train:
      split = 'val'
    if config.is_test:
      split = 'test'

    # this is the h/w the bounding box is based on
    person_h = config.person_h
    person_w = config.person_w
    person_feat_dim = config.person_feat_dim

    obs_person_features = np.zeros(
        (N, T_in, person_h, person_w, person_feat_dim), dtype='float32')

    for i in range(len(data['obs_boxid'])):
      for j in range(len(data['obs_boxid'][i])):
        boxid = data['obs_boxid'][i][j]
        featfile = os.path.join(
            config.person_feat_path, split, '%s.npy' % boxid)
        obs_person_features[i, j] = np.squeeze(
            np.load(featfile), axis=0)

    feed_dict[self.obs_person_features] = obs_person_features

    # add other boxes,
    K = self.K  # max_other boxes
    other_boxes_class = np.zeros(
        (N, T_in, K, config.num_box_class), dtype='float32')
    other_boxes = np.zeros((N, T_in, K, 4), dtype='float32')
    other_boxes_mask = np.zeros((N, T_in, K), dtype='bool')
    for i in range(len(data['obs_other_box'])):
      for j in range(len(data['obs_other_box'][i])):  # -> seq_len
        this_other_boxes = data['obs_other_box'][i][j]
        this_other_boxes_class = data['obs_other_box_class'][i][j]

        other_box_idxs = range(len(this_other_boxes))

        if config.random_other:
          random.shuffle(other_box_idxs)

        other_box_idxs = other_box_idxs[:K]

        # get the current person box
        this_person_x1y1x2y2 = data['obs_box'][i][j]  # (4)

        for k, idx in enumerate(other_box_idxs):
          other_boxes_mask[i, j, k] = True

          other_box_x1y1x2y2 = this_other_boxes[idx]

          other_boxes[i, j, k, :] = self.encode_other_boxes(
              this_person_x1y1x2y2, other_box_x1y1x2y2)
          # one-hot representation
          box_class = this_other_boxes_class[idx]
          other_boxes_class[i, j, k, box_class] = 1

    feed_dict[self.obs_other_boxes] = other_boxes
    feed_dict[self.obs_other_boxes_class] = other_boxes_class
    feed_dict[self.obs_other_boxes_mask] = other_boxes_mask

    # -----------------------------------------------------------

    # ----------------------------training
    if is_train:
      for i, (obs_data, pred_data) in enumerate(zip(data['obs_traj_rel'],
                                                    data['pred_traj_rel'])):
        for j, xy in enumerate(pred_data):
          traj_pred_gt[i, j, :] = xy
          traj_pred_gt_mask[i, j] = True

      for j, _ in enumerate(config.scene_grids):

        this_grid_label = np.zeros([N], dtype='int32')
        this_grid_target = np.zeros([N, 2], dtype='float32')
        for i in range(len(data['pred_grid_class'])):
          # last pred timestep
          this_grid_label[i] = data['pred_grid_class'][i][j, -1]
          # last pred timestep
          this_grid_target[i] = data['pred_grid_target'][i][j, -1]

        # add new label as kxk for more target loss?

        feed_dict[self.grid_pred_labels[j]] = this_grid_label
        feed_dict[self.grid_pred_targets[j]] = this_grid_target

      if config.add_activity:
        future_act = np.zeros((N, config.num_act), dtype='uint8')
        # for experiment, training activity detection model

        for i in range(len(data['future_activity_onehot'])):
          future_act[i, :] = data['future_activity_onehot'][i]

        feed_dict[self.future_act_label] = future_act

    # needed since it is in tf.conf, but all zero in testing
    feed_dict[self.traj_class_gt] = np.zeros((N), dtype='int32')
    if config.multi_decoder and is_train:
      traj_class = np.zeros((N), dtype='int32')
      for i in range(len(data['traj_cat'])):
        traj_class[i] = data['traj_cat'][i]
      feed_dict[self.traj_class_gt] = traj_class

    return feed_dict

  def encode_other_boxes(self, person_box, other_box):
    """Encoder other boxes."""
    # get relative geometric feature
    x1, y1, x2, y2 = person_box
    xx1, yy1, xx2, yy2 = other_box

    x_m = x1
    y_m = y1
    w_m = x2 - x1
    h_m = y2 - y1

    x_n = xx1
    y_n = yy1
    w_n = xx2 - xx1
    h_n = yy2 - yy1

    return [
        math.log(max((x_m - x_n), 1e-3)/w_m),
        math.log(max((y_m - y_n), 1e-3)/h_m),
        math.log(w_n/w_m),
        math.log(h_n/h_m),
    ]


def wd_cost(regex, wd, scope):
  """Given regex to get the parameter to do regularization.

  Args:
    regex: regular expression
    wd: weight decay factor
    scope: variable scope
  Returns:
    Tensor
  """
  params = tf.trainable_variables()
  with tf.name_scope(scope):
    costs = []
    for p in params:
      para_name = p.op.name
      if re.search(regex, para_name):
        regloss = tf.multiply(tf.nn.l2_loss(p), wd, name='%s/wd' % p.op.name)
        assert regloss.dtype.is_floating, regloss
        if regloss.dtype != tf.float32:
          regloss = tf.cast(regloss, tf.float32)
        costs.append(regloss)

    return costs


def reconstruct(tensor, ref, keep):
  """Reverse the flatten function.

  Args:
    tensor: the tensor to operate on
    ref: reference tensor to get original shape
    keep: index of dim to keep

  Returns:
    Reconstructed tensor
  """
  ref_shape = ref.get_shape().as_list()
  tensor_shape = tensor.get_shape().as_list()
  ref_stop = len(ref_shape) - keep
  tensor_start = len(tensor_shape) - keep
  pre_shape = [ref_shape[i] or tf.shape(ref)[i] for i in range(ref_stop)]
  keep_shape = [tensor_shape[i] or tf.shape(tensor)[i]
                for i in range(tensor_start, len(tensor_shape))]
  # keep_shape = tensor.get_shape().as_list()[-keep:]
  target_shape = pre_shape + keep_shape
  out = tf.reshape(tensor, target_shape)
  return out


def flatten(tensor, keep):
  """Flatten a tensor.

  keep how many dimension in the end, so final rank is keep + 1
  [N,M,JI,JXP,dim] -> [N*M*JI,JXP,dim]

  Args:
    tensor: the tensor to operate on
    keep: index of dim to keep

  Returns:
    Flattened tensor
  """
  # get the shape
  fixed_shape = tensor.get_shape().as_list()  # [N, JQ, di] # [N, M, JX, di]
  # len([N, JQ, di]) - 2 = 1 # len([N, M, JX, di] ) - 2 = 2
  start = len(fixed_shape) - keep
  # each num in the [] will a*b*c*d...
  # so [0] -> just N here for left
  # for [N, M, JX, di] , left is N*M
  left = functools.reduce(operator.mul, [fixed_shape[i] or tf.shape(tensor)[i]
                               for i in range(start)])
  # [N, JQ,di]
  # [N*M, JX, di]
  out_shape = [left] + [fixed_shape[i] or tf.shape(tensor)[i]
                        for i in range(start, len(fixed_shape))]
  # reshape
  flat = tf.reshape(tensor, out_shape)
  return flat


def conv2d(x, out_channel, kernel, padding='SAME', stride=1,
           activation=tf.identity, add_bias=True, data_format='NHWC',
           w_init=None, scope='conv'):
  """Convolutional layer."""
  with tf.variable_scope(scope):
    in_shape = x.get_shape().as_list()

    channel_axis = 3 if data_format == 'NHWC' else 1
    in_channel = in_shape[channel_axis]

    assert in_channel is not None

    kernel_shape = [kernel, kernel]

    filter_shape = kernel_shape + [in_channel, out_channel]

    if data_format == 'NHWC':
      stride = [1, stride, stride, 1]
    else:
      stride = [1, 1, stride, stride]

    if w_init is None:
      w_init = tf.variance_scaling_initializer(scale=2.0)
    # common weight tensor, so pylint: disable=g-bad-name
    W = tf.get_variable('W', filter_shape, initializer=w_init)

    conv = tf.nn.conv2d(x, W, stride, padding, data_format=data_format)

    if add_bias:
      b_init = tf.constant_initializer()
      b = tf.get_variable('b', [out_channel], initializer=b_init)
      conv = tf.nn.bias_add(conv, b, data_format=data_format)

    ret = activation(conv, name='output')

  return ret


def softmax(logits, scope=None):
  """a flatten and reconstruct version of softmax."""
  with tf.name_scope(scope or 'softmax'):
    flat_logits = flatten(logits, 1)
    flat_out = tf.nn.softmax(flat_logits)
    out = reconstruct(flat_out, logits, 1)
    return out


def softsel(target, logits, use_sigmoid=False, scope=None):
  """Apply attention weights."""

  with tf.variable_scope(scope or 'softsel'):  # no new variable tho
    if use_sigmoid:
      a = tf.nn.sigmoid(logits)
    else:
      a = softmax(logits)  # shape is the same
    target_rank = len(target.get_shape().as_list())
    # [N,M,JX,JQ,2d] elem* [N,M,JX,JQ,1]
    # second last dim
    return tf.reduce_sum(tf.expand_dims(a, -1)*target, target_rank-2)


def exp_mask(val, mask):
  """Apply exponetial mask operation."""
  return tf.add(val, (1 - tf.cast(mask, 'float')) * -1e30, name='exp_mask')


def linear(x, output_size, scope, add_bias=False, wd=None, return_scope=False,
           reuse=None, activation=tf.identity, keep=1, additional_scope=None):
  """Fully-connected layer."""
  with tf.variable_scope(scope or 'xy_emb', reuse=tf.AUTO_REUSE) as this_scope:
    if additional_scope is not None:
      return linear(x, output_size, scope=additional_scope, add_bias=add_bias,
                    wd=wd, return_scope=return_scope, reuse=reuse,
                    activation=activation, keep=keep, additional_scope=None)
    # since the input here is not two rank,
    # we flat the input while keeping the last dims
    # keeping the last one dim # [N,M,JX,JQ,2d] => [N*M*JX*JQ,2d]
    flat_x = flatten(x, keep)
    # print flat_x.get_shape() # (?, 200) # wd+cwd
    bias_start = 0.0
    # need to be get_shape()[k].value
    if not isinstance(output_size, int):
      output_size = output_size.value

    def init(shape, dtype, partition_info):
      dtype = dtype
      partition_info = partition_info
      return tf.truncated_normal(shape, stddev=0.1)
    # Common weight tensor name, so pylint: disable=g-bad-name
    W = tf.get_variable('W', dtype='float', initializer=init,
                        shape=[flat_x.get_shape()[-1].value, output_size])
    flat_out = tf.matmul(flat_x, W)
    if add_bias:
      # disable=unused-argument
      def init_b(shape, dtype, partition_info):
        dtype = dtype
        partition_info = partition_info
        return tf.constant(bias_start, shape=shape)

      bias = tf.get_variable(
          'b', dtype='float', initializer=init_b, shape=[output_size])
      flat_out += bias

    flat_out = activation(flat_out)

    out = reconstruct(flat_out, x, keep)
    if return_scope:
      return out, this_scope
    else:
      return out


def focal_attention(query, context, use_sigmoid=False, scope=None):
  """Focal attention layer.

  Args:
    query : [N, dim1]
    context: [N, num_channel, T, dim2]
    use_sigmoid: use sigmoid instead of softmax
    scope: variable scope

  Returns:
    Tensor
  """
  with tf.variable_scope(scope or 'attention', reuse=tf.AUTO_REUSE):
    # Tensor dimensions, so pylint: disable=g-bad-name
    _, d = query.get_shape().as_list()
    _, K, _, d2 = context.get_shape().as_list()
    assert d == d2

    T = tf.shape(context)[2]

    # [N,d] -> [N,K,T,d]
    query_aug = tf.tile(tf.expand_dims(
        tf.expand_dims(query, 1), 1), [1, K, T, 1])

    # cosine simi
    query_aug_norm = tf.nn.l2_normalize(query_aug, -1)
    context_norm = tf.nn.l2_normalize(context, -1)
    # [N, K, T]
    a_logits = tf.reduce_sum(tf.multiply(query_aug_norm, context_norm), 3)

    a_logits_maxed = tf.reduce_max(a_logits, 2)  # [N,K]

    attended_context = softsel(softsel(context, a_logits,
                                       use_sigmoid=use_sigmoid), a_logits_maxed,
                               use_sigmoid=use_sigmoid)

    return attended_context


def concat_states(state_tuples, axis):
  """Concat LSTM states."""
  return tf.nn.rnn_cell.LSTMStateTuple(c=tf.concat([s.c for s in state_tuples],
                                                   axis=axis),
                                       h=tf.concat([s.h for s in state_tuples],
                                                   axis=axis))


class Trainer(object):
  """Trainer class for model."""

  def __init__(self, model, config):
    self.config = config
    self.model = model  # this is an model instance

    self.global_step = model.global_step

    learning_rate = config.init_lr

    if config.learning_rate_decay is not None:
      decay_steps = int(config.train_num_examples /
                        config.batch_size * config.num_epoch_per_decay)

      learning_rate = tf.train.exponential_decay(
          config.init_lr,
          self.global_step,
          decay_steps,  # decay every k samples used in training
          config.learning_rate_decay,
          staircase=True)

    if config.optimizer == 'momentum':
      opt_emb = tf.train.MomentumOptimizer(
          learning_rate*config.emb_lr, momentum=0.9)
      opt_rest = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    elif config.optimizer == 'adadelta':
      opt_emb = tf.train.AdadeltaOptimizer(learning_rate*config.emb_lr)
      opt_rest = tf.train.AdadeltaOptimizer(learning_rate)
    elif config.optimizer == 'adam':
      opt_emb = tf.train.AdamOptimizer(learning_rate*config.emb_lr)
      opt_rest = tf.train.AdamOptimizer(learning_rate)
    else:
      raise Exception('Optimizer not implemented')

    # losses
    self.xyloss = model.xyloss
    self.loss = model.loss  # get the loss funcion

    # valist for embding layer
    var_emb = [var for var in tf.trainable_variables()
               if 'emb' in var.name]
    var_rest = [var for var in tf.trainable_variables()
                if 'emb' not in var.name]

    # for training, we get the gradients first, then apply them
    self.grads = tf.gradients(self.loss, var_emb+var_rest)

    if config.clip_gradient_norm is not None:
      # pylint: disable=g-long-ternary
      self.grads = [grad if grad is None else
                    tf.clip_by_value(grad, -1*config.clip_gradient_norm,
                                     config.clip_gradient_norm)
                    for grad in self.grads]

    grads_emb = self.grads[:len(var_emb)]
    grads_rest = self.grads[len(var_emb):]

    train_emb = opt_emb.apply_gradients(zip(grads_emb, var_emb))
    train_rest = opt_rest.apply_gradients(
        zip(grads_rest, var_rest), global_step=self.global_step)
    self.train_op = tf.group(train_emb, train_rest)

  def step(self, sess, batch):
    """One training step."""
    config = self.config
    # idxs is a tuple (23,123,33..) index for sample
    _, batch_data = batch
    feed_dict = self.model.get_feed_dict(batch_data, is_train=True)
    act_loss = -1
    grid_loss = -1
    traj_class_loss = -1
    inputs = [self.loss, self.train_op, self.xyloss]
    num_out = 3
    if config.add_activity:
      inputs += [self.model.activity_loss]
      num_out += 1
    if config.multi_decoder:
      inputs += [self.model.traj_class_loss]
      num_out += 1
    inputs += self.model.grid_loss

    outputs = sess.run(inputs, feed_dict=feed_dict)

    loss, train_op, xyloss = outputs[:3]

    if config.add_activity:
      act_loss = outputs[3]

    if config.multi_decoder:
      if config.add_activity:
        traj_class_loss = outputs[4]
      else:
        traj_class_loss = outputs[3]

    grid_loss = outputs[num_out:]

    return loss, train_op, xyloss, act_loss, traj_class_loss, grid_loss


class Tester(object):
  """Tester for model."""

  def __init__(self, model, config, sess=None):
    self.config = config
    self.model = model
    self.traj_pred_out = self.model.traj_pred_out
    self.grid_pred_class = self.model.grid_class_logits
    self.sess = sess
    if config.add_activity:
      self.future_act_logits = self.model.future_act_logits

    if config.multi_decoder:
      self.traj_class_logits = self.model.traj_class_logits
      self.traj_outs = self.model.traj_pred_outs

  def step(self, sess, batch):
    """One inferencing step."""
    config = self.config
    # give one batch of Dataset, use model to get the result,
    _, batch_data = batch
    feed_dict = self.model.get_feed_dict(batch_data, is_train=False)

    future_act, grid_pred_1, grid_pred_2, traj_class_logits, traj_outs = \
        None, None, None, None, None

    inputs = [self.traj_pred_out]

    num_out = 1
    if config.add_activity:
      inputs += [self.future_act_logits]
      num_out += 1

    if config.multi_decoder:
      inputs += [self.traj_class_logits, self.traj_outs]
      num_out += 2

    inputs += self.grid_pred_class

    outputs = sess.run(inputs, feed_dict=feed_dict)

    pred_out = outputs[0]

    if config.add_activity:
      future_act = outputs[1]
    if config.multi_decoder:
      if not config.add_activity:
        traj_class_logits = outputs[1]
        traj_outs = outputs[2]
      else:
        traj_class_logits = outputs[2]
        traj_outs = outputs[3]

    grid_pred_1, grid_pred_2 = outputs[num_out:]

    return pred_out, future_act, grid_pred_1, grid_pred_2, traj_class_logits, \
        traj_outs
