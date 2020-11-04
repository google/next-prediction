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

"""Preprocess annotations for training and testing.


See README for running instructions and
download_*.sh for downloading annotations.
"""


import argparse
#import cPickle as pickle
import pickle
import glob
import json
import os
import sys
import numpy as np
from tqdm import tqdm

import utils

parser = argparse.ArgumentParser()
parser.add_argument("traj_path", help="Path to the processed trajectory files")
parser.add_argument("output_path", help="Path to put the preprocessed files")
parser.add_argument("--obs_len", default=8, type=int)
parser.add_argument("--pred_len", default=12, type=int)
parser.add_argument("--min_ped", default=0, type=int,
                    help="Minimal pedestrian in the a frame "
                         "to be considered valid datapoint"
                         ". Set 1 for ETH/UCY experiment "
                         "to be consistent with Social GAN.")

# Input features/annotations
parser.add_argument("--add_kp", action="store_true")
parser.add_argument("--kp_path", default=None)

parser.add_argument("--add_person_box", action="store_true")
parser.add_argument("--person_box_path", default=None)
parser.add_argument("--person_boxkey2id_p", default=None,
                    help="For reproducing experiments,"
                    " need person_boxkey2id from previous"
                    " preprocessed files to get the same "
                    "box id so you can you the saved person feature.")

parser.add_argument("--add_other_box", action="store_true")
parser.add_argument("--other_box_path", default=None)

parser.add_argument("--add_activity", action="store_true")
parser.add_argument("--activity_path", default=None)

parser.add_argument("--add_scene", action="store_true")
parser.add_argument("--scene_feat_path", default=None)
parser.add_argument("--scene_map_path", default=None)
parser.add_argument("--scene_id2name", default=None)
parser.add_argument("--scene_h", type=int, default=36)
parser.add_argument("--scene_w", type=int, default=64)

parser.add_argument("--add_grid", action="store_true")
parser.add_argument("--video_h", type=int, default=1080)
parser.add_argument("--video_w", type=int, default=1920)

# Specially for ETH/UCY benchmark
parser.add_argument("--traj_pixel_lst", default=None,
                    help="For ETH/UCY benchmark, "
                         "need to use x,y in pixel to get grid location")
parser.add_argument("--feature_no_split", action="store_true",
                    help="There is not train/val/test"
                         " folder in the feature directory.")
parser.add_argument("--reverse_xy", action="store_true",
                    help="The trajectory file is in frameidx"
                         ", personidx, y, x.")


def main(args):
  # Compute the scene grid
  if args.add_grid:
    args.scene_grid_strides = (2, 4)
    args.num_scene_grid = len(args.scene_grid_strides)

    args.scene_grids = []
    # the following is consistent with tensorflow conv2d when given odd input
    for stride in args.scene_grid_strides:
      h, w = args.scene_h, args.scene_w
      this_h, this_w = round(h*1.0/stride), round(w*1.0/stride)
      this_h, this_w = int(this_h), int(this_w)
      args.scene_grids.append((this_h, this_w))

    # Get the center point for each scale's each grid
    args.scene_grid_centers = []
    for h, w in args.scene_grids:
      h_gap, w_gap = args.video_h*1.0/h, args.video_w*1.0/w
      centers_x = np.cumsum([w_gap for _ in range(w)]) - w_gap/2.0
      centers_y = np.cumsum([h_gap for _ in range(h)]) - h_gap/2.0
      centers_xx = np.tile(np.expand_dims(centers_x, axis=0), [h, 1])
      centers_yy = np.tile(np.expand_dims(centers_y, axis=1), [1, w])
      centers = np.stack((centers_xx, centers_yy), axis=-1)  # [H,W,2]
      args.scene_grid_centers.append(centers)

  # load alternative xy in pixels for ETH/UCY benchmark experiments
  args.traj_pixel = None
  if args.traj_pixel_lst is not None:
    args.traj_pixel = {}
    delim = "\t"
    with open(args.traj_pixel_lst, "r") as traj_pixel_lst:
      for pixel_file in traj_pixel_lst:
        pixel_file = pixel_file.strip()
        filename = os.path.splitext(os.path.basename(pixel_file))[0]
        args.traj_pixel[filename] = {}
        for line in open(pixel_file):
          fid, pid, x, y = line.strip().split(delim)
          p_key = "%d_%d" % (float(fid), float(pid))
          x = float(x)
          y = float(y)
          assert float(x) <= args.video_w, line
          assert float(y) <= args.video_h, line
          args.traj_pixel[filename][p_key] = [float(x), float(y)]

  args.seq_len = args.obs_len + args.pred_len

  if not os.path.exists(args.output_path):
    os.makedirs(args.output_path)

  # For creating the same boxid as previous experiment
  args.person_boxkey2id = None
  if args.person_boxkey2id_p is not None:
    with open(args.person_boxkey2id_p, "rb") as f:
      args.person_boxkey2id = pickle.load(f)

  prepro_each(args.traj_path, "train", os.path.join(
      args.output_path, "data_train.npz"), args)
  prepro_each(args.traj_path, "val", os.path.join(
      args.output_path, "data_val.npz"), args)
  prepro_each(args.traj_path, "test", os.path.join(
      args.output_path, "data_test.npz"), args)


def prepro_each(traj_path, split, prepro_path, args):
  """Preprocess each data split into one npz file.

  Args:
    traj_path: path to the trajectory annotation files
    split: train/val/test
    prepro_path: path to the output npz file
    args: arguments

  Returns:
    None
  """
  traj_path = os.path.join(traj_path, split)

  # traj_path each file is a video, with frameid, personid, x, y
  videos = glob.glob(os.path.join(traj_path, "*.txt"))

  delim = "\t"

  seq_len = args.seq_len
  obs_len = args.obs_len

  # collect data for all videos
  seq_list = []  # [N, seq_len, 2], N is frames*person_per_frame
  seq_list_rel = []
  num_person_in_start_frame = []

  # so we could refer to the original frame for each time step
  seq_frameidx_list = []  # [N, seq_len]
  seq_vid_list = []  # [N] ,N videoid, int
  vid2name = {}
  total_frame_used = {}

  seq_grid_class_list = []  # [N, strides, seq_len]
  seq_grid_target_list = []  # [N, strides, seq_len, 2]

  # the person traj's boxes
  box_seq_list = []  # [N, seq_len, 4]

  person_boxid_list = []  # [N,seq_len]
  person_boxid2key = {}  # frameIdx_personId ->
  person_boxkey2id = {}

  # the other boxes in the last observed frame
  # [N,1] a list of variable number of boxes
  other_box_seq_list = []
  # [N,1] # a list of variable number of boxes classes
  other_box_class_seq_list = []

  # activity annotation, currently just use the last observed frame
  # for current activity and the last predict frame for future activity
  cur_act_list = []  # [N,1] # a list of act id
  future_act_list = []  # [N,1] # a list of act id, could be empty?

  kp_num = 17  # coco style
  kp_list = []  # [N, seq_len, 17, 2]
  kp_list_rel = []

  scene_list = []  # [N, seq_len, 1] # only the frame feature id
  # will have a final scene feature of [num_frame, H, W, class]
  # scene class mask
  # save only the unique frame
  scene_feat_dict = {}  # #frame to feature
  scene_key2feati = {}
  scene_h, scene_w = args.scene_h, args.scene_w

  # load the classes that we used for scene segmantics
  if args.add_scene:
    with open(args.scene_id2name, "r") as f:
      scene_id2name = json.load(f)  # {"oldid2new":,"id2name":}
    scene_oldid2new = scene_id2name["oldid2new"]
    scene_oldid2new = {
        int(oldi): scene_oldid2new[oldi] for oldi in scene_oldid2new}
    # for background class or other class that we ignored
    #assert not scene_oldid2new.has_key(0)
    assert 0 not in scene_oldid2new
    scene_oldid2new[0] = 0
    total_scene_class = len(scene_oldid2new)
    scene_id2name = scene_id2name["id2name"]
    scene_id2name[0] = "BG"
    assert len(scene_oldid2new) == len(scene_id2name)

  # person trajectory processing part is modified from Social GAN
  # https://github.com/agrimgupta92/sgan/blob/master/sgan/data/trajectories.py
  # to keep the experimental setting the same
  for video in tqdm(videos, ascii=True):
    videoname = os.path.splitext(os.path.basename(video))[0]
    vid = len(vid2name)
    vid2name[vid] = videoname

    # load other features if necessary
    kp_feats = {}  # "frameidx_personId"
    # "frameid" -> scene_feat_file_path # load it dynamically
    scene_frameid2file = {}
    if args.add_kp:
      kp_file_path = os.path.join(args.kp_path, split, "%s.p" % videoname)
      with open(kp_file_path, "rb") as f:

        if sys.version_info.major == 2:
          # this works for py2 since the pickle is generated with py2 code
          kp_feats = pickle.load(f)
        else:
          # ugly so it is py3 compatitable
          kp_feats = pickle.load(f, encoding="bytes")
          new_kp_feats = {}
          for k in kp_feats:
            new_kp_feats[k.decode("utf-8")] = kp_feats[k]
          kp_feats = new_kp_feats

    if args.add_scene:
      # get the frameid to file name since scene is not extracted every frames
      scene_file = os.path.join(args.scene_map_path, split, "%s.p" % videoname)
      if args.feature_no_split:
        scene_file = os.path.join(args.scene_map_path, "%s.p" % videoname)
      with open(scene_file, "rb") as f:
        scene_frameid2file = pickle.load(f)
      for frameid in scene_frameid2file:
        scene_frameid2file[frameid] = os.path.join(
            args.scene_feat_path, scene_frameid2file[frameid])

    if args.add_person_box:
      person_box_path = os.path.join(
          args.person_box_path, split, "%s.p" % videoname)
      if args.feature_no_split:
        person_box_path = os.path.join(
            args.person_box_path, "%s.p" % videoname)
      with open(person_box_path, "rb") as f:
        person_boxes = pickle.load(f)

    if args.add_other_box:
      other_box_path = os.path.join(
          args.other_box_path, split, "%s.p" % videoname)
      if args.feature_no_split:
        other_box_path = os.path.join(args.other_box_path, "%s.p" % videoname)
      with open(other_box_path, "rb") as f:
        other_boxes = pickle.load(f)

    if args.add_activity:
      activity_path = os.path.join(
          args.activity_path, split, "%s.p" % videoname)
      with open(activity_path, "rb") as f:
        activities = pickle.load(f)

    # [N,4], [frame_idx, person_id,x,y]
    data = []
    with open(video, "r") as traj_file:
      for line in traj_file:
        if args.reverse_xy:
          fidx, pid, y, x = line.strip().split(delim)
        else:
          fidx, pid, x, y = line.strip().split(delim)
        data.append([fidx, pid, x, y])
    data = np.array(data, dtype="float32")

    # assuming the frameIdx is sorted in ASC
    frames = np.unique(data[:, 0]).tolist()  # all frame_idx
    frame_data = []  # [num_frame, K,4]
    for frame in frames:
      frame_data.append(data[frame == data[:, 0], :])

    for idx, frame in enumerate(frames):
      # [N, 4] # N is seq_len* person_per_frame
      # [obs_frames -> pre_frames all data]
      cur_seq_data = np.concatenate(frame_data[idx:idx + seq_len], axis=0)
      # [K] # all person Id in this sequence frames [20 frames]
      persons_in_cur_seq = np.unique(cur_seq_data[:, 1])
      num_person_in_cur_seq = len(persons_in_cur_seq)
      # [K, seq_len, 2] # x,y for all person sequence, starting at idx frame
      cur_seq = np.zeros((num_person_in_cur_seq, seq_len, 2), dtype="float32")
      # relative x,y for training
      cur_seq_rel = np.zeros((num_person_in_cur_seq, seq_len, 2),
                             dtype="float32")

      # frameid for each seq timestep
      cur_seq_frame = np.zeros((num_person_in_cur_seq, seq_len), dtype="int32")
      cur_seq_vid = np.zeros((num_person_in_cur_seq), dtype="int32")
      cur_seq_vid[:] = vid  # all this sequence is in the same video obviously

      # for grid classification and target
      if args.add_grid:
        cur_seq_grids_class = np.zeros(
            (num_person_in_cur_seq, args.num_scene_grid, seq_len),
            dtype="int32")
        cur_seq_grids_target = np.zeros(
            (num_person_in_cur_seq, args.num_scene_grid, seq_len, 2),
            dtype="float32")

      count_person = 0

      if args.add_kp:
        # absolute pixexl
        kp_feat = np.zeros((num_person_in_cur_seq, seq_len, kp_num, 2),
                           dtype="float32")
        # velocity
        kp_feat_rel = np.zeros((num_person_in_cur_seq, seq_len, kp_num, 2),
                               dtype="float32")

      if args.add_person_box:
        person_box = np.zeros((num_person_in_cur_seq, seq_len, 4),
                              dtype="float32")
        person_boxids = np.zeros((num_person_in_cur_seq, seq_len),
                                 dtype="int32")

      if args.add_other_box:
        other_box = []
        other_box_class = []

      if args.add_activity:
        cur_activity = []
        future_activity = []

      if args.add_scene:
        scene_featidx = np.zeros((num_person_in_cur_seq, seq_len, 1),
                                 dtype="int")
        # this frame to the rest frame for all the persons should be the same
        frame_idxs = frames[idx:idx+seq_len]

        for i, frame_idx in enumerate(frame_idxs):
          # key = "%s_%d"%(videoname,frame_idx)
          # so we only load unique feat once
          key = scene_frameid2file[frame_idx]

          if key not in scene_key2feati:
            feati = len(scene_feat_dict.keys())
            # get the feature new i
            # (H,W)
            scene_feat_dict[key] = np.load(key)
            scene_key2feati[key] = feati

          else:
            feati = scene_key2feati[key]

          scene_featidx[:, i, :] = feati

      for person_id in persons_in_cur_seq:
        # traverse all person starting from idx frames for 20 frames

        # [M, 4]
        cur_person_seq = cur_seq_data[cur_seq_data[:, 1] == person_id, :]

        if len(cur_person_seq) != seq_len:
          # skipping the sequence not fully cover in this frames
          continue

        # [seq_len,2]
        cur_person_seq = cur_person_seq[:, 2:]
        cur_person_seq_rel = np.zeros_like(cur_person_seq)

        # first frame is zeros x,y
        cur_person_seq_rel[1:, :] = cur_person_seq[1:, :] - \
            cur_person_seq[:-1, :]

        cur_seq[count_person, :, :] = cur_person_seq
        cur_seq_rel[count_person, :, :] = cur_person_seq_rel

        frame_idxs = frames[idx:idx+seq_len]

        # get the grid classification
        if args.add_grid:
          this_cur_person_seq = cur_person_seq
          if args.traj_pixel is not None:  # use alternate xy
            this_cur_person_seq = np.zeros_like(cur_person_seq)
            for this_i, frame_idx in enumerate(frame_idxs):
              key = "%d_%d" % (frame_idx, person_id)
              # print key
              x, y = args.traj_pixel[videoname][key]
              this_cur_person_seq[this_i, :] = [x, y]

          # get the grid classification label based on (x,y)
          # grid centers: [H,W,2]
          for i, (center, (h, w)) in enumerate(zip(
              args.scene_grid_centers, args.scene_grids)):

            # grid classification
            h_gap, w_gap = args.video_h*1.0/h, args.video_w*1.0/w
            x_indexes = np.ceil(this_cur_person_seq[:, 0] / w_gap)  # [seq_len]

            y_indexes = np.ceil(this_cur_person_seq[:, 1] / h_gap)  # [seq_len]
            x_indexes = np.asarray(x_indexes, dtype="int")
            y_indexes = np.asarray(y_indexes, dtype="int")

            # ceil(0.0) = 0.0, we need
            x_indexes[x_indexes == 0] = 1
            y_indexes[y_indexes == 0] = 1
            x_indexes = x_indexes - 1
            y_indexes = y_indexes - 1

            one_hot = np.zeros((seq_len, h, w), dtype="uint8")
            one_hot[range(seq_len), y_indexes, x_indexes] = 1
            one_hot_flat = one_hot.reshape((seq_len, -1))  # [seq_len,h*w]
            classes = np.argmax(one_hot_flat, axis=1)  # [seq_len]
            cur_seq_grids_class[count_person, i, :] = classes

            # grid regression
            # tile current person seq xy
            cur_person_seq_tile = np.tile(np.expand_dims(np.expand_dims(
                this_cur_person_seq, axis=1), axis=1), [1, h, w, 1])
            # tile center [seq_len,h,w,2]
            center_tile = np.tile(np.expand_dims(
                center, axis=0), [seq_len, 1, 1, 1])
            # grid_center + target -> actual xy
            all_target = cur_person_seq_tile - center_tile  # [seq_len,h,w,2]
            # only save the one grid
            cur_seq_grids_target[count_person, i, :, :] = \
                all_target[one_hot.astype("bool"), :]

        # record the frame
        cur_seq_frame[count_person, :] = frame_idxs

        # kp feature
        if args.add_kp:
          # get the kp feature from starting frame to seq_len frame
          for i, frame_idx in enumerate(frame_idxs):
            key = "%d_%d" % (frame_idx, person_id)
            # ignore the kp logits
            kp_feat[count_person, i, :, :] = kp_feats[key][:, :2]

          kp_feat_rel[count_person, 1:, :, :] = \
              kp_feat[count_person, 1:, :, :] - kp_feat[count_person, :-1, :, :]

        if args.add_person_box:
          for i, frame_idx in enumerate(frame_idxs):
            key = "%d_%d" % (frame_idx, person_id)
            person_box[count_person, i, :] = person_boxes[key]

            # save this person key to an id
            key = "%s_%s" % (videoname, key)

            if key not in person_boxkey2id:
              if args.person_boxkey2id is not None:
                # use the boxid from previous preprocessed files
                # to reproduce experiments
                prev_boxid = args.person_boxkey2id[split][key]
                person_boxkey2id[key] = prev_boxid
                person_boxid2key[prev_boxid] = key
              else:
                new_person_boxid = len(person_boxkey2id)
                person_boxkey2id[key] = new_person_boxid
                person_boxid2key[new_person_boxid] = key

            person_boxid = person_boxkey2id[key]
            person_boxids[count_person, i] = person_boxid

        if args.add_other_box:
          this_other_box = []
          this_other_box_class = []
          for i, frame_idx in enumerate(frame_idxs):
            key = "%d_%d" % (frame_idx, person_id)
            # a list of [4]
            this_other_box.append(other_boxes[key][0])
            # a list of [1]
            this_other_box_class.append(other_boxes[key][1])

          other_box.append(this_other_box)
          other_box_class.append(this_other_box_class)

        if args.add_activity:
          virat_timestep2fps = 12
          this_cur_activity = []
          this_future_activity = []
          for i, frame_idx in enumerate(frame_idxs):
            key = "%d_%d" % (frame_idx, person_id)
            # a list of [1], act id;
            # should not be empty, should have filled with BG class
            current_actid_list, _, future_actid_list, _ = activities[key]
            assert current_actid_list, current_actid_list
            assert future_actid_list, future_actid_list

            future_frame = int(args.pred_len * virat_timestep2fps)
            future_actid_list_filtered = filter_future_act(
                activities[key], future_frame)

            # overlapping act?
            current_actid_list = list(set(current_actid_list))
            this_cur_activity.append(current_actid_list)

            future_actid_list_filtered = list(set(future_actid_list_filtered))
            this_future_activity.append(future_actid_list_filtered)

          cur_activity.append(this_cur_activity)
          future_activity.append(this_future_activity)

        count_person += 1

      # save the data
      if count_person <= args.min_ped:
        continue
      num_person_in_start_frame.append(count_person)
      # only count_person data is preserved
      seq_list.append(cur_seq[:count_person])
      seq_list_rel.append(cur_seq_rel[:count_person])

      seq_frameidx_list.append(cur_seq_frame[:count_person])
      seq_vid_list.append(cur_seq_vid[:count_person])

      for one in cur_seq_frame[:count_person]:
        for frameidx in one:
          total_frame_used[(videoname, frameidx)] = 1

      # other features
      if args.add_kp:
        kp_list.append(kp_feat[:count_person])
        kp_list_rel.append(kp_feat_rel[:count_person])

      if args.add_scene:
        scene_list.append(scene_featidx[:count_person])

      if args.add_grid:
        seq_grid_class_list.append(cur_seq_grids_class[:count_person])
        seq_grid_target_list.append(cur_seq_grids_target[:count_person])

      if args.add_person_box:
        box_seq_list.append(person_box[:count_person])
        person_boxid_list.append(person_boxids[:count_person])

      if args.add_other_box:
        # other_box: [count_person, seqlen, K, 4] but python list,
        # K is variable length
        other_box_seq_list.extend(other_box)
        other_box_class_seq_list.extend(other_box_class)

      if args.add_activity:
        # [count_person, seqlen, K] K is variable length
        cur_act_list.extend(cur_activity)
        future_act_list.extend(future_activity)

  num_seq = len(seq_list)  # total number of frames across all videos
  # [N*K, seq_len, 2]
  # N is num_frame for each video, K is num_person in each frame
  seq_list = np.concatenate(seq_list, axis=0)
  seq_list_rel = np.concatenate(seq_list_rel, axis=0)
  seq_frameidx_list = np.concatenate(seq_frameidx_list, axis=0)
  seq_vid_list = np.concatenate(seq_vid_list, axis=0)

  print("total frames %s, seq_list shape:%s, total unique frame used:%s" %
        (num_seq, seq_list.shape, len(total_frame_used)))

  # we get the obs traj and pred_traj
  # [N*K, obs_len, 2]
  # [N*K, pred_len, 2]
  obs_traj = seq_list[:, :obs_len, :]
  pred_traj = seq_list[:, obs_len:, :]

  obs_traj_rel = seq_list_rel[:, :obs_len, :]
  pred_traj_rel = seq_list_rel[:, obs_len:, :]

  # only save the obs_frames
  obs_frameidx = seq_frameidx_list[:, :obs_len]
  obs_vid = seq_vid_list[:]

  # the starting idx for each frame in the N*K list,
  # [num_frame, 2]
  cum_start_idx = [0] + np.cumsum(num_person_in_start_frame).tolist()
  seq_start_end = np.array([
      (start, end) for start, end in zip(cum_start_idx, cum_start_idx[1:])
  ], dtype="int")

  # save the data
  data = {
      "obs_traj": obs_traj,
      "obs_traj_rel": obs_traj_rel,
      "pred_traj": pred_traj,
      "pred_traj_rel": pred_traj_rel,
      "seq_start_end": seq_start_end,
      "obs_frameidx": obs_frameidx,
      "obs_vid": obs_vid,
      "vid2name": vid2name,
  }

  if args.add_kp:
    # [N*K, seq_len, 17, 2]
    kp_list = np.concatenate(kp_list, axis=0)
    kp_list_rel = np.concatenate(kp_list_rel, axis=0)

    obs_kp = kp_list[:, :obs_len, :, :]
    pred_kp = kp_list[:, obs_len:, :, :]  # for visualization
    obs_kp_rel = kp_list_rel[:, :obs_len, :, :]

    data.update({
        "obs_kp": obs_kp,
        "obs_kp_rel": obs_kp_rel,
        "pred_kp": pred_kp,
    })

  if args.add_person_box:
    box_seq_list = np.concatenate(box_seq_list, axis=0)

    person_boxid_list = np.concatenate(
        person_boxid_list, axis=0)  # [N,seq_len]

    obs_box = box_seq_list[:, :obs_len, :]

    obs_boxid = person_boxid_list[:, :obs_len]
    data.update({
        "obs_box": obs_box,
        "obs_boxid": obs_boxid,
        "person_boxkey2id": person_boxkey2id,
        "person_boxid2key": person_boxid2key,
    })
    print("total unique person box:%s" % len(person_boxid2key))

  if args.add_other_box:
    # other_box_seq_list a list  [N, count_person, seqlen, K, 4],
    # K is variable length
    other_box_seq_list = np.asarray(
        other_box_seq_list)  # [N*K,seqlen] list type?
    other_box_class_seq_list = np.asarray(
        other_box_class_seq_list)  # [N*K,seqlen] list type?

    data.update({
        "obs_other_box": other_box_seq_list[:, :obs_len],
        "obs_other_box_class": other_box_class_seq_list[:, :obs_len],
    })

  if args.add_activity:
    cur_act_list = np.asarray(cur_act_list)
    future_act_list = np.asarray(future_act_list)

    # current/future activity *at* the last observed frame
    cur_activity = cur_act_list[:, obs_len-1]
    future_activity = future_act_list[:, obs_len-1]

    # check some stats
    cur_bg_only = []
    cur_act_count = []
    fu_act_count = []
    fu_bg_only = []
    for curs, fus in zip(cur_activity, future_activity):
      cur_act_count.append(len(curs))
      fu_act_count.append(len(fus))

      first_cur = curs[0]
      first_fu = fus[0]
      if first_cur == 0:
        assert len(curs) == 1, (curs, fus)
        cur_bg_only.append(1)
      else:
        cur_bg_only.append(0)
      if first_fu == 0:
        assert len(fus) == 1, (curs, fus)
        fu_bg_only.append(1)
      else:
        fu_bg_only.append(0)

    move_cat = [
        utils.activity2id["activity_walking"],
        utils.activity2id["activity_running"],
        utils.activity2id["Riding"]
    ]

    traj_cat = np.zeros((len(cur_activity)), dtype="uint8")
    count_move = 0
    for i in range(len(cur_activity)):
      cur_acts = cur_activity[i]
      move = False
      for actid in cur_acts:
        if actid in move_cat:
          move = True
          count_move += 1
          break
      traj_cat[i] = int(move)  # 0 -> static, 1 -> move

    data.update({
        "cur_activity": cur_activity,
        "future_activity": future_activity,
        "traj_cat": traj_cat  # 0, static, 1 is move
    })

  if args.add_grid:
    seq_grid_class_list = np.concatenate(seq_grid_class_list, axis=0)
    seq_grid_target_list = np.concatenate(seq_grid_target_list, axis=0)

    obs_seq_grid_class = seq_grid_class_list[:, :, :obs_len]
    obs_seq_grid_target = seq_grid_target_list[:, :, :obs_len]
    pred_seq_grid_class = seq_grid_class_list[:, :, obs_len:]
    pred_seq_grid_target = seq_grid_target_list[:, :, obs_len:]
    data.update({
        "video_wh": (args.video_w, args.video_h),
        "scene_grid_strides": args.scene_grid_strides,
        "obs_grid_class": obs_seq_grid_class,
        "obs_grid_target": obs_seq_grid_target,
        "pred_grid_class": pred_seq_grid_class,
        "pred_grid_target": pred_seq_grid_target,
    })
    for i, center in enumerate(args.scene_grid_centers):
      data.update({
          ("grid_center_%d" % i): center,
      })

  if args.add_scene:
    # the ids to the feature
    # [N*K, seq_len, 1]
    scene_list = np.concatenate(scene_list, axis=0)
    obs_scene = scene_list[:, :obs_len, :]
    pred_scene = scene_list[:, obs_len:, :]

    # stack all the feature into one big matrix
    # all frames in all videos # now it is jus the unique feature frame
    total_frames = len(scene_feat_dict)
    scene_feat_final_shape = (total_frames, scene_h,
                              scene_w, total_scene_class)
    # [6804, 288, 513, 41]
    print("initilizing big scene feature matrix : %s.." % list(
        scene_feat_final_shape))
    # each class will be a mask
    scene_feat_final = np.zeros(scene_feat_final_shape, dtype="uint8")
    print("cool.")
    print("making mask scene feature...")
    for key in tqdm(scene_feat_dict, ascii=True):
      feati = scene_key2feati[key]
      scene_feat = scene_feat_dict[key]  # [H,W]
      # transform classid first
      new_scene_feat = np.zeros_like(scene_feat)  # zero for background class
      for i in range(scene_h):
        for j in range(scene_w):
          # rest is ignored and all put into background
          #if scene_oldid2new.has_key(scene_feat[i, j]):
          if scene_feat[i, j] in scene_oldid2new:
            new_scene_feat[i, j] = scene_oldid2new[scene_feat[i, j]]
      # transform to masks
      this_scene_feat = np.zeros(
          (scene_h, scene_w, total_scene_class), dtype="uint8")
      # so we use the H,W to index the mask feat
      # generate the index first
      h_indexes = np.repeat(np.arange(scene_h), scene_w).reshape(
          (scene_h, scene_w))
      w_indexes = np.tile(np.arange(scene_w), scene_h).reshape(
          (scene_h, scene_w))
      this_scene_feat[h_indexes, w_indexes, new_scene_feat] = 1

      scene_feat_final[feati, :, :, :] = this_scene_feat
      del this_scene_feat
      del new_scene_feat

    data.update({
        "obs_scene": obs_scene,
        "pred_scene": pred_scene,
        "scene_feat": scene_feat_final,
    })

  np.savez(prepro_path, **data)


def filter_future_act(acts, future_frame):
  """Get future activity ids.

  future activity from the data is all the future activity,
  here we filter only the activity in pred_len,
  also add the current activity that is still not finished


  Args:
    acts: a tuple of (current_actid_list, current_dist_list,
        future_actid_list, future_dist_list)
    future_frame: how many frame until the future

  Returns:
    future activity ids
  """

  current_actid_list, current_dist_list, \
      future_actid_list, future_dist_list = acts

  # leave the actid happens at future_frame
  actids = []
  for act_id, dist_to_finish in zip(current_actid_list, current_dist_list):
    if act_id == 0:
      continue
    if future_frame <= dist_to_finish:
      actids.append(act_id)

  for act_id, dist_to_start in zip(future_actid_list, future_dist_list):
    if act_id == 0:
      continue
    if future_frame >= dist_to_start:
      actids.append(act_id)

  if not actids:
    actids.append(0)  # BG class

  return actids


if __name__ == "__main__":
  arguments = parser.parse_args()
  main(arguments)
