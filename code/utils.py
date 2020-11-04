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

"""Utility functions and classes."""

import collections
import itertools
import math
import operator
import os
import random
import sys
import numpy as np
import tensorflow as tf
import tqdm

__all__ = ["activity2id", "object2id",
           "initialize", "read_data"]

activity2id = {
    "BG": 0,  # background
    "activity_walking": 1,
    "activity_standing": 2,
    "activity_carrying": 3,
    "activity_gesturing": 4,
    "Closing": 5,
    "Opening": 6,
    "Interacts": 7,
    "Exiting": 8,
    "Entering": 9,
    "Talking": 10,
    "Transport_HeavyCarry": 11,
    "Unloading": 12,
    "Pull": 13,
    "Loading": 14,
    "Open_Trunk": 15,
    "Closing_Trunk": 16,
    "Riding": 17,
    "specialized_texting_phone": 18,
    "Person_Person_Interaction": 19,
    "specialized_talking_phone": 20,
    "activity_running": 21,
    "PickUp": 22,
    "specialized_using_tool": 23,
    "SetDown": 24,
    "activity_crouching": 25,
    "activity_sitting": 26,
    "Object_Transfer": 27,
    "Push": 28,
    "PickUp_Person_Vehicle": 29,
}

object2id = {
    "Person": 0,
    "Vehicle": 1,
    "Parking_Meter": 2,
    "Construction_Barrier": 3,
    "Door": 4,
    "Push_Pulled_Object": 5,
    "Construction_Vehicle": 6,
    "Prop": 7,
    "Bike": 8,
    "Dumpster": 9,
}


def process_args(args):
  """Process arguments.

  Model will be in outbasepath/modelname/runId/save

  Args:
    args: arguments.

  Returns:
    Edited arguments.
  """

  def mkdir(path):
    if not os.path.exists(path):
      os.makedirs(path)

  if args.activation_func == "relu":
    args.activation_func = tf.nn.relu
  elif args.activation_func == "tanh":
    args.activation_func = tf.nn.tanh
  elif args.activation_func == "lrelu":
    args.activation_func = tf.nn.leaky_relu
  else:
    print("unrecognied activation function, using relu...")
    args.activation_func = tf.nn.relu

  args.seq_len = args.obs_len + args.pred_len

  args.outpath = os.path.join(
      args.outbasepath, args.modelname, str(args.runId).zfill(2))
  mkdir(args.outpath)

  args.save_dir = os.path.join(args.outpath, "save")
  mkdir(args.save_dir)
  args.save_dir_model = os.path.join(args.save_dir, "save")
  args.save_dir_best = os.path.join(args.outpath, "best")
  mkdir(args.save_dir_best)
  args.save_dir_best_model = os.path.join(args.save_dir_best, "save-best")

  args.write_self_sum = True
  args.self_summary_path = os.path.join(args.outpath, "train_sum.txt")

  args.record_val_perf = True
  args.val_perf_path = os.path.join(args.outpath, "val_perf.p")

  # assert os.path.exists(args.frame_path)
  # args.resnet_num_block = [3,4,23,3] # resnet 101
  assert os.path.exists(args.person_feat_path)

  args.object2id = object2id
  args.num_box_class = len(args.object2id)

  # categories of traj
  if args.is_actev:
    args.virat_mov_actids = [
        activity2id["activity_walking"],
        activity2id["activity_running"],
        activity2id["Riding"],
    ]
    args.traj_cats = [
        ["static", 0],
        ["mov", 1],
    ]
    args.scenes = ["0000", "0002", "0400", "0401", "0500"]

  args.num_act = len(activity2id.keys())  # include the BG class

  # has to be 2,4 to match the scene CNN strides
  args.scene_grid_strides = (2, 4)
  args.scene_grids = []
  for stride in args.scene_grid_strides:
    h, w = args.scene_h, args.scene_w
    this_h, this_w = round(h*1.0/stride), round(w*1.0/stride)
    this_h, this_w = int(this_h), int(this_w)
    args.scene_grids.append((this_h, this_w))

  if args.load_best:
    args.load = True
  if args.load_from is not None:
    args.load = True

  # if test, has to load
  if not args.is_train:
    args.load = True
    args.num_epochs = 1
    args.keep_prob = 1.0

  args.activity2id = activity2id
  return args


def initialize(load, load_best, args, sess):
  """Initialize graph with given model weights.

  Args:
    load: boolean, whether to load model weights
    load_best: whether to load from best model path
    args: arguments
    sess: tf.Session() instance

  Returns:
    None
  """

  tf.global_variables_initializer().run()

  if load:
    print("restoring model...")
    allvars = tf.global_variables()
    allvars = [var for var in allvars if "global_step" not in var.name]
    restore_vars = allvars
    opts = ["Adam", "beta1_power", "beta2_power",
            "Adam_1", "Adadelta_1", "Adadelta", "Momentum"]
    restore_vars = [var for var in restore_vars \
        if var.name.split(":")[0].split("/")[-1] not in opts]

    saver = tf.train.Saver(restore_vars, max_to_keep=5)

    load_from = None

    if args.load_from is not None:
      load_from = args.load_from
    else:
      if load_best:
        load_from = args.save_dir_best
      else:
        load_from = args.save_dir

    ckpt = tf.train.get_checkpoint_state(load_from)
    if ckpt and ckpt.model_checkpoint_path:
      loadpath = ckpt.model_checkpoint_path

      saver.restore(sess, loadpath)
      print("Model:")
      print("\tloaded %s" % loadpath)
      print("")
    else:
      if os.path.exists(load_from):
        if load_from.endswith(".ckpt"):
          # load_from should be a single .ckpt file
          saver.restore(sess, load_from)
        else:
          print("Not recognized model type:%s" % load_from)
          sys.exit()
      else:
        print("Model not exists")
        sys.exit()
    print("done.")


def read_data(args, data_type):
  """Read propocessed data into memory for experiments.

  Args:
    args: Arguments
    data_type: train/val/test

  Returns:
    Dataset instance
  """

  def get_traj_cat(cur_acts, traj_cats):
    """Get trajectory categories for virat/actev dataset experiments."""

    def is_in(l1, l2):
      """Check whether any of l1"s item is in l2."""
      for i in l1:
        if i in l2:
          return True
      return False

    # 1 is moving act, 0 is static
    act_cat = int(is_in(cur_acts, args.virat_mov_actids))
    i = -1
    for i, (_, actid) in enumerate(traj_cats):
      if actid == act_cat:
        return i
    # something is wrong
    assert i >= 0

  data_path = os.path.join(args.prepropath, "data_%s.npz" % data_type)

  data = dict(np.load(data_path, allow_pickle=True))

  # save some shared feature first

  shared = {}
  shares = ["scene_feat", "video_wh", "scene_grid_strides",
            "vid2name", "person_boxkey2id", "person_boxid2key"]

  excludes = ["seq_start_end"]

  if "video_wh" in data:
    args.box_img_w, args.box_img_h = data["video_wh"]
  else:
    args.box_img_w, args.box_img_h = 1920, 1080

  for i in range(len(args.scene_grid_strides)):
    shares.append("grid_center_%d" % i)

  for key in data:
    if key in shares:
      if not data[key].shape:
        shared[key] = data[key].item()
      else:
        shared[key] = data[key]

  newdata = {}
  for key in data:
    if key not in excludes+shares:
      newdata[key] = data[key]
  data = newdata

  if args.add_activity:  # transform activity ids to a one hot feature
    cur_acts = []
    future_acts = []  # [N, num_act]
    num_act = args.num_act
    for i in range(len(data["cur_activity"])):  # super fast
      cur_actids = data["cur_activity"][i]
      future_actids = data["future_activity"][i]

      cur_act = np.zeros((num_act), dtype="uint8")
      future_act = np.zeros((num_act), dtype="uint8")

      for actid in cur_actids:
        cur_act[actid] = 1
      for actid in future_actids:
        future_act[actid] = 1

      cur_acts.append(cur_act)
      future_acts.append(future_act)

    data["cur_activity_onehot"] = cur_acts
    data["future_activity_onehot"] = future_acts

  assert len(shared["scene_grid_strides"]) == len(args.scene_grid_strides)
  assert shared["scene_grid_strides"][0] == args.scene_grid_strides[0]

  num_examples = len(data["obs_traj"])  # (input,pred)

  for key in data:
    assert len(data[key]) == num_examples, \
        (key, data[key].shape, num_examples)

  # category each trajectory for training
  if args.is_actev:

    data["trajidx2catid"] = np.zeros(
        (num_examples), dtype="uint8")  # 0~256

    boxid2key = shared["person_boxid2key"]

    trajkey2cat = {}

    data["traj_key"] = []
    cat_count = [[cat_name, 0] for cat_name, _ in args.traj_cats]
    for i in range(num_examples):
      cur_acts = data["cur_activity"][i]
      cat_id = get_traj_cat(cur_acts, args.traj_cats)
      data["trajidx2catid"][i] = cat_id

      cat_count[cat_id][1] += 1
      # videoname_frameidx_personid
      key = boxid2key[data["obs_boxid"][i][0]]
      trajkey2cat[key] = cat_id
      data["traj_key"].append(key)

    print(cat_count)
  else:
    data["traj_key"] = []
    boxid2key = shared["person_boxid2key"]
    for i in range(num_examples):
      # videoname_frameidx_personid
      key = boxid2key[data["obs_boxid"][i][0]]
      data["traj_key"].append(key)

  print("loaded %s data points for %s" % (num_examples, data_type))

  return Dataset(data, data_type, shared=shared, config=args)


def get_scene(videoname_):
  """Get the scene camera from the ActEV videoname."""
  s = videoname_.split("_S_")[-1]
  s = s.split("_")[0]
  return s[:4]


def evaluate(dataset, config, sess, tester):
  """Evaluate the dataset using the tester model.

  Args:
    dataset: the Dataset instance
    config: arguments
    sess: tensorflow session
    tester: the Tester instance

  Returns:
    Evaluation results.
  """

  l2dis = []  # [num_example, each_timestep]

  # show the evaluation per trajectory class if actev experiment
  if config.is_actev:
    l2dis_cats = [[] for i in range(len(config.traj_cats))]
    # added 06/2019,
    # show per-scene ADE/FDE for ActEV dataset
    # for leave-one-scene-out experiment
    l2dis_scenes = [[] for i in range(len(config.scenes))]

  grid1_acc = None
  grid2_acc = None
  grid1 = []
  grid2 = []

  # BG class is also used for evaluate
  future_act_scores = {actid: [] for actid in config.activity2id.values()}
  future_act_labels = {actid: [] for actid in config.activity2id.values()}
  act_ap = None

  num_batches_per_epoch = int(
      math.ceil(dataset.num_examples / float(config.batch_size)))

  traj_class_correct = []
  if config.is_actev:
    traj_class_correct_cat = [[] for i in range(len(config.traj_cats))]

  for evalbatch in tqdm.tqdm(dataset.get_batches(config.batch_size, \
    full=True, shuffle=False), total=num_batches_per_epoch, ascii=True):

    # [N,pred_len, 2]
    # here the output is relative output
    pred_out, future_act, grid_pred_1, grid_pred_2, \
        traj_class_logits, _ = tester.step(sess, evalbatch)

    _, batch = evalbatch

    this_actual_batch_size = batch.data["original_batch_size"]

    d = []

    # activity location prediction
    grid_pred_1 = np.argmax(grid_pred_1, axis=1)
    grid_pred_2 = np.argmax(grid_pred_2, axis=1)

    for i in range(len(batch.data["pred_grid_class"])):
      gt_grid1_pred_class = batch.data["pred_grid_class"][i][0, -1]
      gt_grid2_pred_class = batch.data["pred_grid_class"][i][1, -1]

      grid1.append(grid_pred_1[i] == gt_grid1_pred_class)
      grid2.append(grid_pred_2[i] == gt_grid2_pred_class)

    if config.add_activity:
      # get the mean AP
      for i in range(len(batch.data["future_activity_onehot"])):
        # [num_act]
        this_future_act_labels = batch.data["future_activity_onehot"][i]
        for j in range(len(this_future_act_labels)):
          actid = j
          future_act_labels[actid].append(this_future_act_labels[j])
          # for checking AP using the cur act as
          future_act_scores[actid].append(future_act[i, j])

    for i, (obs_traj_gt, pred_traj_gt) in enumerate(
        zip(batch.data["obs_traj"], batch.data["pred_traj"])):
      if i >= this_actual_batch_size:
        break
      # the output is relative coordinates
      this_pred_out = pred_out[i][:, :2]  # [T2, 2]
      # [T2,2]
      this_pred_out_abs = relative_to_abs(this_pred_out, obs_traj_gt[-1])
      # get the errors
      assert this_pred_out_abs.shape == this_pred_out.shape, (
          this_pred_out_abs.shape, this_pred_out.shape)

      # [T2, 2]
      diff = pred_traj_gt - this_pred_out_abs
      diff = diff**2
      diff = np.sqrt(np.sum(diff, axis=1))  # [T2]

      if config.multi_decoder:
        this_traj_class = np.argmax(traj_class_logits[i])
        traj_correct = int(this_traj_class ==
                           batch.data["traj_cat"][i])
        traj_class_correct.append(traj_correct)

        traj_class_correct_cat[batch.data["traj_cat"][i]].append(
            traj_correct)

      d.append(diff)

      if config.is_actev:
        traj_cat_id = batch.data["trajidx2catid"][i]
        l2dis_cats[traj_cat_id].append(diff)  # [T2]
        # per-scene eval
        traj_key = batch.data["traj_key"][i]  # videoname_frameidx_personid
        # videoname has '_'
        videoname = traj_key[::-1].split("_", 2)[-1][::-1]
        scene = get_scene(videoname)  # 0000/0002, etc.
        l2dis_scenes[config.scenes.index(scene)].append(diff)

    l2dis += d

  grid1_acc = np.mean(grid1)
  grid2_acc = np.mean(grid2)

  if config.add_activity:
    act_ap = []
    actids = []
    for actid in future_act_labels:
      list_ = [{"score": future_act_scores[actid][i],
                "label": future_act_labels[actid][i]}
               for i in range(len(future_act_labels[actid]))]
      ap = compute_ap(list_)

      act_ap.append(ap)
      actids.append(actid)

    act_ap = np.mean(act_ap)

  # average displacement
  ade = [t for o in l2dis for t in o]
  # final displacement
  fde = [o[-1] for o in l2dis]
  p = {"ade": np.mean(ade),
       "fde": np.mean(fde),
       "grid1_acc": grid1_acc,
       "grid2_acc": grid2_acc,
       "act_ap": act_ap}

  if config.multi_decoder:
    p.update({
        "traj_class_accuracy":
            np.mean(traj_class_correct) if traj_class_correct else 0.0,
    })
    for i in range(len(config.traj_cats)):
      p.update({
          ("traj_class_accuracy_%s" % i):
              np.mean(traj_class_correct_cat[i]) if traj_class_correct_cat[i]
              else 0.0,
      })

  # show ade and fde for different traj category
  if config.is_actev:

    # per-traj-class eval
    for cat_id, (cat_name, _) in enumerate(config.traj_cats):
      diffs = l2dis_cats[cat_id]
      ade = [t for l in diffs for t in l]
      fde = [l[-1] for l in diffs]
      p.update({
          ("%s_ade" % cat_name): np.mean(ade) if ade else 0.0,
          ("%s_fde" % cat_name): np.mean(fde) if fde else 0.0,
      })

    # per-scene eval
    for scene_id, scene in enumerate(config.scenes):
      diffs = l2dis_scenes[scene_id]
      ade = [t for l in diffs for t in l]
      fde = [l[-1] for l in diffs]
      p.update({
          ("%s_ade" % scene): np.mean(ade) if ade else 0.0,
          ("%s_fde" % scene): np.mean(fde) if fde else 0.0,
      })

  return p


class Dataset(object):
  """Class for batching during training and testing."""

  def __init__(self, data, data_type, config=None, shared=None):
    self.data = data
    self.data_type = data_type
    self.valid_idxs = range(self.get_data_size())
    self.num_examples = len(self.valid_idxs)
    self.shared = shared
    self.config = config

  def get_data_size(self):
    return len(self.data["obs_traj"])

  def get_by_idxs(self, idxs):
    out = collections.defaultdict(list)
    for key, val in self.data.items():
      out[key].extend(val[idx] for idx in idxs)
    return out

  def get_batches(self, batch_size, \
      num_steps=0, shuffle=True, cap=False, full=False):
    """Iterator to get batches.

    should return num_steps -> batches
    step is total/batchSize * epoch
    cap means limits max number of generated batches to 1 epoch

    Args:
      batch_size: batch size.
      num_steps: total steps.
      shuffle: whether shuffling the data
      cap: cap at one epoch
      full: use full one epoch

    Yields:
      Dataset object.
    """

    num_batches_per_epoch = int(
        math.ceil(self.num_examples / float(batch_size)))
    if full:
      num_steps = num_batches_per_epoch

    if cap and (num_steps > num_batches_per_epoch):
      num_steps = num_batches_per_epoch
    # this may be zero
    num_epochs = int(math.ceil(num_steps/float(num_batches_per_epoch)))
    # shuflle
    if shuffle:
      # All epoch has the same order.
      random_idxs = random.sample(self.valid_idxs, len(self.valid_idxs))
      # all batch idxs for one epoch

      def random_grouped():
        return list(grouper(random_idxs, batch_size))
      # grouper
      # given a list and n(batch_size), devide list into n sized chunks
      # last one will fill None
      grouped = random_grouped
    else:
      def raw_grouped():
        return list(grouper(self.valid_idxs, batch_size))
      grouped = raw_grouped

    # all batches idxs from multiple epochs
    batch_idxs_iter = itertools.chain.from_iterable(
        grouped() for _ in range(num_epochs))
    for _ in range(num_steps):  # num_step should be batch_idxs length
      # so in the end batch, the None will not included
      batch_idxs = tuple(i for i in next(batch_idxs_iter)
                         if i is not None)  # each batch idxs

      # so batch_idxs might not be size batch_size
      # pad with the last item
      original_batch_size = len(batch_idxs)
      if len(batch_idxs) < batch_size:
        pad = batch_idxs[-1]
        batch_idxs = tuple(
            list(batch_idxs) + [pad for i in
                                range(batch_size - len(batch_idxs))])

      # get the actual data based on idx
      batch_data = self.get_by_idxs(batch_idxs)

      batch_data.update({
          "original_batch_size": original_batch_size,
      })

      config = self.config

      # assemble a scene feat from the full scene feat matrix for this batch
      oldid2newid = {}
      new_obs_scene = np.zeros((config.batch_size, config.obs_len, 1),
                               dtype="int32")

      for i in range(len(batch_data["obs_scene"])):
        for j in range(len(batch_data["obs_scene"][i])):
          oldid = batch_data["obs_scene"][i][j][0]
          if oldid not in oldid2newid:
            oldid2newid[oldid] = len(oldid2newid.keys())
          newid = oldid2newid[oldid]
          new_obs_scene[i, j, 0] = newid
      # get all the feature used by this mini-batch
      scene_feat = np.zeros((len(oldid2newid), config.scene_h,
                             config.scene_w, config.scene_class),
                            dtype="float32")
      for oldid in oldid2newid:
        newid = oldid2newid[oldid]
        scene_feat[newid, :, :, :] = \
            self.shared["scene_feat"][oldid, :, :, :]

      batch_data.update({
          "batch_obs_scene": new_obs_scene,
          "batch_scene_feat": scene_feat,
      })

      yield batch_idxs, Dataset(batch_data, self.data_type, shared=self.shared)


def grouper(lst, num):
  args = [iter(lst)]*num
  if sys.version_info > (3, 0):
    out = itertools.zip_longest(*args, fillvalue=None)
  else:
    out = itertools.izip_longest(*args, fillvalue=None)
  out = list(out)
  return out


def compute_ap(lists):
  """Compute Average Precision."""
  lists.sort(key=operator.itemgetter("score"), reverse=True)
  rels = 0
  rank = 0
  score = 0.0
  for one in lists:
    rank += 1
    if one["label"] == 1:
      rels += 1
      score += rels/float(rank)
  if rels != 0:
    score /= float(rels)
  return score


def relative_to_abs(rel_traj, start_pos):
  """Relative x,y to absolute x,y coordinates.

  Args:
    rel_traj: numpy array [T,2]
    start_pos: [2]
  Returns:
    abs_traj: [T,2]
  """

  # batch, seq_len, 2
  # the relative xy cumulated across time first
  displacement = np.cumsum(rel_traj, axis=0)
  abs_traj = displacement + np.array([start_pos])  # [1,2]
  return abs_traj
