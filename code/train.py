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

"""Train person prediction model.

See README for running instructions.
"""

import argparse
import math
import sys

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# the following will still have colocation debug info
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import models
from tqdm import tqdm
import utils

parser = argparse.ArgumentParser()

# inputs and outputs
parser.add_argument("prepropath", type=str)
parser.add_argument("outbasepath", type=str,
                    help="full path will be outbasepath/modelname/runId")
parser.add_argument("modelname", type=str)
parser.add_argument("--runId", type=int, default=0,
                    help="used for run the same model multiple times")

# ---- gpu stuff. Now only one gpu is used
parser.add_argument("--gpuid", default=0, type=int)

parser.add_argument("--load", action="store_true",
                    default=False, help="whether to load existing model")
parser.add_argument("--load_best", action="store_true",
                    default=False, help="whether to load the best model")
# use for pre-trained model
parser.add_argument("--load_from", type=str, default=None)

# ------------- experiment settings
parser.add_argument("--obs_len", type=int, default=8)
parser.add_argument("--pred_len", type=int, default=12)
parser.add_argument("--is_actev", action="store_true",
                    help="is actev/virat dataset, has activity info")

# ------------------- basic model parameters
parser.add_argument("--emb_size", type=int, default=128)
parser.add_argument("--enc_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--dec_hidden_size", type=int,
                    default=256, help="hidden size for rnn")
parser.add_argument("--activation_func", type=str,
                    default="tanh", help="relu/lrelu/tanh")

# ---- multi decoder
parser.add_argument("--multi_decoder", action="store_true")

# ----------- add person appearance features
parser.add_argument("--person_feat_path", type=str, default=None)
parser.add_argument("--person_feat_dim", type=int, default=256)
parser.add_argument("--person_h", type=int, default=9,
                    help="roi align resize to feature size")
parser.add_argument("--person_w", type=int, default=5,
                    help="roi align resize to feature size")

# ---------------- other boxes
parser.add_argument("--random_other", action="store_true",
                    help="randomize top k other boxes")
parser.add_argument("--max_other", type=int, default=15,
                    help="maximum number of other box")
parser.add_argument("--box_emb_size", type=int, default=64)

# ---------- person pose features
parser.add_argument("--add_kp", action="store_true")
parser.add_argument("--kp_size", default=17, type=int)

# --------- scene features
parser.add_argument("--scene_conv_kernel", default=3, type=int)
parser.add_argument("--scene_h", default=36, type=int)
parser.add_argument("--scene_w", default=64, type=int)
parser.add_argument("--scene_class", default=11, type=int)
parser.add_argument("--scene_conv_dim", default=64, type=int)
parser.add_argument("--pool_scale_idx", default=0, type=int)

#  --------- activity
parser.add_argument("--add_activity", action="store_true")

#  --------- loss weight
parser.add_argument("--act_loss_weight", default=1.0, type=float)
parser.add_argument("--grid_loss_weight", default=0.1, type=float)
parser.add_argument("--traj_class_loss_weight", default=1.0, type=float)

# ---------------------------- training hparam
parser.add_argument("--save_period", type=int, default=300,
                    help="num steps to save model and eval")
parser.add_argument("--batch_size", type=int, default=64)
# num_step will be num_example/batch_size * epoch
parser.add_argument("--num_epochs", type=int, default=100)
# drop out rate
parser.add_argument("--keep_prob", default=0.7, type=float,
                    help="1.0 - drop out rate")
# l2 weight decay rate
parser.add_argument("--wd", default=0.0001, type=float,
                    help="l2 weight decay loss")
parser.add_argument("--clip_gradient_norm", default=10, type=float,
                    help="gradient clipping")
parser.add_argument("--optimizer", default="adadelta",
                    help="momentum|adadelta|adam")
parser.add_argument("--learning_rate_decay", default=0.95,
                    type=float, help="learning rate decay")
parser.add_argument("--num_epoch_per_decay", default=2.0,
                    type=float, help="how epoch after which lr decay")
parser.add_argument("--init_lr", default=0.2, type=float,
                    help="Start learning rate")
parser.add_argument("--emb_lr", type=float, default=1.0,
                    help="learning scaling factor for emb variables")


def main(args):
  """Run training."""
  val_perf = []  # summary of validation performance, and the training loss

  train_data = utils.read_data(args, "train")
  val_data = utils.read_data(args, "val")

  args.train_num_examples = train_data.num_examples

  # construct model under gpu0
  model = models.get_model(args, gpuid=args.gpuid)

  trainer = models.Trainer(model, args)
  tester = models.Tester(model, args)
  saver = tf.train.Saver(max_to_keep=5)
  bestsaver = tf.train.Saver(max_to_keep=5)

  save_period = args.save_period  # also the eval period

  # start training!
  tfconfig = tf.ConfigProto(allow_soft_placement=True)
  tfconfig.gpu_options.allow_growth = True
  tfconfig.gpu_options.visible_device_list = "%s" % (
      ",".join(["%s" % i for i in [args.gpuid]]))
  with tf.Session(config=tfconfig) as sess:

    utils.initialize(
        load=args.load, load_best=args.load_best, args=args, sess=sess)

    # the total step (iteration) the model will run
    # total / batchSize  * epoch
    num_steps = int(math.ceil(train_data.num_examples /
                              float(args.batch_size)))*args.num_epochs
    # get_batches is a generator, run on the fly

    print(" batch_size:%s, epoch:%s, %s step every epoch, total step:%s,"
          " eval/save every %s steps" % (args.batch_size,
                                         args.num_epochs,
                                         math.ceil(train_data.num_examples/
                                                   float(args.batch_size)),
                                         num_steps,
                                         args.save_period))

    metric = "ade"  # average displacement error # smaller better
    # remember the best eval acc during training
    best = {metric: 999999, "step": -1}

    finalperf = None
    is_start = True
    loss = -1
    grid_loss = -1
    xyloss = -1
    act_loss = -1
    traj_class_loss = -1

    for batch in tqdm(train_data.get_batches(args.batch_size,
                                             num_steps=num_steps),
                      total=num_steps, ascii=True):

      global_step = sess.run(model.global_step) + 1  # start from 0

      # if load from existing model, save if first
      if (global_step % save_period == 0) or \
         (args.load_best and is_start) or \
         (args.load and is_start and (args.ignore_vars is None)):

        tqdm.write("\tsaving model %s..." % global_step)
        saver.save(sess, args.save_dir_model, global_step=global_step)
        tqdm.write("\tdone")

        evalperf = utils.evaluate(val_data, args, sess, tester)

        tqdm.write(("\tlast loss:%.5f, xyloss:%.5f, traj_class_loss:%.5f,"
                    " grid_loss:%s, act_loss:%.5f, eval on validation:%s,"
                    " (best %s:%s at step %s) ") % (
                        loss, xyloss, traj_class_loss, grid_loss, act_loss,
                        ["%s: %s" % (k, evalperf[k])
                         for k in sorted(evalperf.keys())], metric,
                        best[metric], best["step"]))

        # remember the best acc
        if evalperf[metric] < best[metric]:
          best[metric] = evalperf[metric]
          best["step"] = global_step
          # save the best model
          tqdm.write("\t saving best model...")
          bestsaver.save(sess, args.save_dir_best_model,
                         global_step=global_step)
          tqdm.write("\t done.")

          finalperf = evalperf
          val_perf.append((loss, evalperf))
        is_start = False

      loss, _, xyloss, act_loss, traj_class_loss, grid_loss = \
          trainer.step(sess, batch)
      if math.isnan(loss):
        print("nan loss.")
        print(grid_loss)
        sys.exit()

    if global_step % save_period != 0:
      saver.save(sess, args.save_dir_model, global_step=global_step)

    print("best eval on val %s: %s at %s step, final step %s %s is %s" % (
        metric, best[metric], best["step"], global_step, metric,
        finalperf[metric]))


if __name__ == "__main__":
  arguments = parser.parse_args()
  arguments.is_train = True
  arguments.is_test = False
  arguments = utils.process_args(arguments)

  main(arguments)
