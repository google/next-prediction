
## Step 1: Prepare the data and model
We experimented on the [ActEv dataset](https://actev.nist.gov) and
the [ETH & UCY dataset](https://graphics.cs.ucy.ac.cy/research/downloads/crowd-data).
*Please do obtain the data copyright and download the raw videos from their website.*
You can download our prepared features from the [project page](next.cs.cmu.edu)
by running the script `bash scripts/download_prepared_data.sh`.
This will download the following data, and will require
about 31 GB of disk space:

- `next-data/final_annos/`: This folder includes extracted features and
annotations for both experiments. In ActEv experiments, it includes
person appearance features, person keypoint features, scene semantic
features and scene object features. In ETH/UCY, person keypoint features
are not used.
- `next-data/actev_personboxfeat/`: This folder includes person appearance
features for ActEv experiments
- `next-data/ethucy_personboxfeat/`: This folder includes person appearance
features for ETH/UCY experiments

Then download the pretrained model following instructions
[from here](README.md#pretrained-models).

## Step 2: Preprocess - ActEv
Preprocess the data for training and testing.
The following is for ActEv experiments.

```
python code/preprocess.py next-data/final_annos/actev_annos/virat_2.5fps_resized_allfeature/ \
  actev_preprocess --obs_len 8 --pred_len 12 --add_kp \
  --kp_path next-data/final_annos/actev_annos/anno_kp/ --add_scene \
  --scene_feat_path next-data/final_annos/actev_annos/ade20k_out_36_64/ \
  --scene_map_path next-data/final_annos/actev_annos/anno_scene/ \
  --scene_id2name next-data/final_annos/actev_annos/scene36_64_id2name_top10.json \
  --scene_h 36 --scene_w 64 --video_h 1080 --video_w 1920 --add_grid \
  --add_person_box --person_box_path next-data/final_annos/actev_annos/anno_person_box/ \
  --add_other_box --other_box_path next-data/final_annos/actev_annos/anno_other_box/ \
  --add_activity --activity_path next-data/final_annos/actev_annos/anno_activity/ \
  --person_boxkey2id_p next-data/final_annos/actev_annos/person_boxkey2id.p
```

## Step 3: Train the models
You can train your model by running:

```
python code/train.py actev_preprocess next-models/actev_single_model model \
  --runId 2 --is_actev --add_kp --add_activity \
  --person_feat_path next-data/actev_personboxfeat --multi_decoder
```
By default this will train a model for ActEv dataset, periodically saving model
files to `next-models/actev_single_model/model/02/save` at the current working
directory. The script will also periodically evaluate the model on the
validation set and save the latest best model to
`next-models/actev_single_model/model/02/best`.
Detailed commands of the training script:

### Training options

- `--batch_size`: How many trajectories to use in each minibatch during training.
Default is 32.
- `--num_epochs`: Number of training epochs. Default is 100.
- `--init_lr`: Initial Learning rate. Default is 0.2.
- `--keep_prob`: 1 - dropout rate. Default is 0.7.
- `--optimizer`: Optimizer to use. Default is AdaDelta.
- `--act_loss_weight`: Weight for activity label classification loss.
Default is 1.0.
- `--grid_loss_weight`: Weight for activity location classification loss.
Default is 0.1.

###  Basic model options

- `--emb_size`: Embedding size. Default is 128.
- `--enc_hidden_size`: Encoder hidden size. Default is 256.
- `--dec_hidden_size`: Decoder hidden size. Default is 256.
- `--activation_func`: Activation function. Default is tanh.
You could choose from relu/lrelu/tanh.

## Step 4: Test the model
You can use following command to test the newly trained model:

```
python code/test.py actev_preprocess next-models/actev_single_model model \
  --runId 2 --load_best --is_actev --add_kp --add_activity \
  --person_feat_path next-data/actev_personboxfeat --multi_decoder
```
The best model on the validation set will be used.
