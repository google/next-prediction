## Prepared Data
Here are some notes on the prepared data's format, which could be useful for data visualization. The folder `next-data/final_annos/actev_annos` includes the following folders:

- `virat_2.5fps_resized_allfeature/`: This folder includes each data split's trajectories. Each video has one text file. The format is similar to the data in [Social-GAN](https://github.com/agrimgupta92/sgan). Each row is `frame_num person_id X Y`. The frame_num is 0-indexed. The person_id is unique within each video. You can see the frame_num is not continuous since we drop 12 frame every time to down-sample the frames so we have 2.5 FPS as the ETH/UCY data. The X, Y coordinates are **under 1920x1080 scale**, which means we have resized the 1280x720 frames from scene `0002` in the ActEV data. The same goes to all other coordinates of bounding boxes and keypoints.
- `anno_person_box/`: This folder includes each data split's person box coordinates. Each video has one pickle file. The pickle file is a dictionary with keys in the format of `${frame_num}_${person_id}` and the box coordinates are in the format of `x1 y1 x2 y2`.
- `anno_kp/`: Similar but with person keypoint coordinates in [17, 3] shape. The 17 keypoint indexes correspond to the ones in Detectron. Each row is `X Y logit`. Logit is in the range from 0 to 6 according to AlphaPose.
- `anno_activity/`: Similar but with each person's future and current activities: `current_actid_list, _, future_actid_list, _ = activity_pickle[key]`.
- `anno_other_box/`: Similar but with each person's other object boxes in the scene. The object class mappings can be found in `utils.py`. You can read them by:
```
# a list of [4], each is [x1, y1, x2, y2]
this_other_box.append(other_box_pickle[key][0])
# a list of [1]
this_other_box_class.append(other_box_pickle[key][1])
```
- `ade20k_out_36_64/`: This folder includes extracted downsized scene semantic segmentation features. We only keep one frame every 30 frames to save computation and disk space. During preprocessing, the closest frame's feature will be used according to the mappings in `anno_scene/`.
- `person_boxkey2id.p`: This pickle file includes mappings from `${videoname}_${framenum}_${person_id}` to person_appearance_feature_id. You can then get each person's appearance feature from `next-data/actev_personboxfeat/${data_split}/${person_appearance_feature_id}.npy`, which is a numpy array of shape [1, 9, 5, 256].

