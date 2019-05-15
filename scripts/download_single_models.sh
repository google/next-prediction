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


# Download single models for the ActEv and ETH/UCY experiments

mkdir -p next-models

wget https://next.cs.cmu.edu/data/pretrained_models/actev_single_model.tar -O next-models/actev_single_model.tar
wget https://next.cs.cmu.edu/data/pretrained_models/ethucy_single_model.tar -O next-models/ethucy_single_model.tar

# extract and delete the tar files
cd next-models
tar -xvf actev_single_model.tar
rm actev_single_model.tar
tar -xvf ethucy_single_model.tar
rm ethucy_single_model.tar
cd ..
