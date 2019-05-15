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


# Download prepared data for the ActEv and ETH/UCY experiments

mkdir -p next-data

wget https://next.cs.cmu.edu/data/final_annos.tgz -O next-data/final_annos.tgz
wget https://next.cs.cmu.edu/data/person_features/actev_personboxfeat.tgz -O next-data/actev_personboxfeat.tgz
wget https://next.cs.cmu.edu/data/person_features/ethucy_personboxfeat.tgz -O next-data/ethucy_personboxfeat.tgz

# extract and delete the tar files
cd next-data

tar -zxvf final_annos.tgz
rm final_annos.tgz
tar -zxvf actev_personboxfeat.tgz
rm actev_personboxfeat.tgz
tar -zxvf ethucy_personboxfeat.tgz
rm ethucy_personboxfeat.tgz

cd ..
