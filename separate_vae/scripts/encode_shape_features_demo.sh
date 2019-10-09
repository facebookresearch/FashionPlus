
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -ex
NZ=8
OUTPUT_NC=18
MAX_MULT=8
DOWN_SAMPLE=7
BOTNK='1d'
LAMBDA_KL=0.0001
DIVIDE_K=4

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    case "$KEY" in
            CLASS)               CLASS=${VALUE} ;;
	    LABEL_DIR)           LABEL_DIR=${VALUE} ;;
	    SHAPE_GEN_PATH)      SHAPE_GEN_PATH=${VALUE} ;;
            *)
    esac
done


python ./encode_features.py \
  --phase test \
  --dataroot ./datasets/demo \
  --label_dir ${LABEL_DIR} \
  --label_txt_path ./datasets/${CLASS}/clothing_labels.txt \
  --dataset_param_file ./datasets/${CLASS}/garment_label_part_map.json \
  --name ${CLASS} \
  --share_decoder \
  --share_encoder \
  --separate_clothing_unrelated \
  --nz ${NZ} \
  --checkpoints_dir ${SHAPE_GEN_PATH} \
  --output_nc ${OUTPUT_NC} \
  --use_dropout \
  --lambda_kl ${LAMBDA_KL}\
  --max_mult ${MAX_MULT} \
  --n_downsample_global ${DOWN_SAMPLE} \
  --bottleneck ${BOTNK} \
  --resize_or_crop pad_and_resize \
  --loadSize 256 \
  --batchSize 1 \
  --divide_by_K ${DIVIDE_K}
