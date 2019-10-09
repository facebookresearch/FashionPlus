#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

NZ=8
OUTPUT_NC=18
MAX_MULT=8
DOWN_SAMPLE=7
BOTNK='1d'
DIVIDE_K=4
BATCH_SIZE=4
LAMBDA_KL=0.0001
# command

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    case "$KEY" in
            CLASS)               CLASS=${VALUE} ;;
            SAVE_CODES_DIR)      SAVE_CODES_DIR=${VALUE} ;;
            SAVE_MASKS_DIR)      SAVE_MASKS_DIR=${VALUE} ;;
	    SHAPE_GEN_PATH)      SHAPE_GEN_PATH=${VALUE} ;;
            *)
    esac
done

python ./decode_masks_from_dict.py \
  --phase test \
  --dataroot ./datasets/${CLASS} \
  --label_dir /datasets01/humanparsing/092818/SegmentationClassAug \
  --label_txt_path ./datasets/${CLASS}/clothing_labels.txt \
  --name ${CLASS} \
  --share_decoder \
  --share_encoder \
  --separate_clothing_unrelated \
  --nz ${NZ} \
  --checkpoints_dir ${SHAPE_GEN_PATH}  \
  --output_nc ${OUTPUT_NC} \
  --max_mult ${MAX_MULT} \
  --n_downsample_global ${DOWN_SAMPLE} \
  --bottleneck ${BOTNK} \
  --resize_or_crop pad_and_resize \
  --loadSize 256 \
  --batchSize ${BATCH_SIZE} \
  --divide_by_K ${DIVIDE_K} \
  --results_dir ${SAVE_MASKS_DIR}\
  --load_feat_dir ${SAVE_CODES_DIR}
