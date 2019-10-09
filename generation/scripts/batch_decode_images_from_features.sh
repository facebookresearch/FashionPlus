#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# parsing the arguments passed in from batch wrapper
for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    case "$KEY" in
            CLASS)               CLASS=${VALUE} ;;
            COLOR_MODE)          COLOR_MODE=${VALUE} ;;
            MODEL)               MODELNAME=${VALUE} ;;
            TEXTURE_FEAT_NUM)    FEAT_NUM=${VALUE} ;;
            SAVE_CODES_DIR)      SAVE_CODES_DIR=${VALUE} ;;
            SAVE_MASKS_DIR)      SAVE_MASKS_DIR=${VALUE} ;;
            SAVE_IMGS_DIR)       SAVE_IMGS_DIR=${VALUE} ;;
	    TEXTURE_GEN_PATH)    TEXTURE_GEN_PATH=${VALUE} ;;
            *)
    esac
done
LAMBDA_Z=0.5 # 0.5
LAMBDA_KL=0.01 # 0.01
LAMBDA_FEAT=10 # 50

case ${CLASS} in
'humanparsing')
  LABEL_DIR='/datasets01/humanparsing/092818/SegmentationClassAug'
  IMG_DIR='/datasets01/humanparsing/092818/JPEGImages'
  NUM_LABEL=18
  ;;
*)
  echo 'WRONG class '${CLASS}
  ;;
esac

case ${MODELNAME} in
'pix2pixHD')

  python ./decode_clothing_features_from_dict.py \
    --dataroot ./datasets/${CLASS} \
    --name ${CLASS} \
    --model ${MODELNAME} \
    --feat_num ${FEAT_NUM} \
    --label_feat \
    --checkpoints_dir ${TEXTURE_GEN_PATH}/ \
    --label_dir ${LABEL_DIR} \
    --img_dir ${IMG_DIR} \
    --resize_or_crop pad_and_resize \
    --loadSize 256 \
    --label_nc ${NUM_LABEL} \
    --color_mode ${COLOR_MODE} \
    --results_dir ${SAVE_IMGS_DIR} \
    --load_feat_dir ${SAVE_CODES_DIR} \
    --use_avg_features \
    --cluster_path ${TEXTURE_GEN_PATH}/${CLASS}/train_avg_features.p
  ;;
*)
  echo 'WRONG model '${MODELNAME}
  ;;
esac

# command
