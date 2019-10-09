#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

NETG='local' # local, global
MODELNAME='pix2pixHD'
PHASE='test'

for ARGUMENT in "$@"
do
    KEY=$(echo $ARGUMENT | cut -f1 -d=)
    VALUE=$(echo $ARGUMENT | cut -f2 -d=)
    case "$KEY" in
            CLASS)               CLASS=${VALUE} ;;
	    COLOR_MODE)          COLOR_MODE=${VALUE} ;;
	    FEAT_NUM)            FEAT_NUM=${VALUE} ;;
	    LABEL_DIR)           LABEL_DIR=${VALUE} ;;
	    IMG_DIR)             IMG_DIR=${VALUE} ;;
	    TEXTURE_GEN_PATH)    TEXTURE_GEN_PATH=${VALUE} ;;
            *)
    esac
done


case ${CLASS} in
'humanparsing')
  NUM_LABEL=18
  ;;
*)
  echo 'WRONG class '${CLASS}
  ;;
esac

case ${MODELNAME} in
'pix2pixHD')
  python ./encode_clothing_features.py \
    --dataroot ./datasets/demo/ \
    --phase ${PHASE} \
    --name demo \
    --model ${MODELNAME} \
    --feat_num ${FEAT_NUM} \
    --label_feat \
    --checkpoints_dir ${TEXTURE_GEN_PATH}/ \
    --load_pretrain ${TEXTURE_GEN_PATH}/${CLASS} \
    --label_dir ${LABEL_DIR} \
    --img_dir ${IMG_DIR} \
    --resize_or_crop pad_and_resize \
    --loadSize 256 \
    --label_nc ${NUM_LABEL} \
    --color_mode ${COLOR_MODE}
  ;;
*)
  echo 'WRONG model '${MODELNAME}
  ;;
esac
