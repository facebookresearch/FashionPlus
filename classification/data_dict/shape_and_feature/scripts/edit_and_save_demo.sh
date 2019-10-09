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
	    ROOT_DIR)            ROOT_DIR=${VALUE} ;;
	    CLASS)               CLASS=${VALUE} ;;
            COLOR_MODE)          COLOR_MODE=${VALUE} ;;
            NET_ARCH)            NET_ARCH=${VALUE} ;;
            MODEL)               MODEL=${VALUE} ;;
            UPDATE_TYPE)         UPDATE_TYPE=${VALUE} ;;
            TEXTURE_FEAT_NUM)    TEXTURE_FEAT_NUM=${VALUE} ;;
            SAVE_CODES_DIR)      SAVE_CODES_DIR=${VALUE} ;;
	    UPDATE_FNAME)        UPDATE_FNAME=${VALUE} ;;
	    MAXITER)             MAXITER=${VALUE} ;;
            UPDATE_STEP_SZ)      UPDATE_STEP_SZ=${VALUE} ;;
	    AUTO_SWAP)           AUTO_SWAP=${VALUE} ;;
	    SWAPPED_PARTID)      SWAPPED_PARTID=${VALUE} ;;
            ITERATIVE_SAVE)      ITERATIVE_SAVE=${VALUE} ;;
            *)
    esac
done

case ${CLASS} in
'humanparsing')
  case ${TEXTURE_FEAT_NUM} in
  8)
    DFEAT=64
    ;;
  *)
    echo 'WRONG feature_dimension '${TEXTURE_FEAT_NUM}
    ;;
  esac
  ;;
*)
  echo 'WRONG category'${CLASS}
  ;;
esac

case ${MODEL} in
'pix2pixHD')
  case ${DFEAT} in
  64)
    PARAM_M=3
    PARAM_K=256
    # CLF_EPOCH=50
    CLF_EPOCH=120
    DATASET_DIR='../../datasets/'
    SAVE_DIR=${SAVE_CODES_DIR}
    TEXTURE_PATH=${ROOT_DIR}'generation/results/Lab/demo/test_features.p'
    SHAPE_PATH=${ROOT_DIR}'separate_vae/results/Lab/demo/test_shape_codes.p'
    CLASSIFIER_PATH='../../checkpoint/m'${PARAM_M}'k'${PARAM_K}'/'
  ;;
  *)
    echo 'WRONG feature_dimension '${DFEAT}
    ;;
  esac
  ;;
*)
  echo 'WRONG category'${MODEL}
  ;;
esac

exec_options="python update_demo.py \
                     --update_fname ${UPDATE_FNAME} \
                     --update_type ${UPDATE_TYPE} \
                     --max_iter_hr ${MAXITER} \
		     --swapped_partID ${SWAPPED_PARTID} \
                     --lr ${UPDATE_STEP_SZ} \
                     --model_type ${MODEL} \
                     --texture_feat_num ${TEXTURE_FEAT_NUM} \
                     --texture_feat_file ${TEXTURE_PATH} \
                     --shape_feat_file ${SHAPE_PATH} \
                     --dataset_dir ${DATASET_DIR} \
                     --param_m ${PARAM_M} \
                     --param_k ${PARAM_K} \
                     --load_pretrain_clf ${CLASSIFIER_PATH} \
                     --network_arch ${NET_ARCH} \
                     --in_dim ${DFEAT} \
                     --clf_epoch ${CLF_EPOCH} \
                     --lambda_smooth 0 \
                     --display_freq 1 \
                     --classname ${CLASS} \
                     --color_mode ${COLOR_MODE} \
                     --save_dir ${SAVE_DIR} \
                     --generate_or_save save"

if [[ "${ITERATIVE_SAVE}" == 'True'  ]];
then
  exec_options="${exec_options} --iterative_generation"
fi
if [[ "${AUTO_SWAP}" == 'True'  ]];
then
  exec_options="${exec_options} --autoswap"
fi

eval ${exec_options}
