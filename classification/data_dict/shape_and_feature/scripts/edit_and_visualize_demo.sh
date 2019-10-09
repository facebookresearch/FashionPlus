#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

CLASS='humanparsing'  # segmentation definition from dataset "humanparsing"
COLOR_MODE='Lab' # RGB, Lab
NET_ARCH='mlp' # linear, mlp
MODEL='pix2pixHD'
LAMBDA_KL=0.0001 # hyperparameter for VAE 0.0001
DIVIDE_K=4 # hyperparameter for VAE 4
TEXTURE_FEAT_NUM=8
ROOT_DIR='/FashionPlus/' # absolute path for FashionPlus


# Editing module options
UPDATE_FNAME=$1 # filename to update
UPDATE_TYPE=$2 # specify whether to edit shape_only, texture_only, or shape_and_texture
AUTO_SWAP=$3 # auto_swap is True if automatically deciding which part to swap out; then swapped_partID will be unused
SWAPPED_PARTID=$4 # swapped_partID specifies which part to update; for class='humanparsing', partID mapping is: 0 top, 1 skirt, 2 pants, 3 dress
MAXITER=$5 # editing module stops at maxtier iterations
UPDATE_STEP_SZ=$6 # editing module takes step size at each update iteration
ITERATIVE_SAVE='False' # iterative_save is True when we generate edited resutls from each iteration

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
    CLF_EPOCH=120
    DATASET_DIR='../../datasets/'
    SAVE_DIR='results/'${COLOR_MODE}'/'${CLASS}'/'${UPDATE_TYPE}'/demo/'
    TEXTURE_PATH=${ROOT_DIR}'generation/results/Lab/demo/test_features.p'
    TEXTURE_GEN_PATH=${ROOT_DIR}'/checkpoint/'
    SAVE_IMGS_DIR=${ROOT_DIR}'generation/results/'${COLOR_MODE}'/'${CLASS}'/'${UPDATE_TYPE}'/demo'
    SHAPE_PATH=${ROOT_DIR}'separate_vae/results/Lab/demo/test_shape_codes.p'
    SHAPE_GEN_PATH=${ROOT_DIR}'/checkpoint/'
    SAVE_MASKS_DIR=${ROOT_DIR}'separate_vae/results/'${COLOR_MODE}'/'${CLASS}'/'${UPDATE_TYPE}'/demo'
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



############### UPDATE AND GENERATE  ###############
exec_options="python update_demo.py \
                     --update_fname ${UPDATE_FNAME} \
                     --update_type ${UPDATE_TYPE} \
                     --max_iter_hr ${MAXITER} \
		     --swapped_partID ${SWAPPED_PARTID} \
                     --lr ${UPDATE_STEP_SZ} \
                     --min_thresholdloss 0.00009 \
                     --model_type ${MODEL} \
                     --texture_feat_num ${TEXTURE_FEAT_NUM} \
                     --texture_feat_file ${TEXTURE_PATH} \
                     --shape_feat_file ${SHAPE_PATH} \
                     --dataset_dir ${DATASET_DIR} \
                     --param_m ${PARAM_M} \
                     --param_k ${PARAM_K} \
                     --load_pretrain_clf ${CLASSIFIER_PATH} \
                     --load_pretrain_texture_gen ${TEXTURE_GEN_PATH} \
                     --load_pretrain_shape_gen ${SHAPE_GEN_PATH} \
                     --network_arch ${NET_ARCH} \
                     --in_dim ${DFEAT} \
                     --clf_epoch ${CLF_EPOCH} \
                     --lambda_smooth 0 \
                     --display_freq 1 \
                     --classname ${CLASS} \
                     --color_mode ${COLOR_MODE} \
                     --save_dir ${SAVE_DIR}"

if [[ "${ITERATIVE_SAVE}" == 'True'  ]];
then
  exec_options="${exec_options} --iterative_generation"
  printf -v ITER_HEADER "%03d" ${MAXITER}
else
  ITER_HEADER='final'
fi
if [[ "${AUTO_SWAP}" == 'True'  ]];
then
  exec_options="${exec_options} --autoswap"
fi
eval ${exec_options}


############## POSTPROCESS ###############
LABEL_DIR=${ROOT_DIR}/datasets/images/
IMG_DIR=${ROOT_DIR}/datasets/labels/
cd ${ROOT_DIR}/postprocess
python process_face.py \
        --fname ${ITER_HEADER}_${UPDATE_FNAME} \
        --orig_img_dir ${LABEL_DIR} \
	--orig_mask_dir ${IMG_DIR} \
        --gen_img_dir ${SAVE_IMGS_DIR} \
	--gen_mask_dir ${SAVE_MASKS_DIR} \
	--bbox_pickle_file ${ROOT_DIR}/generation/datasets/demo/test.p \
	--result_dir ${ROOT_DIR}/classification/data_dict/shape_and_feature/results/demo/images/
python process_face.py \
        --fname 001_${UPDATE_FNAME} \
        --orig_img_dir ${LABEL_DIR} \
	--orig_mask_dir ${IMG_DIR} \
        --gen_img_dir ${SAVE_IMGS_DIR} \
	--gen_mask_dir ${SAVE_MASKS_DIR} \
	--bbox_pickle_file ${ROOT_DIR}/generation/datasets/demo/test.p \
	--result_dir ${ROOT_DIR}/classification/data_dict/shape_and_feature/results/demo/images/
