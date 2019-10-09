#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

set -ex
ROOT_DIR='/FashionPlus/' # absolute path for FashionPlus
CLASS='humanparsing'  # segmentation definition from dataset "humanparsing"
MODEL='pix2pixHD'
COLOR_MODE='Lab' # RGB, Lab
NET_ARCH='mlp' # linear, mlp
TEXTURE_FEAT_NUM=8
LAMBDA_KL=0.0001 # hyperparameter for VAE 0.0001
DIVIDE_K=4 # hyperparameter for VAE 4

# Editing module options
UPDATE_FNAME='3.jpg' # filename to update
UPDATE_TYPE='shape_and_texture' # specify whether to edit shape_only, texture_only, or shape_and_texture
AUTO_SWAP='True' # auto_swap is True if automatically deciding which part to swap out; then swapped_partID will be unused
SWAPPED_PARTID=0 # swapped_partID specifies which part to update; for class='humanparsing', partID mapping is: 0 top, 1 skirt, 2 pants, 3 dress
MAXITER=10 # editing module stops at maxiter iterations
UPDATE_STEP_SZ=0.05 # editing module takes step size at each update iteration
ITERATIVE_SAVE='False' # iterative_save is True when we generate edited results from each iteration

case ${MODEL} in
'pix2pixHD')
  case ${TEXTURE_FEAT_NUM} in
  8)
    SAVE_CODES_DIR=${ROOT_DIR}'classification/data_dict/shape_and_feature/results/demo/'
    SAVE_MASKS_DIR=${ROOT_DIR}'separate_vae/results/'${COLOR_MODE}'/'${CLASS}'/'${UPDATE_TYPE}'/demo'
    SAVE_IMGS_DIR=${ROOT_DIR}'generation/results/'${COLOR_MODE}'/'${CLASS}'/'${UPDATE_TYPE}'/demo'
    TEXTURE_GEN_PATH=${ROOT_DIR}'/checkpoint/'
    SHAPE_GEN_PATH=${ROOT_DIR}'/checkpoint/'
    ;;
  *)
    echo 'WRONG feature_dimension '${TEXTURE_FEAT_NUM}
    ;;
  esac
  ;;
*)
  echo 'WRONG category'${MODEL}
  ;;
esac

############### UPDATE ###############
bash scripts/edit_and_save_demo.sh \
  ROOT_DIR=${ROOT_DIR} \
  CLASS=${CLASS} \
  COLOR_MODE=${COLOR_MODE} \
  NET_ARCH=${NET_ARCH} \
  MODEL=${MODEL} \
  UPDATE_TYPE=${UPDATE_TYPE} \
  TEXTURE_FEAT_NUM=${TEXTURE_FEAT_NUM} \
  SAVE_CODES_DIR=${SAVE_CODES_DIR} \
  UPDATE_FNAME=${UPDATE_FNAME} \
  MAXITER=${MAXITER} \
  UPDATE_STEP_SZ=${UPDATE_STEP_SZ} \
  AUTO_SWAP=${AUTO_SWAP} \
  SWAPPED_PARTID=${SWAPPED_PARTID} \
  ITERATIVE_SAVE=${ITERATIVE_SAVE}

############## MASKS ###############
cd ${ROOT_DIR}/separate_vae

bash scripts/batch_decode_masks_from_features.sh \
  CLASS=${CLASS} \
  SAVE_CODES_DIR=${SAVE_CODES_DIR} \
  SAVE_MASKS_DIR=${SAVE_MASKS_DIR} \
  SHAPE_GEN_PATH=${SHAPE_GEN_PATH}

############## IMAGES ###############
cd ${ROOT_DIR}/generation

bash scripts/batch_decode_images_from_features.sh \
  CLASS=${CLASS} \
  COLOR_MODE=${COLOR_MODE} \
  MODEL=${MODEL} \
  TEXTURE_FEAT_NUM=${TEXTURE_FEAT_NUM} \
  SAVE_IMGS_DIR=${SAVE_IMGS_DIR} \
  SAVE_CODES_DIR=${SAVE_CODES_DIR} \
  SAVE_MASKS_DIR=${SAVE_MASKS_DIR} \
  TEXTURE_GEN_PATH=${TEXTURE_GEN_PATH}

############## POSTPROCESS ###############
cd ${ROOT_DIR}/postprocess
if [[ "${ITERATIVE_SAVE}" == 'True'  ]];
then
  printf -v ITER_HEADER "%03d" ${MAXITER}
else
  ITER_HEADER='final'
fi

LABEL_DIR=${ROOT_DIR}/datasets/images/
IMG_DIR=${ROOT_DIR}/datasets/labels/
python process_face.py \
        --fname ${ITER_HEADER}_${UPDATE_FNAME} \
        --orig_img_dir ${LABEL_DIR} \
	--orig_mask_dir ${IMG_DIR} \
        --gen_img_dir ${SAVE_IMGS_DIR} \
	--gen_mask_dir ${SAVE_MASKS_DIR} \
	--bbox_pickle_file ${ROOT_DIR}/generation/datasets/demo/test.p \
	--result_dir ${SAVE_CODES_DIR}/images/
python process_face.py \
        --fname 001_${UPDATE_FNAME} \
        --orig_img_dir ${LABEL_DIR} \
	--orig_mask_dir ${IMG_DIR} \
        --gen_img_dir ${SAVE_IMGS_DIR} \
	--gen_mask_dir ${SAVE_MASKS_DIR} \
	--bbox_pickle_file ${ROOT_DIR}/generation/datasets/demo/test.p \
	--result_dir ${SAVE_CODES_DIR}/images/
