#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

ROOT_DIR='/FashionPlus/' # absolute path for FashionPlus
CLASS='humanparsing' # the dataset name that defines the segmenation labels
COLOR_MODE='Lab' # RGB, Lab
FEAT_NUM=8
LABEL_DIR=${ROOT_DIR}'/datasets/labels/' #  directory with segmentation labels
IMG_DIR=${ROOT_DIR}'/datasets/images/' # directiroy with RGB images
SHAPE_GEN_PATH=${ROOT_DIR}'/checkpoint/' # directory with cGAN weights
TEXTURE_GEN_PATH=${ROOT_DIR}'/checkpoint/' # directory with VAE weights

# Extract shape encodings
cd ../separate_vae
./scripts/encode_shape_features_demo.sh \
	CLASS=${CLASS} \
	LABEL_DIR=${LABEL_DIR} \
	SHAPE_GEN_PATH=${SHAPE_GEN_PATH}

# Extract texture encoding
cd ../generation
./scripts/encode_texture_features_demo.sh \
	CLASS=${CLASS} \
	COLOR_MODE=${COLOR_MODE} \
	FEAT_NUM=${FEAT_NUM} \
	LABEL_DIR=${LABEL_DIR} \
	IMG_DIR=${IMG_DIR} \
	TEXTURE_GEN_PATH=${TEXTURE_GEN_PATH}
