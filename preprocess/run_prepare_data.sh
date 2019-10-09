#!/bin/bash

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

ROOT_DIR='/private/home/kimberlyhsiao/code/pix2pixHD/FashionPlus/'
LABEL_DIR=${ROOT_DIR}'/datasets/labels/' #  directory with segmentation labels
IMG_DIR=${ROOT_DIR}'/datasets/images/' # directiroy with RGB images
python prepare_input_data.py --img_dir ${IMG_DIR} \
	                     --mask_dir ${LABEL_DIR} \
			     --output_pickle_file ${ROOT_DIR}/generation/datasets/demo/test.p \
                             --output_json_file ${ROOT_DIR}/classification/datasets/demo_dict.json
