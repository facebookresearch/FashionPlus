# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
##############################################################################
#
# Based on:
# pix2pixHD (https://github.com/NVIDIA/pix2pixHD)
### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--lambda_kl', type=float, default=0.01, help='weight for KL loss') # Not for training network, but for directory name
        self.parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        self.parser.add_argument('--dataset_param_file', type=str, default='./dataset/<classname>/', help='file with dataset specific label mapping.')
        self.parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        self.parser.add_argument('--load_feat_dir', type=str, default='./results/', help='load pre-computed features from here.')
        self.parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        self.parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--n_samples', type=int, default=5, help='#samples')
        self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
        self.parser.add_argument('--cluster_path', type=str, default='features_clustered_010.npy', help='the path for clustered results of encoded features')
        self.parser.add_argument("--export_onnx", type=str, help="export ONNX model to a given file")
        self.parser.add_argument("--engine", type=str, help="run serialized TRT engine")
        self.parser.add_argument("--onnx", type=str, help="run ONNX model via TRT")
        self.parser.add_argument('--reference_idx', type=int, help='reference image index where we want a piece from it')
        self.parser.add_argument('--condition_idx', type=int, help='conditioned image index where we base our outfit on')
        self.parser.add_argument('--swap_piece', type=int, help='which piece we want to swap out from condiontion_img to reference_img')
        self.isTrain = False
