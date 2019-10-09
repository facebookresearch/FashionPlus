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
import pickle
import numpy as np
import os
from PIL import Image

from options.test_options import TestOptions
from data.base_dataset import get_params, get_transform, normalize
from models.models import create_model
import util.util as util

opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
update = True

remove_background = True

util.mkdirs(opt.results_dir)

############ Initialize #########
if not opt.engine and not opt.onnx:
    model = create_model(opt)
    if opt.data_type == 16:
        model.half()
    elif opt.data_type == 8:
        model.type(torch.uint8)

    if opt.verbose:
        print(model)
if opt.use_avg_features:
    with open(opt.cluster_path, 'rb') as readfile:
        avg_features = pickle.load(readfile)
    model.set_avg_features(avg_features)

# Read in the saved feature dict
with open(os.path.join(opt.load_feat_dir, 'texture_codes_dict.p'), 'rb') as readfile:
    fname_feature_dict = pickle.load(readfile)

for fname in fname_feature_dict:
    if update:
        path = os.path.join(os.path.abspath(opt.results_dir).replace('generation', 'separate_vae'), '%s.png' % (fname))
    else:
        path = os.path.join(opt.label_dir, fname[4:-3] + '.png')
    label = Image.open(path)

    if label:
        params = get_params(opt, label.size)
        transform_label = get_transform(opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        inst_tensor = transform_label(label)
    else:
        print('%s not exist!' % fname)
        exit()
    features = fname_feature_dict[fname]
    generated = model.inference_given_feature(label_tensor.unsqueeze(0), inst_tensor.unsqueeze(0), features, from_avg=opt.use_avg_features)

    # util.save_image(util.tensor2label(label_tensor, gan_opt.label_nc), '%03d_%s.jpg' % (epoch, fname))
    if opt.color_mode == 'Lab':
        if remove_background:
            util.save_image(util.tensor2LABim_nobackground(generated.data[0], label_tensor.data), os.path.join(opt.results_dir, '%s.jpg' % (fname)))
        else:
            util.save_image(util.tensor2LABim(generated.data[0]), os.path.join(opt.results_dir, '%s.jpg' % (fname)))
    else:
        util.save_image(util.tensor2im(generated.data[0]), os.path.join(opt.results_dir, '%s.jpg' % (fname)))
