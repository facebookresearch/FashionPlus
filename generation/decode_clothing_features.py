# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import importlib
if 'options.test_options' in sys.modules.keys():
    importlib.reload(sys.modules['options'])
    importlib.reload(sys.modules['options.base_options'])
    importlib.reload(sys.modules['options.test_options'])
if 'models' in sys.modules.keys():
    importlib.reload(sys.modules['models'])
if 'util.util' in sys.modules.keys():
    importlib.reload(sys.modules['util'])
    importlib.reload(sys.modules['util.util'])

import torch
import pickle
from PIL import Image
from data.base_dataset import get_params, get_transform, normalize
from options.test_options import TestOptions as gan_TestOptions
from models.models import create_model as gan_create_model
import util.util as gan_util

def initialize_option(classname, decode=False):
    gan_opt = gan_TestOptions().get_opt()
    gan_opt.name = classname
    if classname == 'humanparsing':
        gan_opt.dataroot = './datasets/humanparsing'
        gan_opt.checkpoints_dir = '/checkpoint/'
        if decode:
            gan_opt.label_dir = '/datasets/decoded_masks'
        else:
            gan_opt.label_dir = '/datasets/labels/'
        gan_opt.img_dir = '/datasets/images'
        gan_opt.label_nc = 18
    else:
        raise NotImplementedError
    gan_opt.label_feat = True
    if decode:
        gan_opt.resize_or_crop = 'none'
    else:
        gan_opt.resize_or_crop = 'pad_and_resize'
    gan_opt.loadSize = 256
    gan_opt.nThreads = 1   # test code only supports nThreads = 1
    gan_opt.batchSize = 1  # test code only supports batchSize = 1
    gan_opt.serial_batches = True  # no shuffle
    gan_opt.no_flip = True  # no flip
    return gan_opt

# Generation from encoded mask (don't need any preprocess)
def generation_from_decoded_mask(epoch, save_path, fname, features, checkpoints_dir, \
                                color_mode, netG, model_type, classname, feat_num=3, \
                                original_mask=False, update=True, debug=True, from_avg=False, remove_background=False):
    ''' Generate final output image from decoded mask and texture features
        Args: epoch (int), edit module result at epoch
              save_path (str), save output image to path
              fname (str), save output image to filename
              checkpoints_dir (str), load generator weights from checkpoint
              color_mode (str), we use Lab color space for stability here
              netG (str), local for output image size 256x256;
                          global for output image higher-res image; not supported
              model_type (str), only support pix2pixHD here
              classname (str), label taxonomy from dataset classname
              feat_num (int), latent code feature dimension
              original_mask (boolean), whether to use decoded mask or original mask
                                       not supporting using original mask to avoid confusion
              update (boolean), use updated decoded mask
              debug (boolean), deprecated
              from_avg (boolean), use average feature values for missing parts
              remove_background (boolean), use white background
    '''
    gan_opt = initialize_option(classname, decode = not original_mask)
    gan_opt.checkpoints_dir = checkpoints_dir
    gan_opt.model = model_type
    gan_opt.feat_num = feat_num
    if model_type.startswith('cvae') or model_type.startswith('bicycle'):
        gan_opt.use_vae = True
    if netG == 'local':
        gan_opt.netG = 'local'
        print('local')
    gan_util.mkdirs(save_path)

    model = gan_create_model(gan_opt)
    if from_avg:
        with open(os.path.join(checkpoints_dir, classname, 'train_avg_features.p'), 'rb') as readfile:
            avg_features = pickle.load(readfile)
        model.set_avg_features(avg_features)

    if original_mask:
        raise NotImplementedError
    else:
        if update:
            path = os.path.join(os.path.abspath(save_path).replace('generation', 'separate_vae'), '%s_%s.png' % (epoch, fname))
        else:
            raise NotImplementedError
        label = Image.open(path)

    if label:
        params = get_params(gan_opt, label.size)
        transform_label = get_transform(gan_opt, params, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        inst_tensor = transform_label(label)
    else:
        print('%s not exist!' % fname)
        exit()
    generated = model.inference_given_feature(label_tensor.unsqueeze(0), inst_tensor.unsqueeze(0), features, from_avg=from_avg)

    if color_mode == 'Lab':
        if remove_background:
            gan_util.save_image(gan_util.tensor2LABim_nobackground(generated.data[0], label_tensor.data), os.path.join(save_path, '%s_%s.jpg' % (epoch, fname)))
        else:
            gan_util.save_image(gan_util.tensor2LABim(generated.data[0]), os.path.join(save_path, '%s_%s.jpg' % (epoch, fname)))
    else:
        gan_util.save_image(gan_util.tensor2im(generated.data[0]), os.path.join(save_path, '%s_%s.jpg' % (epoch, fname)))
