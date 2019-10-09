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
import numpy as np
import torch
from options.test_options import TestOptions as vae_TestOptions
import util.util as vae_util

def initialize_option(classname):
    vae_opt = vae_TestOptions().get_opt()
    vae_opt.share_encoder = True
    vae_opt.share_decoder = True
    vae_opt.separate_clothing_unrelated = True
    vae_opt.name = classname
    if classname == 'humanparsing':
        vae_opt.dataroot = './datasets/humanparsing'
        vae_opt.checkpoints_dir = '/checkpoint/'
        vae_opt.label_dir = '/datasets/labels'
        vae_opt.img_dir = '/datasets/images'
        vae_opt.label_txt_path = './datasets/humanparsing/clothing_labels.txt'
        vae_opt.output_nc = 18
    else:
        raise NotImplementedError

    vae_opt.resize_or_crop = 'pad_and_resize'
    vae_opt.loadSize = 256
    vae_opt.batchSize = 1  # test code only supports batchSize = 1
    vae_opt.nz = 8
    vae_opt.divide_by_K = 4
    vae_opt.max_mult = 8
    vae_opt.n_downsample_global = 7
    vae_opt.bottleneck = '1d'
    return vae_opt

def single_generation_from_update(save_path, fname, features, checkpoints_dir, classname, black=True):
    ''' Generate decoded segmentation map from input features
        Args: save_path (str), save generated masks to path
              fname (str), save generated masks with fname
              features (numpy array): input features to be decoded
              checkpoints_dir (str), load VAE weights from path
              classname (str), label taxonomy defined by dataset with classname
              black (boolean), black is True for regular generation;
                               black is False for debugging, thus the generated mask
                               is not in the format for cGAN input
    '''
    vae_opt = initialize_option(classname)
    vae_opt.checkpoints_dir = checkpoints_dir

    vae_util.mkdirs(save_path)

    if vae_opt.share_decoder and vae_opt.share_encoder:
        if vae_opt.separate_clothing_unrelated:
            from models.separate_clothing_encoder_models import create_model as vae_create_model
        else:
            print('Only supports separating clothing and clothing-irrelevant')
            raise NotImplementedError
    else:
        print('Only supports sharing encoder and decoder among all parts')
        raise NotImplementedError

    model = vae_create_model(vae_opt)
    generated = model.generate_from_random(torch.Tensor(features).cuda())

    if black:
        vae_util.save_image(vae_util.tensor2label_black(generated.data[0], vae_opt.output_nc, normalize=True), os.path.join(save_path, '%s.png' % (fname)))
    else:
        vae_util.save_image(vae_util.tensor2label(generated.data[0], vae_opt.output_nc, normalize=True), os.path.join(save_path, '%s.png' % (fname)))

def batch_generation_from_update(batch_size, save_path, fname_list, features_mat, checkpoints_dir, classname, black=True):
    ''' Generate decoded segmentation map from input features
        Args: batch_size (int), batch size for the VAE to take in
              save_path (str), save generated masks to path
              fname_list (list of str), save generated masks with fname
              features_mat (numpy array): input features to be decoded,
                                          in the order of fname_list
              checkpoints_dir (str), load VAE weights from path
              classname (str), label taxonomy defined by dataset with classname
              black (boolean), black is True for regular generation;
                               black is False for debugging, thus the generated mask
                               is not in the format for cGAN input
    '''
    # Create and load model weights in
    vae_opt = initialize_option(classname)
    vae_opt.batchSize = batch_size
    vae_opt.checkpoints_dir = checkpoints_dir

    vae_util.mkdirs(save_path)

    if vae_opt.share_decoder and vae_opt.share_encoder:
        if vae_opt.separate_clothing_unrelated:
            from models.separate_clothing_encoder_models import create_model as vae_create_model
        else:
            print('Only supports separating clothing and clothing-irrelevant')
            raise NotImplementedError
    else:
        print('Only supports sharing encoder and decoder among all parts')
        raise NotImplementedError

    model = vae_create_model(vae_opt)
    # Forward input into the model
    dataset_size = len(fname_list)
    num_batch = int(dataset_size / batch_size)
    elem_in_last_batch = dataset_size % batch_size
    for i in range(num_batch):
        generated = model.generate_from_random(torch.Tensor(features_mat[i * batch_size: (i+1) * batch_size, :]).cuda())
        for j in range(batch_size):
            if black:
                vae_util.save_image(vae_util.tensor2label_black(generated.data[j], vae_opt.output_nc, normalize=True), os.path.join(save_path, '%s.png' % (fname_list[i * batch_size + j])))
            else:
                vae_util.save_image(vae_util.tensor2label(generated.data[j], vae_opt.output_nc, normalize=True), os.path.join(save_path, '%s.png' % (fname_list[i * batch_size + j])))
    # Remaining instance in the last batch needs to be generated here
    if elem_in_last_batch > 0:
        generated = model.generate_from_random(torch.Tensor(features_mat[-elem_in_last_batch:, :]).cuda())
        for j in range(elem_in_last_batch):
            if black:
                vae_util.save_image(vae_util.tensor2label_black(generated.data[j], vae_opt.output_nc, normalize=True), os.path.join(save_path, '%s.png' % (fname_list[-elem_in_last_batch + j])))
            else:
                vae_util.save_image(vae_util.tensor2label(generated.data[j], vae_opt.output_nc, normalize=True), os.path.join(save_path, '%s.png' % (fname_list[-elem_in_last_batch + j])))
