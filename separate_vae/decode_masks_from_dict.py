# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pickle
import numpy as np

import torch
from torch.autograd import Variable

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
import util.util as util
from util.visualizer import Visualizer

opt = TestOptions().parse()
if opt.share_decoder and opt.share_encoder:
    if opt.separate_clothing_unrelated:
        from models.separate_clothing_encoder_models import create_model
    else:
        print('Only supports separating clothing and clothing-irrelevant')
        raise NotImplementedError
else:
    print('Only supports sharing encoder and decoder among all parts')
    raise NotImplementedError

util.mkdirs(opt.results_dir)
black = True
model = create_model(opt)

# Read in the saved feature dict
with open(os.path.join(opt.load_feat_dir, 'shape_codes_dict.p'), 'rb') as readfile:
    fname_feature_dict = pickle.load(readfile)

# Compose the feautre dict into batches of tensor for efficiency
fname_list = []
num_feat = opt.nz * (1 + len(model.clothing_labels))
features_mat = np.zeros((len(fname_feature_dict), num_feat))
count = 0
for f in fname_feature_dict:
    fname_list.append(f)
    features_mat[count, :] = fname_feature_dict[f]
    count += 1

# Batch generate the masks
dataset_size = len(fname_list)
num_batch = int(dataset_size / opt.batchSize)
elem_in_last_batch = dataset_size % opt.batchSize
for i in range(num_batch):
    generated = model.generate_from_random(torch.Tensor(features_mat[i * opt.batchSize: (i+1) * opt.batchSize, :]).cuda())
    for j in range(opt.batchSize):
        if black:
            util.save_image(util.tensor2label_black(generated.data[j], opt.output_nc, normalize=True), os.path.join(opt.results_dir, '%s.png' % (fname_list[i * opt.batchSize + j])))
        else:
            util.save_image(util.tensor2label(generated.data[j], opt.output_nc, normalize=True), os.path.join(opt.results_dir, '%s.png' % (fname_list[i * opt.batchSize + j])))

# Remaining instance in the last batch yet to be generated
if elem_in_last_batch > 0:
    generated = model.generate_from_random(torch.Tensor(features_mat[-elem_in_last_batch:, :]).cuda())
    for j in range(elem_in_last_batch):
        if black:
            util.save_image(util.tensor2label_black(generated.data[j], opt.output_nc, normalize=True), os.path.join(opt.results_dir, '%s.png' % (fname_list[-elem_in_last_batch + j])))
        else:
            util.save_image(util.tensor2label(generated.data[j], opt.output_nc, normalize=True), os.path.join(opt.results_dir, '%s.png' % (fname_list[-elem_in_last_batch + j])))
