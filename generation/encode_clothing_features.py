# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
import pickle
import numpy as np
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

opt = TrainOptions().parse()
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.instance_feat = True

name = '%s_features' % opt.phase
if opt.name == 'demo':
	save_path = os.path.join('results/Lab/', opt.name)
else:
	save_path = os.path.join(opt.checkpoints_dir, opt.name)


############ Initialize #########
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
model = create_model(opt)

########### Encode features ###########
reencode = True
simple_encode = False
fname_feature_dict = dict()
count = 0
if reencode:
    for i, data in enumerate(dataset):
        assert(opt.batchSize==1) # Only implemented bacthSize==1
        fname = os.path.split(data['path'][0])[-1][:-4]

        if simple_encode:
            feat = model.module.simple_encode_features(data['image'], data['label'])
            inst_np = data['label'][0].cpu().numpy().astype(int)
            for label in np.unique(inst_np):
                fname_feature_dict['_'.join([fname, str(label)])] = feat[label]
        else:
            feat = model.module.encode_features(data['image'], data['label'])
            inst_np = data['label'][0].cpu().numpy().astype(int)
            for label in np.unique(inst_np):
                fname_feature_dict['_'.join([fname, str(label)])] = feat[label][:, :-1] # Last dimension ineffective
        print('%d / %d images' % (i+1, dataset_size))

    save_name = os.path.join(save_path, name + '.p')
    with open(save_name, 'wb') as writefile:
        pickle.dump(fname_feature_dict, writefile)
