# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pdb
import json
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

# Initialize
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

model = create_model(opt)

demo = True
sanity_check = False
web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
img_dir = os.path.join(web_dir, 'images')
print('create web directory %s...' % web_dir)
util.mkdirs([web_dir, img_dir])

# Set label to partID mapping
label_dict = dict()
with open(opt.dataset_param_file, 'r') as readfile:
    garment_label_mapping = json.load(readfile)
for garment_name in garment_label_mapping:
    if garment_name != 'background':
        garment_label = garment_label_mapping[garment_name]['label']
        garment_partID = garment_label_mapping[garment_name]['partID']
        label_dict[garment_label] = garment_partID

# Save the extracted feature into dictionary
fname_feature_dict = dict()
for i, data in enumerate(dataset):
    fname = os.path.split(data['path'][0])[-1][:-4]
    # Call a function that encodes a segmentation mask into latent codes for segmentation labels
    inst_np = data['input'][0].cpu().numpy().astype(int)
    label_encodings, num_labels = model.encode_features(Variable(data['input']))
    # convert tensor to numpy
    label_encodings = label_encodings.data.cpu().numpy()
    for label in label_dict:
        fname_feature_dict['_'.join([fname, str(label)])] = \
                            label_encodings[:, label_dict[label]*opt.nz: (label_dict[label]+1)*opt.nz]

    fname_feature_dict['_'.join([fname, str(0)])] = label_encodings[:, -1*opt.nz:]

    if sanity_check:
        # Sanity check the label_encodings
        generated = model.generate_from_random(label_encodings)
        util.save_image(util.tensor2label(data['input'][0], opt.output_nc, normalize=False), os.path.join(img_dir, 'input_label_%s.jpg' % (fname)))
        util.save_image(util.tensor2label(generated.data[0], opt.output_nc, normalize=True), os.path.join(img_dir, 'synthesized_label_%s.jpg' % (fname)))

if demo:
    save_name = os.path.join('results/Lab/demo/', '%s_shape_codes.p' % opt.phase)
else:
    save_name = os.path.join(img_dir, '%s_shape_codes.p' % opt.phase)
with open(save_name, 'wb') as writefile:
    pickle.dump(fname_feature_dict, writefile)
