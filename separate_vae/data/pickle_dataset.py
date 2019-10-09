# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from PIL import Image
import numpy as np
import pickle
import cv2

class PickleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dataroot = opt.dataroot

        ### Read in picke file
        with open(os.path.join(opt.dataroot, '{}.p'.format(opt.phase)), 'rb') as readfile:
            self.pickle_file = pickle.load(readfile)

        ### input A (label maps)
        if opt.label_dir:
            self.dir_A = opt.label_dir

        self.dataset_size = len(self.pickle_file)

    def __getitem__(self, index):
        ### person bbox (used by all inputs)
        bbox = self.pickle_file[index]['bbox']
        ### input A (label maps)
        A_path = os.path.join(self.dir_A, self.pickle_file[index]['seg'])
        A = Image.open(A_path)
        A = self.crop_person(A, bbox)
        params = get_params(self.opt, A.size)
        if self.opt.output_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0
        input_dict = {'input': A_tensor, 'path': A_path, 'filename': self.pickle_file[index]['seg']}

        return input_dict

    def __len__(self):
        return len(self.pickle_file) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'

    def crop_person(self, img, bbox):
        # Image crop
        return img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))) # left, upper, right, lower

    def remove_background(self, img, label):
        img = np.array(img)
        mask = np.array(label)
        img[mask==0] = 0
        return Image.fromarray(img)

    def pil_rgb2lab(self, img):
        img = np.array(img)
        lab_img = cv2.cvtColor(img,cv2.COLOR_RGB2LAB)
        return Image.fromarray(lab_img)

    def pil_lab2rgb(self, img):
        img = np.array(img)
        print('lab2rgb', np.max(img[:,:,0]), np.max(img[:,:,1]), np.max(img[:,:,2]))
        rgb_img = cv2.cvtColor(img,cv2.COLOR_LAB2RGB)
        return Image.fromarray(rgb_img)
