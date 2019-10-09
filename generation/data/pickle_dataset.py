# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os.path
import random
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
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

        ### input B (real images)
        if opt.isTrain:
            if opt.img_dir:
                self.dir_B = opt.img_dir

        ### instance maps
        if not opt.no_instance:
            if opt.label_dir:
                self.dir_inst = opt.label_dir

        ### load precomputed instance-wise encoded features
        if opt.load_features:
            self.dir_feat = os.path.join(opt.dataroot, opt.phase + '_feat')
            print('----------- loading features from %s ----------' % self.dir_feat)
            self.feat_paths = sorted(make_dataset(self.dir_feat))

        self.dataset_size = len(self.pickle_file)

    def __getitem__(self, index):
        ### person bbox (used by all inputs)
        bbox = self.pickle_file[index]['bbox']
        ### input A (label maps)
        A_path = os.path.join(self.dir_A, self.pickle_file[index]['seg'])
        A = Image.open(A_path)
        A = self.crop_person(A, bbox)
        params = get_params(self.opt, A.size)
        if self.opt.label_nc == 0:
            transform_A = get_transform(self.opt, params)
            A_tensor = transform_A(A.convert('RGB'))
        else:
            transform_A = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
            A_tensor = transform_A(A) * 255.0

        random_B_tensor = B_tensor = inst_tensor = feat_tensor = 0
        ### if using instance maps
        if not self.opt.no_instance:
            inst_path = os.path.join(self.dir_A, self.pickle_file[index]['seg'])
            inst = Image.open(inst_path)
            inst = self.crop_person(inst, bbox)
            inst_tensor = transform_A(inst)

            if self.opt.load_features:
                feat_path = self.feat_paths[index]
                feat = Image.open(feat_path).convert('RGB')
                norm = normalize()
                feat_tensor = norm(transform_A(feat))

        ### input B (real images)
        if self.opt.isTrain:
            B_path = os.path.join(self.dir_B, self.pickle_file[index]['filename'])
            B = Image.open(B_path).convert('RGB')
            B = self.crop_person(B, bbox)
            B = self.remove_background(B, A)

            if self.opt.color_mode == 'Lab':
                B = self.pil_rgb2lab(B)
            if self.opt.color_mode == 'Lab':
                transform_B = get_transform(self.opt, params, labimg=True)
            else:
                transform_B = get_transform(self.opt, params)

            B_tensor = transform_B(B)

            # We need a random real image to train the discriminator
            if (self.opt.model == 'cvae++pix2pixHD') or (self.opt.model == 'bicycle-pix2pixHD'):
                random_index = random.randint(0, self.dataset_size-1)
                bbox = self.pickle_file[random_index]['bbox']
                ### input A (label maps) for removing background of random B
                A_path = os.path.join(self.dir_A, self.pickle_file[random_index]['seg'])
                A = Image.open(A_path)
                A = self.crop_person(A, bbox)
                ### input random B (real images)
                B_path = os.path.join(self.dir_B, self.pickle_file[random_index]['filename'])
                B = Image.open(B_path).convert('RGB')
                B = self.crop_person(B, bbox)
                B = self.remove_background(B, A)
                # print(B_path)
                if self.opt.color_mode == 'Lab':
                    B = self.pil_rgb2lab(B)

                random_B_tensor = transform_B(B)

                input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                              'feat': feat_tensor, 'path': A_path, 'random_image': random_B_tensor}
            else:
                input_dict = {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                              'feat': feat_tensor, 'path': A_path}

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
