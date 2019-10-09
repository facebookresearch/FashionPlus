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
# and based on:
# bicycleGAN (https://github.com/junyanz/BicycleGAN)
### License: https://github.com/junyanz/BicycleGAN/blob/master/LICENSE.
import argparse
import os
from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, help='path to pickle file')
        self.parser.add_argument('--label_dir', type=str, help='path to directory containing images (of segmentaiton labels)')
        self.parser.add_argument('--label_txt_path', type=str, help='path to text file containing separation of labels')
        self.parser.add_argument('--batchSize', type=int, default=2, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--nz', type=int, default=8, help='#latent vector')
        self.parser.add_argument('--nef', type=int, default=64, help='# of encoder filters in first conv layer')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2, -1 for CPU mode')
        self.parser.add_argument('--name', type=str, default='', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='not implemented')
        self.parser.add_argument('--dataset_mode', type=str, default='aligned', help='aligned,single')
        self.parser.add_argument('--model', type=str, default='bicycle_gan', help='chooses which model to use. bicycle,, ...')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=4, type=int, help='# sthreads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom display port')
        self.parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        self.parser.add_argument('--log_to_filename', type=str, default="/checkpoint/kimberlyhsiao/.visdom/", help='visdom log file location')
        self.parser.add_argument('--use_dropout', action='store_true', help='use dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data argumentation')

        # models
        self.parser.add_argument('--share_decoder', action='store_true', help='if specified, combine the latent encoding to feed in to the shared decoder')
        self.parser.add_argument('--share_encoder', action='store_true', help='if specified, different labels will share the same encoder weights')
        self.parser.add_argument('--separate_clothing_unrelated', action='store_true', help='clothing related share an encoder while clothing unrelated share another')
        self.parser.add_argument('--which_model_netG', type=str, default='unet_256', help='selects model to use for netG')
        self.parser.add_argument('--which_model_netE', type=str, default='resnet_256', help='selects model to use for netE')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--upsample', type=str, default='basic', help='basic | bilinear')
        self.parser.add_argument('--nl', type=str, default='relu', help='non-linearity activation: relu | lrelu | elu')
        self.parser.add_argument('--n_downsample_global', type=int, default=4, help='number of downsampling layers in netG')
        self.parser.add_argument('--n_blocks_global', type=int, default=9, help='number of residual blocks in the global generator network')
        self.parser.add_argument('--max_mult', type=int, default=16, help='the maximum times to multiply ngf')
        self.parser.add_argument('--divide_by_K', type=int, default='16', help='dividing number of filters at each layer by K')
        self.parser.add_argument('--bottleneck', type=str, default='2d', help='bottleneck dimension: 1d | 2d')

        # extra parameters
        self.parser.add_argument('--where_add', type=str, default='all', help='input|all|middle; where to add z in the network G')
        self.parser.add_argument('--init_type', type=str, default='xavier', help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--center_crop', action='store_true', help='if apply for center cropping for the test')
        self.parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        self.parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{which_model_netG}_size{loadSize}')
        self.parser.add_argument('--tf_log', action='store_true', help='if specified, use tensorboard logging. Requires tensorflow installed')
        # special tasks
        self.initialized = True

    def get_opt(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_known_args()[0]
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])
        return self.opt

    def parse(self, save=True):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        if self.opt.suffix:
            suffix = ('_' + self.opt.suffix.format(**vars(self.opt))) if self.opt.suffix != '' else ''
            self.opt.name = self.opt.name + suffix
        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
