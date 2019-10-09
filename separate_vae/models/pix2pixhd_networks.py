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
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler

###############################################################################
# Functions
###############################################################################

# Initialize weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# Initialize net
# Get normalize layer
def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

# Get non-linear layer
# Get scheduler
def get_scheduler(optimizer, opt, last_epoch=-1):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=last_epoch)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(
            optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler

# Define network
def define_ED(input_nc, z_nc, ngf, K, bottleneck, n_downsample_global=3, n_blocks_global=9,
              max_mult=16, norm='instance', gpu_ids=[], vaeLike=False):
    norm_layer = get_norm_layer(norm_type=norm)
    encoder = E_Resnet(1, z_nc, ngf, K, bottleneck, n_downsample_global, n_blocks_global, max_mult, norm_layer, vaeLike=vaeLike)
    decoder = D_NLayers(input_nc, z_nc*input_nc, ngf, K, bottleneck, n_downsample_global, n_blocks_global, max_mult, norm_layer, vaeLike=vaeLike)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        encoder.cuda(gpu_ids[0])
        decoder.cuda(gpu_ids[0])
    encoder.apply(weights_init)
    decoder.apply(weights_init)
    return encoder, decoder

def define_paired_EDs(num_pairs, input_nc, z_nc, ngf, K, bottleneck, n_downsample_global=3, n_blocks_global=9,
              max_mult=16, norm='instance', gpu_ids=[], vaeLike=False):
    norm_layer = get_norm_layer(norm_type=norm)
    list_of_encoders = []
    list_of_decoders = []
    for i in range(num_pairs):
        encoder = E_Resnet(input_nc, z_nc, ngf, K, bottleneck, n_downsample_global, n_blocks_global, max_mult, norm_layer, vaeLike=vaeLike)
        decoder = D_NLayers(input_nc, z_nc, ngf, K, bottleneck, n_downsample_global, n_blocks_global, max_mult, norm_layer, vaeLike=vaeLike)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            encoder.cuda(gpu_ids[0])
            decoder.cuda(gpu_ids[0])
        encoder.apply(weights_init)
        decoder.apply(weights_init)
        list_of_encoders.append(encoder)
        list_of_decoders.append(decoder)
    return list_of_encoders, list_of_decoders

def define_Es_shareD(num_pairs, input_nc, z_nc, ngf, K, bottleneck, n_downsample_global=3, n_blocks_global=9,
              max_mult=16, norm='instance', gpu_ids=[], vaeLike=False):
    norm_layer = get_norm_layer(norm_type=norm)
    list_of_encoders = []
    for i in range(num_pairs):
        encoder = E_Resnet(input_nc, z_nc, ngf, K, bottleneck, n_downsample_global, n_blocks_global, max_mult, norm_layer, vaeLike=vaeLike)
        if len(gpu_ids) > 0:
            assert(torch.cuda.is_available())
            encoder.cuda(gpu_ids[0])
        encoder.apply(weights_init)
        list_of_encoders.append(encoder)
    decoder = D_NLayers(num_pairs, z_nc*num_pairs, ngf, K, bottleneck, n_downsample_global, n_blocks_global, max_mult, norm_layer, vaeLike=vaeLike)
    decoder.apply(weights_init)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        decoder.cuda(gpu_ids[0])
    return list_of_encoders, decoder

def define_separate_Es_and_D(num_labels_together, input_nc, z_nc, ngf, K, bottleneck, n_downsample_global=3, n_blocks_global=9,
              max_mult=16, norm='instance', gpu_ids=[], vaeLike=False):
    norm_layer = get_norm_layer(norm_type=norm)
    separate_encoder = E_Resnet(1, z_nc, ngf, K, bottleneck, n_downsample_global, n_blocks_global, max_mult, norm_layer, vaeLike=vaeLike)
    together_encoder = E_Resnet(num_labels_together, z_nc, ngf, K, bottleneck, n_downsample_global, n_blocks_global, max_mult, norm_layer, vaeLike=vaeLike)
    num_labels_separate = input_nc-num_labels_together
    decoder = D_NLayers(input_nc, z_nc*(num_labels_separate+1), ngf, K, bottleneck, n_downsample_global, n_blocks_global, max_mult, norm_layer, vaeLike=vaeLike)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        separate_encoder.cuda(gpu_ids[0])
        together_encoder.cuda(gpu_ids[0])
        decoder.cuda(gpu_ids[0])
    separate_encoder.apply(weights_init)
    together_encoder.apply(weights_init)
    decoder.apply(weights_init)
    return separate_encoder, together_encoder, decoder


###############################################################################
# Network architecture
###############################################################################


class E_Resnet(nn.Module):
    def __init__(self, input_nc, z_nc, ngf=64, K=16, bottleneck='2d', n_downsampling=3, n_blocks=9, max_mult=16,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', vaeLike=False):
        assert(n_blocks >= 0)
        super(E_Resnet, self).__init__()
        self.vaeLike = vaeLike
        activation = nn.ReLU(True)
        ngf = int(ngf/K)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            # Clip at 1024
            if mult >= max_mult:
                model += [nn.Conv2d(ngf * max_mult, ngf * max_mult, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * max_mult), activation]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = min(2**n_downsampling, max_mult)
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        self.conv = nn.Sequential(*model)
        self.fc = nn.Sequential(*[nn.Linear(ngf * mult * 4, z_nc)]) # Because flatten will give us CxHxW
        if vaeLike:
            self.fcVar = nn.Sequential(*[nn.Linear(ngf * mult * 4, z_nc)]) # Because flatten will give us CxHxW

        self.bottleneck = bottleneck

    def forward(self, input):
        if self.bottleneck == '2d':
            return self.conv(input)
        else:
            input_conv = self.conv(input)
            conv_flat = input_conv.view(input.size(0), -1) # Flatten
            output = self.fc(conv_flat)
            if self.vaeLike:
                outputVar = self.fcVar(conv_flat)
                return output, outputVar
            return output


class D_NLayers(nn.Module):
    def __init__(self, input_nc, z_nc, ngf=64, K=16, bottleneck='2d', n_downsampling=3, n_blocks=9, max_mult=16,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', vaeLike=False):
        assert(n_blocks >= 0)
        super(D_NLayers, self).__init__()
        self.vaeLike = vaeLike
        self.bottleneck = bottleneck
        activation = nn.ReLU(True)
        ngf = int(ngf/K)

        if self.bottleneck == '1d':
            # Additional ConvTranspose2d to upsample from 1x1 to 2x2
            model = [nn.ConvTranspose2d(z_nc, ngf * max_mult, kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(ngf * max_mult)]
            mult = 2**(n_downsampling)
            if mult > max_mult:
                model += [nn.ConvTranspose2d(ngf * max_mult, ngf * max_mult, kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(ngf * max_mult), activation]
            else:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(int(ngf * mult / 2)), activation]
        else:
            mult = 2**(n_downsampling)
            if mult > max_mult:
                model = [nn.ConvTranspose2d(ngf * max_mult, ngf * max_mult, kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(ngf * max_mult), activation]
            else:
                model = [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(int(ngf * mult / 2)), activation]

        ### upsample
        for i in range(1, n_downsampling):
            mult = 2**(n_downsampling - i)
            if mult > max_mult:
                model += [nn.ConvTranspose2d(ngf * max_mult, ngf * max_mult, kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(ngf * max_mult), activation]
            else:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, input_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        self.conv = nn.Sequential(*model)


    def forward(self, z):
        if self.bottleneck == '2d':
            return self.conv(z)
        else:
            return self.conv(z.view(z.size() + (1, 1))) # Expand z to B C H W


class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, bottleneck='2d', n_downsampling=3, n_blocks=9, max_mult=16,
                 norm_layer=nn.BatchNorm2d, padding_type='reflect', vaeLike=False):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        self.vaeLike = vaeLike
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            # Clip at 1024
            if mult >= max_mult:
                model += [nn.Conv2d(ngf * max_mult, ngf * max_mult, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * max_mult), activation]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                          norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = min(2**n_downsampling, max_mult)
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            if mult > max_mult:
                model += [nn.ConvTranspose2d(ngf * max_mult, ngf * max_mult, kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(ngf * max_mult), activation]
            else:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                           norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Sigmoid()]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        return self.model(input)

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
