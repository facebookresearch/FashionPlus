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
import torch
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np

# from . import spectral_normalization
from .spectral_normalization import SpectralNorm


###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def weights_init_debug(m):
    assert(False)
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.fill_(0.05)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.fill_(0.05)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[], deterministic_vae = False, pseudo_vae = False, debug = False):
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    elif netG == 'local':
        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
                                  n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    elif netG == 'vaencoder':
        netG = VAEncoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer, deterministic_vae = deterministic_vae, pseudo_vae = pseudo_vae)
    else:
        raise('generator not implemented!')
    # print(netG)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netG.cuda(gpu_ids[0])
    if debug:
        netG.apply(weights_init_debug)
    else:
        netG.apply(weights_init)
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[], debug=False):
    if norm == 'spectral':
        netD = MultiscaleSNDiscriminator(input_nc, ndf, n_layers_D, use_sigmoid, num_D, getIntermFeat)
    else:
        norm_layer = get_norm_layer(norm_type=norm)
        netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    # print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    if norm != 'spectral': # Spectral uses pytorch native initialization
        netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())
        return loss

class VGGLosses(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLosses, self).__init__()
        self.vgg = Vgg19().cuda()
        self.perceptual_criterion = nn.L1Loss()
        self.perceptual_weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]
        self.style_weights = [1.0/(64**2), 1.0/(128**2), 1.0/(256**2), 1.0/(512**2), 1.0/(512**2)]

    def forward(self, x, y):
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        perceptual_loss = 0
        style_loss = 0
        for i in range(len(x_vgg)):
            perceptual_loss += self.perceptual_weights[i] * self.perceptual_criterion(x_vgg[i], y_vgg[i].detach())
            if i < (len(x_vgg)-1):
                style_loss += self.style_criterion(x_vgg[i], y_vgg[i].detach())
        return perceptual_loss, style_loss

    def style_criterion(self, x, y):
        criterion = nn.MSELoss()
        return criterion(self.GramMatrix(x), self.GramMatrix(y))

    def GramMatrix(self, x):
        # G(x) = 1/M(x)N(x) * <F(x),F(x)>
        b, c, h, w = x.size()
        F = x.view(b, c, h * w)
        G = torch.bmm(F, F.transpose(1, 2))
        # G.div_(h * w)
        G.div_(c * h * w)
        return G




##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                               norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]

            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()
        activation = nn.ReLU(True)

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
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

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()
        ###################################################################################
        # Original init
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone() # Accumulate gradients to outputs
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        list_of_feat = torch.zeros(self.output_nc * len(inst_list))
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4
                if indices.nelement() == 0: # If this image b does not have label i in it, then no need to go though following for loop
                    continue
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat
            #         list_of_feat[count*self.output_nc + j] = mean_feat[0]
            # count += 1
        return outputs_mean

    def forward_fast(self, input, inst):
        assert(False)
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone() # Accumulate gradients to outputs
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        # list_of_feat = torch.zeros(self.output_nc * len(inst_list))
        # count = 0
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i))
                if indices.nonzero().nelement() == 0: # If this image b does not have label i in it, then no need to go though following for loop
                    continue
                outputs_b = outputs[b,:].unsqueeze(0)
                output_ins = outputs_b * indices.float() # Equivalent to masking while keeping shape
                outpus_sum = output_ins.sum(dim=2).sum(dim=2)
                num_indices = indices.float().sum(dim=2).sum(dim=2)
                mean_feat = outpus_sum / num_indices
                psuedo_mean_feat = mean_feat.view(-1, self.output_nc, 1, 1).expand_as(outputs_b)
                expand_indices = indices.expand_as(outputs_b)
                outputs_mean[b, expand_indices[0,]] = psuedo_mean_feat[0, expand_indices[0,]]
            #     list_of_feat[count*self.output_nc: (count+1)*self.output_nc] = mean_feat
            # count += 1
        return outputs_mean

class VAEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d, \
                deterministic_vae = False, pseudo_vae = False):
        super(VAEncoder, self).__init__()
        print('Construct VAE')
        self.output_nc = output_nc
        self.deterministic_vae = deterministic_vae
        self.pseudo_vae = pseudo_vae

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        self.fc1 =  nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()])
        self.fc2 =  nn.Sequential(*[nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()])
        self.model = nn.Sequential(*model)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp()
        if self.deterministic_vae:
            eps = self.get_z_deterministic(std.size(0))
        elif self.pseudo_vae:
            eps = self.get_z_random_pseudo(std.size(0))
        else:
            eps = self.get_z_random(std.size(0))
        return eps.mul(std).add(mu)

    def get_z_random(self, nz, random_type='gauss'):
        if random_type == 'uni':
            z = torch.rand(nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(nz)
        return z.cuda()

    def get_z_random_pseudo(self, nz):
        return torch.zeros(nz).cuda()

    def get_z_deterministic(self, nz):
        return torch.ones(nz).cuda()

    def forward(self, input, inst, is_background=False):
        outputs = self.model(input)
        outputs_mu = self.fc1(outputs)
        if is_background:
            # (B, C, H, W) -> (B, C)
            return torch.mean(torch.mean(outputs_mu, dim=3), dim=2)
        else:
            # instance-wise average pooling
            outputs_mean_mu = torch.zeros_like(outputs_mu)
            # outputs_mean_logvar = torch.zeros_like(outputs_logvar) # We won't use logvar
            inst_list = np.unique(inst.cpu().numpy().astype(int))
            for i in inst_list:
                for b in range(input.size()[0]):
                    indices = (inst[b:b+1] == int(i)).nonzero() # n (row)  x 4 (col) matrix, for example: [[0,0,0,0],[0,0,0,1],....[0,0,255,254],[0,0,255,255]]
                    # each of the n row is a position where inst == i, and the columns in each row specify the batch-idx, channel-idx, x-pos, y-pos of the pixel
                    if indices.nelement() == 0: # If this image b does not have label i in it, then no need to go though following for loop
                        continue
                    for j in range(self.output_nc): # Need the for loop because we cannot access indices[:,1]:indices[:,1]+self.output_nc at the same time
                        # mu
                        output_mu_ins = outputs_mu[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                        mean_feat = torch.mean(output_mu_ins).expand_as(output_mu_ins)
                        outputs_mean_mu[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat

            return outputs_mean_mu

    def forward_fast(self, input, inst):
        assert(False)
        outputs = self.model(input)
        outputs_mu = self.fc1(outputs)

        # instance-wise average pooling
        outputs_mean_mu = torch.zeros_like(outputs_mu) # Accumulate gradients to outputs
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        # list_of_feat = torch.zeros(self.output_nc * len(inst_list))
        # count = 0
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i))
                if indices.nonzero().nelement() == 0: # If this image b does not have label i in it, then no need to go though following for loop
                    continue
                outputs_b = outputs_mu[b,:].unsqueeze(0)
                output_mu_ins = outputs_b * indices.float() # Equivalent to masking while keeping shape
                outpus_sum = output_mu_ins.sum(dim=2).sum(dim=2)
                num_indices = indices.float().sum(dim=2).sum(dim=2)
                mean_feat = outpus_sum / num_indices
                psuedo_mean_feat = mean_feat.view(-1, self.output_nc, 1, 1).expand_as(outputs_b)
                expand_indices = indices.expand_as(outputs_b)
                outputs_mean_mu[b, expand_indices[0,]] = psuedo_mean_feat[0, expand_indices[0,]]
            #     list_of_feat[count*self.output_nc: (count+1)*self.output_nc] = mean_feat
            # count += 1
        return outputs_mean_mu

    def forward_and_reparameterize_fast(self, input, inst): #, reparam_list):
        assert(False)
        outputs = self.model(input)
        outputs_mu = self.fc1(outputs)
        outputs_logvar = self.fc2(outputs)

        # instance-wise average pooling
        outputs_mean = torch.zeros_like(outputs_mu)
        inst_list = np.unique(inst.cpu().numpy().astype(int))
        batch_mu = torch.zeros(input.size()[0], outputs_mu.size()[1], len(inst_list)).cuda()
        batch_logvar = torch.zeros(input.size()[0], outputs_mu.size()[1], len(inst_list)).cuda()
        for count_i,i in enumerate(inst_list):
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i))
                if indices.nonzero().nelement() == 0: # If this image b does not have label i in it, then no need to go though following for loop
                    continue
                num_indices = indices.float().sum(dim=2).sum(dim=2)
                # mu
                outputs_b = outputs_mu[b,:].unsqueeze(0)
                output_mu_ins = outputs_b * indices.float() # Equivalent to masking while keeping shape
                outpus_sum = output_mu_ins.sum(dim=2).sum(dim=2)
                batch_mu[b, :, count_i] = outpus_sum / num_indices
                # logvar
                outputs_b = outputs_logvar[b,:].unsqueeze(0)
                output_logvar_ins = outputs_b * indices.float() # Equivalent to masking while keeping shape
                outpus_sum = output_logvar_ins.sum(dim=2).sum(dim=2)
                batch_logvar[b, :, count_i] = outpus_sum / num_indices
                # reparametrization trick
                # reparam_mean = reparam_list[b, :, count_i]
                reparam_mean = self.reparameterize(batch_mu[b, :, count_i].clone(), batch_logvar[b, :, count_i].clone()) # make sure the reparameterize does not modify their values in-place
                psuedo_mean_feat = reparam_mean.view(-1, self.output_nc, 1, 1).expand_as(outputs_b)
                expand_indices = indices.expand_as(outputs_b)
                outputs_mean[b, expand_indices[0,]] = psuedo_mean_feat[0, expand_indices[0,]]
        return outputs_mean, batch_mu, batch_logvar, inst_list


    def forward_and_reparameterize(self, input, inst, bk_z):
        outputs = self.model(input)
        outputs_mu = self.fc1(outputs)
        outputs_logvar = self.fc2(outputs)

        # instance-wise average pooling
        outputs_mean = torch.zeros_like(outputs_mu)
        inst_list = np.unique(inst.cpu().numpy().astype(int))

        # Processing background
        if (bk_z is not None) and (0 in inst_list):
            # 1) # Exclude background
            inst_list = inst_list[1:]
            # 2) Broadcast pre-encoded background code to background indices
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == 0).nonzero() # n (row)  x 4 (col) matrix, for example: [[0,0,0,0],[0,0,0,1],....[0,0,255,254],[0,0,255,255]]
                # each of the n row is a position where inst == i, and the columns in each row specify the batch-idx, channel-idx, x-pos, y-pos of the pixel
                if indices.nelement() == 0: # If this image b does not have label i in it, then no need to go though following for loop
                    continue
                for j in range(self.output_nc): # Two for loops because reparameterization trick needs to be done on the entire vector, not a single dimension j
                    output_ins = outputs_logvar[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    reparam_mean_feat = bk_z[b, j].expand_as(output_ins) # CHECK bk_z shape
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = reparam_mean_feat

        batch_mu = torch.zeros(input.size()[0], outputs_mu.size()[1], len(inst_list)).cuda() # shape is: batchSize, feature dimension, number of unique labels
        batch_logvar = torch.zeros(input.size()[0], outputs_mu.size()[1], len(inst_list)).cuda()
        for count_i,i in enumerate(inst_list):
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n (row)  x 4 (col) matrix, for example: [[0,0,0,0],[0,0,0,1],....[0,0,255,254],[0,0,255,255]]
                # each of the n row is a position where inst == i, and the columns in each row specify the batch-idx, channel-idx, x-pos, y-pos of the pixel
                if indices.nelement() == 0: # If this image b does not have label i in it, then no need to go though following for loop
                    continue
                for j in range(self.output_nc): # Need the for loop because we cannot access indices[:,1]:indices[:,1]+self.output_nc at the same time
                    # mu
                    output_mu_ins = outputs_mu[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    mean_mu = torch.mean(output_mu_ins)
                    batch_mu[b, j, count_i] = mean_mu
                    # logvar
                    output_logvar_ins = outputs_logvar[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    mean_logvar = torch.mean(output_logvar_ins)
                    batch_logvar[b, j, count_i] = mean_logvar
                # reparametrization trick
                reparam_mean = self.reparameterize(batch_mu[b, :, count_i].clone(), batch_logvar[b, :, count_i].clone()) # make sure the reparameterize does not modify their values in-place
                for j in range(self.output_nc): # Two for loops because reparameterization trick needs to be done on the entire vector, not a single dimension j
                    output_ins = outputs_logvar[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]
                    reparam_mean_feat = reparam_mean[j].expand_as(output_ins)
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = reparam_mean_feat
        return outputs_mean, batch_mu, batch_logvar, inst_list



class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

# COULD BE FURTHER SIMPLIFIED AND MERGED
class MultiscaleSNDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, use_sigmoid=False,
                 num_D=3, getIntermFeat=False):
        super(MultiscaleSNDiscriminator, self).__init__()
        print('Initializing Spectral Norm Discriminator')
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):
            netD = NLayerSNDiscriminator(input_nc, ndf, n_layers, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerSNDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 use_sigmoid=False, getIntermFeat=False):
        super(NLayerSNDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2))
        self.conv1 = SpectralNorm(
            nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw))
        # self.conv1 = nn.Conv2d(input_nc, ndf, kw, stride=2, padding=padw)

        self.conv_others = []
        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            # self.conv_others.append(
            #     nn.Conv2d(nf_prev, nf,
            #                   kernel_size=kw, stride=2,
            #                   padding=padw))
            self.conv_others.append(
                SpectralNorm(
                    nn.Conv2d(nf_prev, nf,
                              kernel_size=kw, stride=2,
                              padding=padw)))

        sequence = [[self.conv1,
                     nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                self.conv_others[n - 1],
                # norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        self.conv_final1 = SpectralNorm(
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw))
        # self.conv_final1 = nn.Conv2d(
        #     nf_prev, nf, kernel_size=kw, stride=1, padding=padw)
        sequence += [[
            self.conv_final1,
            # norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        self.conv_final2 = SpectralNorm(
            nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw))
        # self.conv_final2 = nn.Conv2d(
        #     nf, 1, kernel_size=kw, stride=1, padding=padw)
        sequence += [[self.conv_final2]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
