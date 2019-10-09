# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
##############################################################################
#
# Based on:
# bicycleGAN (https://github.com/junyanz/BicycleGAN)
### License: https://github.com/junyanz/BicycleGAN/blob/master/LICENSE.
import os
import sys
import numpy as np
import torch
from . import pix2pixhd_networks as networks
import pickle

def create_model(opt):
    model = VAE_MODEL()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model

class VAE_MODEL(torch.nn.Module):
    def name(self):
        return 'VAE_MODEL'

    def initialize(self, opt):
        self.use_vae = True and opt.isTrain # Inference time uses regular autoencoder
        self.isTrain = opt.isTrain
        self.gpu_ids = opt.gpu_ids
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        # Get the labels that are clothing related
        clothing_labels = np.loadtxt(opt.label_txt_path , delimiter=',', dtype=int)
        self.clothing_labels = clothing_labels.tolist()
        self.not_clothing_labels = []
        for i in range(opt.output_nc):
            if i not in self.clothing_labels:
                self.not_clothing_labels.append(i)
        num_clothing_irrelevant_labels = len(self.not_clothing_labels)
        # Define encoders for each label, and a shared decoder
        self.Separate_encoder, self.Together_encoder, self.Decoder = networks.define_separate_Es_and_D(num_clothing_irrelevant_labels, opt.output_nc, opt.nz, opt.nef,
                                                                opt.divide_by_K, opt.bottleneck, opt.n_downsample_global, opt.n_blocks_global,
                                                                opt.max_mult, opt.norm, gpu_ids=self.gpu_ids, vaeLike=self.use_vae)
        # print(self.Separate_encoder)
        # print(self.Together_encoder)
        # print(self.Decoder)

        # set up optimizer
        if self.isTrain:
            if opt.bottleneck == '2d': # Deprecated...
                self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                params = list(self.Separate_encoder.parameters())
                params += list(self.Together_encoder.parameters())
                params += list(self.Decoder.parameters())
                self.optimizer = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))
            self.criterionMSE = torch.nn.MSELoss()
            self.loss_names = ['MSE', 'KL']
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            self.old_lr = self.optimizer.param_groups[0]['lr']
        self.opt = opt

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            self.load_network(self.Separate_encoder, 'Separate_encoder', opt.which_epoch, pretrained_path)
            self.load_network(self.Together_encoder, 'Together_encoder', opt.which_epoch, pretrained_path)
            self.load_network(self.Decoder, 'Decoder', opt.which_epoch, pretrained_path)


    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        self.old_lr = self.optimizer.param_groups[0]['lr']
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def set_continue_learning_rate(self, last_epoch):
        self.scheduler = networks.get_scheduler(self.optimizer, self.opt, last_epoch)
        self.old_lr = self.optimizer.param_groups[0]['lr']


    def encode(self, encoder, input_image):
        '''Forward input into encoder and perform reparametrization trick
           Args: encoder, VAE's encoder
                 input_image, input tensor to be encoded
           Return: z (tensor), encoded vector
                   mu, mean
                   logvar, variance
        '''
        mu, logvar = encoder(input_image)
        std = logvar.mul(0.5).exp_()
        eps = self.get_z_random(std.size(0), std.size(1))
        z = eps.mul(std).add_(mu)
        return z, mu, logvar


    def forward2D(self, real_B_encoded, infer=False):
        # get encoded z
        real_B_encoded = self.one_hot_tensor(real_B_encoded)
        real_B_encoded = real_B_encoded - 0.5 # normalized to [-0.5, 0.5]
        # generate fake_B_encoded
        fake_B_encoded = self.netG(real_B_encoded)
        fake_B_encoded = fake_B_encoded - 0.5 # normalized to [-0.5, 0.5]

        # 2. KL loss
        if self.opt.lambda_kl > 0.0 and self.use_vae:
            kl_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
            loss_kl = torch.sum(kl_element).mul_(-0.5) * self.opt.lambda_kl
        else:
            loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            loss_MSE = self.criterionMSE(fake_B_encoded, real_B_encoded) * self.opt.lambda_L1
        else:
            loss_MSE = 0.0

        return [loss_MSE, loss_kl], real_B_encoded, None if not infer else fake_B_encoded

    def forward(self, real_B_encoded, infer=False, sample=False, epoch=0, epoch_iter=0):
        '''Forward input image into VAE
           Args: real_B_encoded (tensor), input tensor image
                 infer (boolean), whether to return the decoded image
                 sample (boolean), epoch (int), epoch_iter (int): deprecated
           Return: loss_MSE, reconstruction loss
                   loss_kl, kl-divergence loss
                   real_B_encoded: first binarized then normalized input image tensor
                   fake_B_encoded (optional): decoded image
        '''
        # get encoded z
        real_B_encoded = self.one_hot_tensor(real_B_encoded)
        real_B_encoded = real_B_encoded - 0.5 # normalized to [-0.5, 0.5]

        # separately forward each label
        zs_encoded = torch.zeros(self.opt.batchSize, self.opt.nz * (len(self.clothing_labels)+1) ).cuda()
        list_of_mus = []
        list_of_logvars = []
        if self.use_vae:
             # Clothing related
            for count_i, label_i in enumerate(self.clothing_labels):
                z_encoded, mu, logvar = self.encode(self.Separate_encoder, real_B_encoded[:,label_i].unsqueeze(1))
                zs_encoded[:, count_i*self.opt.nz: (count_i+1)*self.opt.nz] = z_encoded
                list_of_mus.append(mu)
                list_of_logvars.append(logvar)
            # Clothing unrelated
            z_encoded, mu, logvar = self.encode(self.Together_encoder, real_B_encoded[:,self.not_clothing_labels])
            zs_encoded[:, -1*self.opt.nz:] = z_encoded
            list_of_mus.append(mu)
            list_of_logvars.append(logvar)
        else: # regular auto-encoder
            for count_i, label_i in enumerate(self.clothing_labels):
                zs_encoded[:, count_i*self.opt.nz: (count_i+1)*self.opt.nz] = self.Separate_encoder(real_B_encoded[:,label_i].unsqueeze(1))
            zs_encoded[:, -1*self.opt.nz:] = self.Together_encoder(real_B_encoded[:,self.not_clothing_labels])

        # generate fake_B_encoded
        fake_B_encoded = self.Decoder(zs_encoded)
        fake_B_encoded = fake_B_encoded - 0.5 # normalized to [-0.5, 0.5]

        # 2. KL loss
        if self.opt.lambda_kl > 0.0 and self.use_vae:
            loss_kl = 0
            for i in range(len(self.clothing_labels) + 1):
                kl_element = list_of_mus[i].pow(2).add_(list_of_logvars[i].exp()).mul_(-1).add_(1).add_(list_of_logvars[i])
                loss_kl += torch.sum(kl_element).mul_(-0.5)
            loss_kl = loss_kl * self.opt.lambda_kl
        else:
            loss_kl = 0
        # 3, reconstruction |fake_B-real_B|
        if self.opt.lambda_L1 > 0.0:
            loss_MSE = self.criterionMSE(fake_B_encoded, real_B_encoded) * self.opt.lambda_L1
        else:
            loss_MSE = 0.0
        return [loss_MSE, loss_kl], real_B_encoded, None if not infer else fake_B_encoded
        # loss_G = loss_G_L1 + loss_kl

    def encode_features(self, real_B_encoded):
        ''' Encode input image into latent codes at inference time
            Args: real_B_encoded, original input image tensor
            Return: zs_encoded, concatenated latent codes
                    number of labels
        '''
        # get encoded z
        real_B_encoded = self.one_hot_tensor(real_B_encoded)
        real_B_encoded = real_B_encoded - 0.5 # normalized to [-0.5, 0.5]

        with torch.no_grad():
            # separately forward each label
            zs_encoded = torch.zeros(self.opt.batchSize, self.opt.nz * (len(self.clothing_labels)+1) ).cuda()

            # Inference does not need reparametrization
            for count_i, label_i in enumerate(self.clothing_labels):
                zs_encoded[:, count_i*self.opt.nz: (count_i+1)*self.opt.nz] = self.Separate_encoder(real_B_encoded[:,label_i].unsqueeze(1))
            zs_encoded[:, -1*self.opt.nz:] = self.Together_encoder(real_B_encoded[:,self.not_clothing_labels])
            return zs_encoded, (len(self.clothing_labels)+1)

    def inference_reconstruct(self, real_B_encoded):
        ''' Reconstruct input image with regular autoencoder at inference time
            (Mostly used for sanity checks)
            Args: real_B_encoded, original input image tensor
            Return: fake_B_encoded, decoded (reconsturcted) image
        '''
        # get encoded z
        real_B_encoded = self.one_hot_tensor(real_B_encoded)
        real_B_encoded = real_B_encoded - 0.5 # normalized to [-0.5, 0.5]
        with torch.no_grad():
            if self.use_vae:
                z_encoded, mu, logvar = self.encode(real_B_encoded)
            else: # regular auto-encoder
                z_encoded = self.Encoder(real_B_encoded)
            # generate fake_B_encoded
            fake_B_encoded = self.Decoder(z_encoded)
            fake_B_encoded = fake_B_encoded - 0.5 # normalized to [-0.5, 0.5]
            print(torch.nn.MSELoss()(fake_B_encoded, real_B_encoded))
        return fake_B_encoded

    def generate_from_random(self, z):
        ''' Generate an image from latent code z
            Args: z, latent code; can be randomly sampled or pre-encoded
            Return: decoded image
        '''
        with torch.no_grad():
            print(z.shape)
            return (self.Decoder(z)-0.5) # normalized to [-0.5, 0.5]

    def one_hot_tensor(self, label_map):
        ''' Convert the original HxW label_map to nCxHxW one hot volume
            Args: label_map (batch_sz x H x W tensor), with value in range [0, output_nc)
            Return: input_label (batch_sz x nc x H x W tensor), each slice at the channel dimension is a one hot map
        '''
        size = label_map.size()
        oneHot_size = (size[0], self.opt.output_nc, size[2], size[3])
        input_label = torch.cuda.FloatTensor(torch.Size(oneHot_size)).zero_()
        input_label = input_label.scatter_(1, label_map.data.long().cuda(), 1.0)
        return input_label

    def get_z_random(self, batchSize, nz, random_type='gauss'):
        ''' Randomly sample a latent code (later to be decoded)
            Args: batchSize, number of instances in a batch
                  nz, dimension for latent code
                  random_type, randomly draw z from what distribution
            Return: z, randomly drawn code
        '''
        if random_type == 'uni':
            z = torch.rand(batchSize, nz) * 2.0 - 1.0
        elif random_type == 'gauss':
            z = torch.randn(batchSize, nz)
        return z.cuda()

    def save(self, which_epoch):
        if self.opt.bottleneck == '1d':
            self.save_network(self.Separate_encoder, 'Separate_encoder', which_epoch, self.gpu_ids)
            self.save_network(self.Together_encoder, 'Together_encoder', which_epoch, self.gpu_ids)
            self.save_network(self.Decoder, 'Decoder', which_epoch, self.gpu_ids)
        else:
            self.save_network(self.netG, 'netG', which_epoch, self.gpu_ids)

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):
        save_filename = '%s_%s.pth' % (epoch_label, network_label)
        if not save_dir:
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)
        if not os.path.isfile(save_path):
            print('%s not exists yet!' % save_path)
        else:
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:
                pretrained_dict = torch.load(save_path)
                model_dict = network.state_dict()
                try:
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
                    network.load_state_dict(pretrained_dict)
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():
                        if v.size() == model_dict[k].size():
                            model_dict[k] = v

                    if sys.version_info >= (3,0):
                        not_initialized = set()
                    else:
                        from sets import Set
                        not_initialized = Set()

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])

                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)
