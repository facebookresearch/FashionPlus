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

def create_model(opt):
    if opt.model == 'pix2pixHD':
        from .pix2pixHD_model import Pix2PixHDModel, InferenceModel
        if opt.isTrain:
            model = Pix2PixHDModel()
        else:
            model = InferenceModel()
    else:
    	from .ui_model import UIModel
    	model = UIModel()
    model.initialize(opt)
    if opt.verbose:
        print("model [%s] was created" % (model.name()))

    if opt.isTrain and len(opt.gpu_ids):
        model = torch.nn.DataParallel(model, device_ids=opt.gpu_ids)

    return model
