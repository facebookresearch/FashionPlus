# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np


class MLP(nn.Module):
    def __init__(self, opt):
        ''' Args: opt.in_dim (int): input feature dimension
                                    (concatenated shape and texture encodings for all parts)
                  opt.out_dim (int): output feature dimension
                                     for binary classification, it is always 2
                  opt.param_m (int): number of layers after the input layer
                  opt.param_k (int): number of neurons for each layer
        '''
        super(MLP, self).__init__()
        model = []
        if opt.param_m > 1:
            post_dim = opt.param_k
            model += [nn.Linear(opt.in_dim, post_dim), nn.ReLU()]
            if opt.use_dropout:
                model += [nn.Dropout(0.2)]
            prev_dim = post_dim
        else:
            post_dim = opt.param_k / 2
            prev_dim = opt.in_dim

        for i in range(opt.param_m-1):
            model += [nn.Linear(prev_dim, post_dim), nn.ReLU()]
            if opt.use_dropout:
                model += [nn.Dropout(0.2)]
        # print(prev_dim)
        # print(opt.param_k / 2)
        if opt.use_dropout:
            model += [nn.Linear(prev_dim, int(opt.param_k / 2)), nn.ReLU(), nn.Dropout(0.2), nn.Linear(int(opt.param_k / 2), opt.out_dim)]
        else:
            model += [nn.Linear(prev_dim, int(opt.param_k / 2)), nn.ReLU(), nn.Linear(int(opt.param_k / 2), opt.out_dim)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class LinearClassifier(nn.Module):
    def __init__(self, opt):
        ''' Args: opt.in_dim (int): input feature dimension
                  opt.fc2_dim (int): output feature dimension
        '''
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(opt.in_dim, opt.fc2_dim)

    def forward(self, x):
        return self.fc1(x)
