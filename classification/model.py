# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import torch
import numpy as np

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    clf_models = [f for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pth" in f]
    if clf_models is None:
        return None
    clf_models.sort()
    last_model_name = clf_models[-1]
    return last_model_name

def load_network(opt, model, which_epoch, save_dir=''):
    if which_epoch > 0:
        if opt.network_arch == 'mlp':
            save_filename = '%s_classifier.pth' % (which_epoch)
        else:
            save_filename = '%s_linear_classifier.pth' % (which_epoch)
        if not save_dir:
            save_dir = opt.save_dir
    else:
        if opt.network_arch == 'mlp':
            save_filename = get_model_list(save_dir, 'classifier')
        else:
            save_filename = get_model_list(save_dir, 'linear_classifier')

    if not save_filename:
        print("file does not exist")
        exit()
    save_path = os.path.join(save_dir, save_filename)

    if not os.path.isfile(save_path):
        print('%s not exists yet!' % save_path)
        exit()
    else:
        try:
            model.load_state_dict(torch.load(save_path))
        except:
            print("cannot load pretrained model")
            exit()

    if which_epoch > 0:
        print('Resume from iteration %d' % which_epoch)
        return which_epoch
    else:
        iterations = int(save_filename.split('_')[0])
        print('Resume from iteration %d' % iterations)
        return iterations

def save_network(opt, model, which_epoch):
    if opt.network_arch == 'mlp':
        save_filename = '%s_classifier.pth' % (which_epoch)
    else:
        save_filename = '%s_linear_classifier.pth' % (which_epoch)
    save_dir = os.path.join(opt.save_dir, opt.how_to_swap, opt.feature_type)
    # if opt.shape_and_texture:
    #     save_dir = os.path.join(opt.save_dir, opt.how_to_swap, 'shape_and_texture')
    # else:
    #     save_dir = os.path.join(opt.save_dir, opt.how_to_swap)
    # mkdir(save_dir)
    # torch.save(model.cpu().state_dict(), os.path.join(save_dir, save_filename))
    print(save_dir)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def accumulate_acc(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred, 1) == label)
    return np.sum(np.float32(test_np))

def separate_accumulate_acc(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    # print(np.argmax(pred[label==1,:], 1))
    pos_np = (np.argmax(pred[label==1,:], 1) == 1)
    # print(pos_np)
    neg_np = (np.argmax(pred[label==0,:], 1) == 0)
    return np.sum(np.float32(pos_np)), np.sum(np.float32(neg_np)), np.sum(label==1), np.sum(label==0)



def evaluate_acc(pred, label):
    pred = pred.cpu().data.numpy()
    label = label.cpu().data.numpy()
    test_np = (np.argmax(pred, 1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)
