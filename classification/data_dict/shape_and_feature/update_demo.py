# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import copy
import json
import numpy as np
import os
import pdb
import pickle
import sys

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from classifier_options import option_parser


#######################################
############## Function ###############
#######################################

class StopCriterion:
    ''' Stop Criterion for our editing module
    '''
    def __init__(self, stop_criterion):
        self.stop_criterion = stop_criterion

    def delegate_child(self):
        if self.stop_criterion == 'maxiter':
            return MaxIterStopCriterion()
        elif self.stop_criterion == 'thresholdloss':
            return ThresholdStopCriterion()
        elif self.stop_criterion == 'deltaloss':
            return DeltaStopCriterion()
        else:
            print('Unsupported stop criterion')
            return None

class MaxIterStopCriterion(StopCriterion):
    ''' Stop criterion is when the number of iterations reaches maximum limit
    '''
    def __init__(self):
        print('init maxiter')
    def __call__(self, loss):
        return False # We use while loop to stop

class ThresholdStopCriterion(StopCriterion):
    ''' Stop criterion is when the the model's output loss is lower than a thershold
    '''
    def __init__(self):
        self.min_thresholdloss = argopt.min_thresholdloss
    def __call__(self, loss):
        if loss <= self.min_thresholdloss:
            return True
        else:
            return False

class DeltaStopCriterion(StopCriterion):
    ''' Stop criterion is when the model's output loss doesn't change more than delta
    '''
    def __init__(self):
        self.min_deltaloss = argopt.min_deltaloss
    def set_initial_loss(self, initial_loss):
        self.initial_loss = initial_loss
    def __call__(self, loss):
        delta_loss = loss - self.initial_loss
        if delta_loss > self.min_deltaloss:
            return True
        else:
            return False

class InputFeature:
    '''Wraps the numpy feature for classifier's input format in this class
    '''
    def __init__(self, shape_feat_num, texture_feat_num, part_num, init_feat=None):
        ''' Args: shape_feat_num (int), feature dimension for shape encoding
                  texture_feat_num (int), feature dimension for texture encoding
                  part_num (int), number of parts in an outfit
                  init_feat (numpy array), initialization for feature
        '''
        self.shape_feat_num = shape_feat_num
        self.texture_feat_num = texture_feat_num
        # Feature for each part is the concatenation of first texture then shape features;
        # The enitre feature is the concatenation of all part features
        if init_feat is not None:
            self.feature = init_feat
        else:
            self.feature = np.zeros((part_num * (self.shape_feat_num + self.texture_feat_num), ))
    def get_feature(self, partID, mode):
        '''Get shape and/or texture feature in partID
           Args: partID (int), which part of feature to retrieve
                 mode (str), one of "shape_only", "texture_only", "shape_and_texture",
                             specifying which component of the part feature to retrieve
        '''
        if mode == 'shape_only':
            return self.feature[partID * (self.texture_feat_num + self.shape_feat_num) + self.texture_feat_num: (partID + 1) * (self.texture_feat_num + self.shape_feat_num)]
        elif mode == 'texture_only':
            return self.feature[partID * (self.texture_feat_num + self.shape_feat_num): partID * (self.texture_feat_num + self.shape_feat_num) + self.texture_feat_num]
        elif mode == 'shape_and_texture':
            return self.feature[partID * (self.texture_feat_num + self.shape_feat_num): (partID + 1) * (self.texture_feat_num + self.shape_feat_num)]
        else:
            raise NotImplementedError

    def overwrite_feature(self, target_feature, partID, mode):
        '''Overwrites shape and/or texture feature in partID with target_feature
           Args: target_feature (numpy array), overwrite the feature with values in target_feature
                 partID (int), which part of feature to be overwritten
                 mode (str), one of "shape_only", "texture_only", "shape_and_texture",
                             specifying which component of the part feature to be overwritten
        '''
        if mode == 'shape_only':
            self.feature[partID * (self.texture_feat_num + self.shape_feat_num) + self.texture_feat_num: \
                         (partID + 1) * (self.texture_feat_num + self.shape_feat_num)] = target_feature
        elif mode == 'texture_only':
            self.feature[partID * (self.texture_feat_num + self.shape_feat_num): \
                         partID * (self.texture_feat_num + self.shape_feat_num) + self.texture_feat_num] = target_feature
        elif mode == 'shape_and_texture':
            self.feature[partID * (self.texture_feat_num + self.shape_feat_num): \
                         (partID + 1) * (self.texture_feat_num + self.shape_feat_num)] = target_feature
        else:
            raise NotImplementedError

def set_dataset_parameters(classname):
    '''Create the mapping between segmentation labels and the concatenated feature's ordering
       Args: classname (str), the dataset name that defines the segmentation taxanomy
       Return: part_type_dict (python dictionary), mapping from ordering in feature to segmentation label
               type_part_dict (python dictionary), mapping from segmentation label to ordering in feature
               PART_NUM (int), number of distinct parts in the concatenated feature
    '''
    if classname == 'humanparsing':
        with open('datasets/%s/garment_label_part_map.json' % classname, 'r') as readfile:
            garment_map = json.load(readfile)
        # top 4: 0
        # skirt 5: 1
        # pants 6: 2
        # dress 7: 3
        # background 0: 4
        PART_NUM = 0
        part_type_dict = dict()
        type_part_dict = dict()
        for garment_name in garment_map:
            if garment_name != 'background':
                PART_NUM += 1
            partID = garment_map[garment_name]['partID']
            type = garment_map[garment_name]['label']
            part_type_dict[type] = partID
            type_part_dict[partID] = type
        return part_type_dict, type_part_dict, PART_NUM
    else:
        raise NotImplementedError

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)
    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)
    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def setID(piece):
    ''' Parse the oufitID for the piece
        Args: piece (str), in the format of <outfitID>_<typeID>
        Return: outfitID (str)
    '''
    return '_'.join(piece.split('_')[:-1])

def get_mask(feature):
    ''' Create a masking array that exclude the absent parts in the outfit
        Args: feature (numpy array)
        Return: mask (numpy array)
    '''
    mask = np.zeros((len(feature),))
    # Missing parts have default value == 0
    mask[feature!=0] = 1
    return mask

def get_composing_pieces():
    ''' Get the composing pieces in an outfit
        Return: a list of pieceID, in the format of <outfitID>_<typeID>
    '''
    if fname in outfit_dict:
        return outfit_dict[fname]
    else:
        return None

#######################################
############ Optimization #############
#######################################

# update feature
#        |
#         ---- only swapped part
#               |
#                ----- only shape
#                ----- only texture
#                ----- shape and texture
#        |
#         ----- full
# stop criterion
#        |
#         ------ delta
#         ------ threshold
#         ------ max_iter
# whether or not to add smooth term

def decide_which_part(opt_feat, feature):
    ''' Use the maximal gradient direction to decide which part to update
        Args: opt_feat (pytorch tensor), optimized feature from the classifier
              feature (numpy array), original input feature
        Return: largest_magnitude_i (int), partID with largest gradient magnitude
    '''
    largest_magnitude = 0
    largest_magnitude_i = -1
    for i in range(PART_NUM):
        original_feature_i = feature[i * (TEXTURE_FEAT_NUM + SHAPE_FEAT_NUM): (i+1) * (TEXTURE_FEAT_NUM + SHAPE_FEAT_NUM)]
        updated_feature_i = opt_feat.data.cpu().numpy()[i * (TEXTURE_FEAT_NUM + SHAPE_FEAT_NUM): (i+1) * (TEXTURE_FEAT_NUM + SHAPE_FEAT_NUM)]
        magnitude_i = np.sum(np.abs(updated_feature_i-original_feature_i))
        print('part %d magnitude %f' %(i, magnitude_i))
        if magnitude_i > largest_magnitude:
            largest_magnitude_i = i
            largest_magnitude = magnitude_i
    return largest_magnitude_i

def manually_update_feature(opt_feat, input_feature, swapped_partID):
    ''' Overwrites the original input feature with the opitmized feature
        Args: opt_feat (pytorch tensor), feature optimized by the classifier;
                                         this is the target to overwrite with
              input_feature (InputFeature object), contains the original feature
                                                   this is the source that will be overwritten
              swapped_partID (int), which part to swap out in the input_feature
        Return: opt_feat (pytorch tensor), overwritten feature that keeps the original feature for all other parts,
                                           except for the swapped_partID that is substituted by target
    '''
    if argopt.update_full:
        # Update the entire outfit
        return opt_feat
    else:
        # Only update a specific part in the output, specified by swapped_partID
        opt_feature = InputFeature(SHAPE_FEAT_NUM, TEXTURE_FEAT_NUM, PART_NUM, opt_feat.data.cpu().numpy())
        input_feature.overwrite_feature(opt_feature.get_feature(swapped_partID, mode=argopt.update_type), \
                                        swapped_partID , mode=argopt.update_type)
        opt_feat = torch.from_numpy(input_feature.feature).cuda().float()
        opt_feat.requires_grad = True
        return opt_feat

def compute_updated_feature(input_feature):
    ''' Use the pre-trained classifier as editing module,
        and optimize the input_feature towards the positive direction.
        Args: input_feature (InputFeature object), the outfit's original feature before editing
        Return: updated_feature (list of InputFeature object), in the order of iterative updates
    '''
    global swapped_partID
    global swapped_type
    auto_swapped_partID = -1
    auto_swapped_type = -1
    loss = None
    updated_feature = []
    n_iter=0
    # Prepare mask, original_feat for computing smooth term; opt_feat for actual update
    mask = torch.from_numpy(get_mask(input_feature.feature)).cuda().byte()
    opt_feat = torch.from_numpy(input_feature.feature).cuda().float()
    opt_feat.requires_grad = True
    original_feat = copy.deepcopy(opt_feat)
    original_feat.requires_grad = False
    # Note: feature--numpy feature to keep parts unchanged;
    #       original_feat--tensor to compute smooth term
    #       opt_feat--tensor we update
    # Prepare positive label == 1 to be the direction to update
    label = torch.tensor([1]).cuda()
    # Prepare optimizer that optimizes the features
    optimizer = optim.SGD([opt_feat], lr = argopt.lr, momentum=0.9)
    # Prepare for stop criterion
    initial_loss = None
    meet_stop_criterion = StopCriterion(argopt.stop_criterion).delegate_child()
    while (n_iter <= argopt.max_iter_hr):
        if loss is not None: # Update is computed
            if swapped_partID >= 0: # Which part to swap out is either specified or computed
                #### Manual update ####
                opt_feat = manually_update_feature(opt_feat, input_feature, swapped_partID)
            else: # Has not specified or computed which part to swap out
                #### Auto decide which part to update ####
                auto_swapped_partID = decide_which_part(opt_feat, input_feature.feature)
                swapped_partID = auto_swapped_partID
                swapped_type = type_part_dict[swapped_partID]
                print('automatically decide part %d to swap out' % swapped_partID)
                #### Manual update ####
                opt_feat = manually_update_feature(opt_feat, input_feature, swapped_partID)
                # print(opt_feat)
        optimizer = optim.SGD([opt_feat], lr = argopt.lr, momentum=0.9)
        optimizer.zero_grad()
        ###### Forward ######
        output = model(opt_feat.unsqueeze(0)) # Add additional dimension for batchsize
        classify_loss = classify_criterion(output, label)
        smooth_loss = 0.0
        if argopt.lambda_smooth > 0:
            smooth_loss = smooth_criterion(opt_feat[mask], original_feat[mask]) # MSELoss(input, target), target should have requires_grad=False
        loss = classify_loss +  argopt.lambda_smooth * smooth_loss
        if (argopt.stop_criterion == 'deltaloss') and (initial_loss is None):
            initial_loss  = loss.data.cpu().numpy()
            meet_stop_criterion.set_initial_loss(initial_loss)

        if meet_stop_criterion(loss.data.cpu().numpy()):
            break
        ##### Backward #####
        loss.backward()
        if n_iter % argopt.display_freq == (argopt.display_freq-1):
            print('Iteration: %d, loss: %f'%(n_iter+1, loss.data[0]))
            # toTensor covert float64 to float32, while float32 cannot be converted to tensor again
            updated_feature.append(InputFeature(SHAPE_FEAT_NUM, TEXTURE_FEAT_NUM, \
                                              PART_NUM, np.float64(opt_feat.data.cpu().numpy())))
        n_iter += 1
        optimizer.step()
    return updated_feature


#######################################
############## Generation #############
#######################################

# generate original images
# generate update image
#        |
#         ---- swap
#               |
#                ----- iteration
#                ----- final
#        |
#         ----- full
#               |
#                ----- iteration
#                ----- final


def concatenate_to_shape_feature():
    '''Prepare shape feature in the format for VAE: numpy array (1, part_num * shape_feat_dum)
    '''
    feature = np.zeros((1, len(part_type_dict) * SHAPE_FEAT_NUM))
    for typeID in part_type_dict:
        pieceID = '_'.join([orig_outfitID, str(typeID)])
        feature[0, part_type_dict[typeID] * SHAPE_FEAT_NUM: (part_type_dict[typeID]+1) * SHAPE_FEAT_NUM] = piece_shape_feat_dict[pieceID]
    return feature

def construct_texture_feature():
    '''Prepare texture feature in the format for cGAN: python dict: {pieceID (str): numpy array}
    '''
    outfit_feat_dict = {}
    for old_piece in composing_pieces:
        if old_piece in piece_texture_feat_dict:
            outfit_feat_dict[int(old_piece.split('_')[-1])] = piece_texture_feat_dict[old_piece]
    return outfit_feat_dict

def generate_segmentation_mask_per_iteration(shape_feature, updated_feature):
    ''' Generate updated segmentation mask from updated feature
        Args: shape_feature (numpy array), original feature in the format for VAE to decode
              updated_feature (list of InputFeature objects), classifier input feature after updating
    '''
    # Add generator path
    parentPath = os.path.abspath("../../../separate_vae")
    if parentPath not in sys.path:
        sys.path.insert(1, parentPath)
    # print(sys.path)
    from decode_masks import batch_generation_from_update

    # Initilize first element in list
    batch_shape_feature = copy.deepcopy(shape_feature)
    feature = updated_feature[0]
    batch_shape_feature[0, swapped_partID * SHAPE_FEAT_NUM: (swapped_partID+1) * SHAPE_FEAT_NUM] = \
                            np.expand_dims(feature.get_feature(swapped_partID, mode='shape_only'), axis=0)
    filenames = [('_'.join(['%03d' % (argopt.display_freq), fname]))]
    for iter in range(1, len(updated_feature)):
        iter_shape_feature = copy.deepcopy(shape_feature)
        feature = updated_feature[iter]
        iter_shape_feature[0, swapped_partID * SHAPE_FEAT_NUM: (swapped_partID+1) * SHAPE_FEAT_NUM] = \
                            np.expand_dims(feature.get_feature(swapped_partID, mode='shape_only'), axis=0)
        batch_shape_feature = np.vstack((batch_shape_feature, iter_shape_feature))
        filenames.append('_'.join(['%03d' % ((iter + 1) * argopt.display_freq), fname]))
    with cd('../../../separate_vae'):
        batch_generation_from_update(4, argopt.save_dir, filenames, batch_shape_feature, argopt.load_pretrain_shape_gen, argopt.classname)#, black=False)
    # Remove generator path
    if parentPath in sys.path:
        sys.path.remove(parentPath)
    # print(sys.path)

def generate_segmentation_mask_final(shape_feature, updated_feature):
    ''' Generate updated segmentation mask from updated feature
        Args: shape_feature (numpy array), original feature in the format for VAE to decode
              updated_feature (list of InputFeature objects), classifier input feature after updating
    '''
    # Add generator path
    parentPath = os.path.abspath("../../../separate_vae")
    if parentPath not in sys.path:
        sys.path.insert(1, parentPath)
    from decode_masks import single_generation_from_update
    # Generate fianl update
    batch_shape_feature = copy.deepcopy(shape_feature)
    feature = updated_feature[-1] # Last feature is the final update
    batch_shape_feature[0, swapped_partID * SHAPE_FEAT_NUM: (swapped_partID+1) * SHAPE_FEAT_NUM] = \
                           np.expand_dims(feature.get_feature(swapped_partID, mode='shape_only'), axis=0)
    filename = '_'.join(['final', fname])
    with cd('../../../separate_vae'):
        single_generation_from_update(argopt.save_dir, filename, batch_shape_feature, argopt.load_pretrain_shape_gen, argopt.classname)#, black=False)
    # Generate reconstructed
    batch_shape_feature = copy.deepcopy(shape_feature) # Use original feature for reconstruction
    filename = '_'.join(['001', fname])
    with cd('../../../separate_vae'):
        single_generation_from_update(argopt.save_dir, filename, batch_shape_feature, argopt.load_pretrain_shape_gen, argopt.classname)#, black=False)

    # Remove generator path
    if parentPath in sys.path:
        sys.path.remove(parentPath)
    # print(sys.path)

def generate_texture_on_mask_per_iteration(outfit_feat_dict, updated_feature):
    ''' Generate updated texture on updated mask from updated feature
        Args: outfit_feat_dict (python dict), original texture feature in the format for cGAN to decode
              updated_feature (list of InputFeature objects), classifier input feature after updating
    '''
    parentPath = os.path.abspath("../../../generation")
    if parentPath not in sys.path:
        sys.path.insert(1, parentPath)
    from decode_clothing_features import generation_from_decoded_mask

    for iter in range(len(updated_feature)):
        iter_outfit_feat_dict = copy.deepcopy(outfit_feat_dict)
        feature = updated_feature[iter]
        if argopt.update_full:
            raise NotImplementedError
        else:
            iter_outfit_feat_dict[swapped_type] = np.expand_dims(feature.get_feature(swapped_partID, mode='texture_only'), axis=0)
        with cd('../../../generation'): # WRITE A BATCH VERSION LATER
            generation_from_decoded_mask('%03d' % ((iter + 1) * argopt.display_freq), argopt.save_dir, fname, iter_outfit_feat_dict, \
                                        argopt.load_pretrain_texture_gen, argopt.color_mode, argopt.netG, argopt.model_type, argopt.classname, argopt.texture_feat_num, original_mask=False, update=True, from_avg=True, remove_background=True)
    # Remove generator path
    if parentPath in sys.path:
        sys.path.remove(parentPath)
    # print(sys.path)

def generate_texture_on_mask_final(outfit_feat_dict, updated_feature):
    ''' Generate updated texture on updated mask from updated feature
        Args: outfit_feat_dict (python dict), original texture feature in the format for cGAN to decode
              updated_feature (list of InputFeature objects), classifier input feature after updating
    '''
    parentPath = os.path.abspath("../../../generation")
    if parentPath not in sys.path:
        sys.path.insert(1, parentPath)
    from decode_clothing_features import generation_from_decoded_mask

    # Generation for final update
    iter_outfit_feat_dict = copy.deepcopy(outfit_feat_dict)
    feature = updated_feature[-1] # Last feature is the final update
    if argopt.update_full:
        raise NotImplementedError
    else:
        iter_outfit_feat_dict[swapped_type] = np.expand_dims(feature.get_feature(swapped_partID, mode='texture_only'), axis=0)
    with cd('../../../generation'):
        generation_from_decoded_mask('final', argopt.save_dir, fname, iter_outfit_feat_dict, \
                                    argopt.load_pretrain_texture_gen, argopt.color_mode, argopt.netG, argopt.model_type, argopt.classname, argopt.texture_feat_num, original_mask=False, update=True, from_avg=True, remove_background=True)
    # Reconstruct original image before updating
    iter_outfit_feat_dict = copy.deepcopy(outfit_feat_dict)
    with cd('../../../generation'):
        generation_from_decoded_mask('001', argopt.save_dir, fname, iter_outfit_feat_dict, \
                                    argopt.load_pretrain_texture_gen, argopt.color_mode, argopt.netG, argopt.model_type, argopt.classname, argopt.texture_feat_num, original_mask=False, update=True, from_avg=True, remove_background=True)
    # Remove generator path
    if parentPath in sys.path:
        sys.path.remove(parentPath)
    # print(sys.path)

def generate_swapped_outfit_iteration(updated_feature, composing_pieces):
    shape_feature = concatenate_to_shape_feature()
    outfit_feat_dict = construct_texture_feature()
    ############# SHAPE ###########
    generate_segmentation_mask_per_iteration(shape_feature, updated_feature)
    ############# TEXTURE ###########
    generate_texture_on_mask_per_iteration(outfit_feat_dict, updated_feature)

def generate_swapped_final_update(updated_feature, composing_pieces):
    shape_feature = concatenate_to_shape_feature()
    outfit_feat_dict = construct_texture_feature()
    ############# SHAPE ###########
    generate_segmentation_mask_final(shape_feature, updated_feature)
    ############# TEXTURE ###########
    generate_texture_on_mask_final(outfit_feat_dict, updated_feature)

########################################
############## Writing out #############
########################################
def addin_shape_codes_per_iteration(shape_feature, updated_feature):
    ''' Add the updated shape feature into dictionary
        Args: shape_feature (numpy array), original shape feature in the format for VAE to decode
              updated_feature (list of InputFeature objects), classifier input feature after updating
    '''
    for iter in range(len(updated_feature)):
        iter_shape_feature = copy.deepcopy(shape_feature)
        feature = updated_feature[iter]
        if argopt.update_full:
            for partID in range(PART_NUM):
                iter_shape_feature[0, partID * SHAPE_FEAT_NUM: (partID+1) * SHAPE_FEAT_NUM] = \
                                   np.expand_dims(feature.get_feature(partID, mode='shape_only'), axis=0)

        else:
            iter_shape_feature[0, swapped_partID * SHAPE_FEAT_NUM: (swapped_partID+1) * SHAPE_FEAT_NUM] = \
                               np.expand_dims(feature.get_feature(swapped_partID, mode='shape_only'), axis=0)
        # print('iter_shape_feature addin', iter_shape_feature)
        shape_codes_dict[('_'.join(['%03d' % ((iter + 1) * argopt.display_freq), fname]))] = iter_shape_feature

def addin_texture_codes_per_iteration(outfit_feat_dict, updated_feature):
    ''' Add the updated texture feature into dictionary
        Args: outfit_feat_dict (python dict), original texture feature in the format for cGAN to decode
              updated_feature (list of InputFeature objects), classifier input feature after updating
    '''
    for iter in range(len(updated_feature)):
        iter_outfit_feat_dict = copy.deepcopy(outfit_feat_dict)
        feature = updated_feature[iter]
        if argopt.update_full:
            for typeID in part_type_dict:
                if (typeID in iter_outfit_feat_dict) and (typeID != 0):
                    iter_outfit_feat_dict[typeID] = np.expand_dims(feature.get_feature(part_type_dict[typeID], mode='texture_only'), axis=0)
        else:
            iter_outfit_feat_dict[swapped_type] = np.expand_dims(feature.get_feature(swapped_partID, mode='texture_only'), axis=0)
        texture_codes_dict[('_'.join(['%03d' % ((iter + 1) * argopt.display_freq), fname]))] = iter_outfit_feat_dict

def addin_shape_codes_final(shape_feature, updated_feature):
    ''' Add the updated shape feature into dictionary
        Args: shape_feature (numpy array), original feature in the format for VAE to decode
              updated_feature (list of InputFeature objects), classifier input feature after updating
    '''
    # Final update
    batch_shape_feature = copy.deepcopy(shape_feature)
    feature = updated_feature[-1] # Last feature for final update
    batch_shape_feature[0, swapped_partID * SHAPE_FEAT_NUM: (swapped_partID+1) * SHAPE_FEAT_NUM] = \
                           np.expand_dims(feature.get_feature(swapped_partID, mode='shape_only'), axis=0)
    shape_codes_dict[('_'.join(['final', fname]))] = batch_shape_feature
    # Reconsturct original
    batch_shape_feature = copy.deepcopy(shape_feature)
    shape_codes_dict[('_'.join(['001', fname]))] = batch_shape_feature


def addin_texture_codes_final(outfit_feat_dict, updated_feature):
    ''' Add the updated texture feature into dictionary
        Args: outfit_feat_dict (python dict), original texture feature in the format for cGAN to decode
              updated_feature (list of InputFeature objects), classifier input feature after updating
    '''
    # Final update
    iter_outfit_feat_dict = copy.deepcopy(outfit_feat_dict)
    feature = updated_feature[-1] # Last feature for final update
    iter_outfit_feat_dict[swapped_type] = np.expand_dims(feature.get_feature(swapped_partID, mode='texture_only'), axis=0)
    texture_codes_dict[('_'.join(['final', fname]))] = iter_outfit_feat_dict
    # Reconstruct original
    iter_outfit_feat_dict = copy.deepcopy(outfit_feat_dict)
    texture_codes_dict[('_'.join(['001', fname]))] = iter_outfit_feat_dict


def addin_swapped_final_outfit(updated_feature):
    shape_feature = concatenate_to_shape_feature()
    outfit_feat_dict = construct_texture_feature()
    ############# SHAPE ###########
    addin_shape_codes_final(shape_feature, updated_feature)
    ############# TEXTURE ###########
    addin_texture_codes_final(outfit_feat_dict, updated_feature)

def addin_swapped_outfit_iteration(updated_feature):
    shape_feature = concatenate_to_shape_feature()
    outfit_feat_dict = construct_texture_feature()
    ############# SHAPE ###########
    addin_shape_codes_per_iteration(shape_feature, updated_feature)
    ############# TEXTURE ###########
    addin_texture_codes_per_iteration(outfit_feat_dict, updated_feature)


argopt = option_parser()
SHAPE_FEAT_NUM = argopt.shape_feat_num
TEXTURE_FEAT_NUM = argopt.texture_feat_num
############### Read in original features, dictionary of pieces in outfit ################
part_type_dict, type_part_dict, PART_NUM = set_dataset_parameters(argopt.classname)
with open(os.path.join(argopt.dataset_dir, 'demo_dict.json'), 'r') as readfile:
        outfit_dict = json.load(readfile)
with open(argopt.texture_feat_file, 'rb') as readfile:
    piece_texture_feat_dict = pickle.load(readfile)
with open(argopt.shape_feat_file, 'rb') as readfile:
    piece_shape_feat_dict = pickle.load(readfile)

############## Read in trained classifier ##################
# Add classifier path
parentPath = os.path.abspath("../../")
if parentPath not in sys.path:
    sys.path.insert(0, parentPath)
from model import load_network
from search_networks import MLP, LinearClassifier
# Create model
if argopt.network_arch == 'mlp':
    model = MLP(argopt).cuda()
else:
    raise NotImplementedError
print(model)
resume_iteration = 0
if argopt.load_pretrain_clf:
    load_network(argopt, model, argopt.clf_epoch, save_dir=argopt.load_pretrain_clf)
for param in model.parameters(): # DON't backprop gradients to classifier
    param.requires_grad = False
# Remove classifier path
if parentPath in sys.path:
    sys.path.remove(parentPath)
#################### Initialize optimizer ######################
classify_criterion = nn.CrossEntropyLoss().cuda()
smooth_criterion = nn.MSELoss().cuda()
################### Start updating and visualizing ###############
if argopt.generate_or_save == 'save':
    shape_codes_dict = dict()
    texture_codes_dict = dict()

# 1) Compose encodings into classifier input
input_feature = InputFeature(SHAPE_FEAT_NUM, TEXTURE_FEAT_NUM, PART_NUM)
orig_outfitID = argopt.update_fname[:-4] # strip away file extension
fname = orig_outfitID
for piece in outfit_dict[fname]:
    # Ignore pieces that are not clothing types
    if (int(piece.split('_')[-1]) in part_type_dict) and (piece.split('_')[-1] != '0'):
        part_idx = part_type_dict[int(piece.split('_')[-1])]
        if piece in piece_texture_feat_dict:
            input_feature.overwrite_feature(np.hstack((piece_texture_feat_dict[piece].ravel(), piece_shape_feat_dict[piece].ravel())), \
                                      part_idx, mode='shape_and_texture')
        else: # Sometimes no texture feature for that piece...
            input_feature.overwrite_feature(piece_shape_feat_dict[piece].ravel(), \
                                      part_idx, mode='shape_only')
# 2) Start updating
swapped_partID = -1
swapped_type = -1
composing_pieces = get_composing_pieces()
if not argopt.autoswap:
    swapped_partID = argopt.swapped_partID
    swapped_type = part_type_dict[swapped_partID]
updated_feature = compute_updated_feature(input_feature)
# 3) Generate or save
if argopt.generate_or_save == 'generate':
    if argopt.iterative_generation:
        generate_swapped_outfit_iteration(updated_feature, composing_pieces)
    else:
        # render the features on the image
        generate_swapped_final_update(updated_feature, composing_pieces)
elif argopt.generate_or_save == 'save':
    if argopt.iterative_generation:
        addin_swapped_outfit_iteration(updated_feature)
    else:
        addin_swapped_final_outfit(updated_feature)

if argopt.generate_or_save == 'save':
    if not os.path.exists(argopt.save_dir):
        os.makedirs(argopt.save_dir)
    with open(os.path.join(argopt.save_dir, 'shape_codes_dict.p'), 'wb') as writefile: # save in shape results
        pickle.dump(shape_codes_dict, writefile)
    with open(os.path.join(argopt.save_dir, 'texture_codes_dict.p'), 'wb') as writefile: # save in texture results
        pickle.dump(texture_codes_dict, writefile)
