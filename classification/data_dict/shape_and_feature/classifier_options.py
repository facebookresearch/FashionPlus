# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse

def option_parser():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--texture_feat_file', type=str, help='pickle file with texture features of pieces')
    parser.add_argument('--shape_feat_file', type=str, help='pickle file with shape features of pieces')
    parser.add_argument('--dataset_dir', type=str, help='directory path to read and write files')
    parser.add_argument('--load_pretrain_clf', type=str, default='', help='load the pretrained classification model from the specified location')
    parser.add_argument('--update_fname', type=str, help='the filename of the imagee we are updating')
    parser.add_argument('--clf_epoch', type=int, default = -1, help='load model at epoch; -1 for highest epoch')
    parser.add_argument('--save_dir', type=str, default='results/fashion/updates/', help='path to save generated image')

    # Network
    # 1) Classifier
    parser.add_argument('--network_arch', type=str, default='mlp', help='architecture of the network [mlp|linear]')
    parser.add_argument('--in_dim', type=int, default = 12, help='input dimension for first fc layer')
    parser.add_argument('--out_dim', type=int, default = 2, help='output dimension for first fc layer')
    parser.add_argument('--param_m', type=int, default = 1, help='number of hidden layers in MLP')
    parser.add_argument('--param_k', type=int, default = 8, help='number of neurons at each hidden layer')
    parser.add_argument('--fc1_dim', type=int, default = 8, help='dimension for fc layer')
    parser.add_argument('--fc2_dim', type=int, default = 2, help='dimension for fc layer')
    parser.add_argument('--use_dropout', action='store_true', help='if specified, use dropout layer')
    # 2) Generator
    #    2-1) Texture generator
    parser.add_argument('--load_pretrain_texture_gen', type=str, default='', help='load the pretrained generator model from the specified location')
    parser.add_argument('--color_mode', type=str, default='RGB', help='color mode of our color image [Lab|RGB]')
    parser.add_argument('--model_type', type=str, default='pix2pixHD', help='currently only suppport pix2pixHD')
    parser.add_argument('--texture_feat_num', type=int, default=3, help='texture generator feature dimension')
    #    2-2) Shape generator
    parser.add_argument('--load_pretrain_shape_gen', type=str, default='', help='load the pretrained generator model from the specified location')
    parser.add_argument('--shape_feat_num', type=int, default=8, help='shapee generator feature dimension')

    # Learning
    parser.add_argument('--stop_criterion', type=str, default = 'maxiter', help='stop scriterion for optimization process: maxiter | deltaloss | thresholdloss')
    parser.add_argument('--max_iter_hr', type=int, default = 15, help='how many iterations to run')
    parser.add_argument('--min_deltaloss', type=float, default = 0.0, help='the amount of change the loss should make before stop')
    parser.add_argument('--min_thresholdloss', type=float, default = 0.0, help='the thresholded loss that the optimizer needs to reach before stops')
    parser.add_argument('--lr', type=float, default = 0.05, help='optimizer learning rate; here is the step size for updating module')
    parser.add_argument('--lambda_smooth', type=float, default = 10, help='weight of the smooth term')

    # Output
    parser.add_argument('--netG', type=str, default='global', help='generator architecture [global|local]')
    parser.add_argument('--update_full', action='store_true', help='if specified, update the whole outfit instead of only the swapped')
    parser.add_argument('--update_type', type=str, default='shape_and_texture', help='when partially update: shape_only | texture_only | shape_and_texture')
    parser.add_argument('--display_freq', type=int, default = 5, help='how often to compute accuracy')
    parser.add_argument('--autoswap', action='store_true', help='if specified, automatically decide which part to swap out; should not be used together with swapped_partID')
    parser.add_argument('--generate_or_save', type=str, default = 'generate', help='generate updated image or save the updated vector: generate | save')
    parser.add_argument('--iterative_generation', action='store_true', help='if specified, generate each iteration of an image update')
    parser.add_argument('--classname', type=str, help='segmentation definition from dataset:  humanparsing')
    parser.add_argument('--swapped_partID', type=int, default = 0, help='predefine which part to swap to; has no effect when autoswap option is specified; for humanparsing classname, 0: top; 1: skirt; 2: pants; 3:dress')

    return parser.parse_args()
