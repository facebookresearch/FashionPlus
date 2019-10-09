# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import pdb
import glob
import json
import pickle
import argparse
import numpy as np
from PIL import Image

def get_composing_pieces_in_outfit(mask_dir, filename):
    mask = Image.open(os.path.join(mask_dir, filename))
    np_mask = np.array(mask)
    unique_labels = np.unique(np_mask)
    return unique_labels

def get_bbox(img_path):
    image = Image.open(img_path)
    width, height = image.size
    # We assume images are all tightly cropped
    return np.array([0., 0., width, height])

def option_parser():
    parser = argparse.ArgumentParser()

    # Input
    parser.add_argument('--img_dir', type=str, help='directory path with RGB images in it')
    parser.add_argument('--mask_dir', type=str, help='directory path with segmentation masks in it')
    parser.add_argument('--output_pickle_file', type=str, help='pickle file path for generator networks')
    parser.add_argument('--output_json_file', type=str, help='json file path for classifier network')
    return parser.parse_args()

# Prepare pickle files for generators, in the format:
# {imageID:  { 'filename': <RGB_image_filename>,
#              'bbox': <numpy array for human bounding box, in format [leftmost_x, uppermost_y, rightmost_x, bottommost_y]>,
#              'label': 1 ,
#              'seg': <segmentation_mask filename>,
# }           }

# Prepare json files for classifier in the format:
# {filename (without extension): [<filename>_<piece>, ...]}

opt = option_parser()
list_of_images = sorted(glob.glob(opt.img_dir + '*.jpg'))


pickle_dict = dict()
outfit_composition_dict = dict()
for img_i, img_path in enumerate(list_of_images):
    img_filename = os.path.split(img_path)[-1]
    mask_filename = img_filename[:-4] + '.png' # replace jpg with png
    bbox = get_bbox(img_path)
    pickle_dict_template = {'filename': img_filename,
                           'bbox':      bbox,
                           'label':     1,
                           'seg':       mask_filename}
    pickle_dict[img_i] = pickle_dict_template
    unique_pieces = get_composing_pieces_in_outfit(opt.mask_dir, mask_filename)
    composing_pieceIDs = ['_'.join([str(img_filename[:-4]), str(piece)]) for piece in unique_pieces]
    outfit_composition_dict[str(img_filename[:-4])] = composing_pieceIDs


print('Saving pickle input file for generators at %s' % opt.output_pickle_file )
with open(opt.output_pickle_file, 'wb') as writefile:
    pickle.dump(pickle_dict, writefile)
print('Saving json input file for classifier at %s' % opt.output_json_file )
with open(opt.output_json_file, 'w') as writefile:
    json.dump(outfit_composition_dict, writefile)
