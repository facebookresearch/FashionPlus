# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import sys
import pdb
import pickle
import argparse
import numpy as np
from PIL import Image
import copy
import math

def polygon_bbox(y_index, x_index):
    ''' Get the bounding box that contains the enitre polygon
        Args: y_index, x_index: index positions with value 1
        Return: bounding box, uppermost y position,
                              leftmost x position,
                              bottommost y position,
                              rightmost x position
    '''
    return np.min(y_index), np.min(x_index), np.max(y_index), np.max(x_index)

def get_bbox(fname):
    ''' Get the precomputed (person) bbox position for the image
        Args: fname, filename of the image
        Return: bounding box, uppermost y position,
                              leftmost x position,
                              bottommost y position,
                              rightmost x position
    '''
    for idx in dataset:
        if dataset[idx]['seg'] == fname:
            return dataset[idx]['bbox']
    return None

def crop_and_resize(img, bbox, mode='RGB'):
    ''' Crop out the human bounding box,
        and resize the cropped image to height=256
        Args: img (PIL image), original imge
              bbox, human bounding box in img
              mode (str), color space
        Return: new_img, cropped and resized image
    '''
    img = crop_person(img, bbox)
    # Pad and resize to 256 x 256
    # 1) Resize h to 256
    w, h = img.size
    new_h = 256
    new_w = int((new_h/float(h)) * w)
    img = img.resize((new_w, new_h))
    # 2) Create a new image (256x256)  and paste the resized on it
    new_img = Image.new(mode, (256, 256))
    new_img.paste(img, ((256-new_w)//2, 0))
    return new_img

def crop_person(img, bbox):
    ''' Crop out the person bounding box in the image
        Args: img (PIL image), entire image
              bbox, person bounding box positions, uppermost y position,
                                                   leftmost x position,
                                                   bottommost y position,
                                                   rightmost x position
        Return: the cropped image tightly containing the person in it
    '''
    # Image crop
    return img.crop((int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))) # left, upper, right, lower

def scale_to_same_width(image, new_image):
    ''' Scale image to the same width as new_image
        Args: image (PIL image), source image
              new_image (PIL image), target image
        Return: scaled image with the same width as new_image
    '''
    w, h = image.size
    new_w, _ = new_image.size
    ratio = new_w/float(w)
    new_h = int(h * ratio)
    return image.resize((new_w, new_h)), ratio

def scale_to_same_height(image, new_image):
    ''' Scale image to the same height as new_image
        Args: image (PIL image), source image
              new_image (PIL image), target image
        Return: scaled image with the same height as new_image
    '''
    w, h = image.size
    _, new_h = new_image.size
    ratio = new_h/float(h)
    new_w = int(w * ratio)
    return image.resize((new_w, new_h)), ratio


def option_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--fname', type=str, help='filename to postprocess')
    parser.add_argument('--bbox_pickle_file', type=str, help='pickle file that contains human bounding box')
    parser.add_argument('--orig_img_dir', type=str, help='directory path with original RGB images in it')
    parser.add_argument('--orig_mask_dir', type=str, help='directory path with original segmentation masks in it')
    parser.add_argument('--gen_img_dir', type=str, help='directory path with generated RGB images in it')
    parser.add_argument('--gen_mask_dir', type=str, help='directory path with generated segmentation masks in it')
    parser.add_argument('--result_dir', type=str, help='directory to write the final post-processed images to')
    return parser.parse_args()

opt = option_parser()
fname = opt.fname

# 1) Read in person bbox
with open(opt.bbox_pickle_file, 'rb') as readfile:
    dataset = pickle.load(readfile)

# 2) Read in original image and mask
# 4) Crop human bbox and resize original image and mask
# Strip iteration header away
original_fname = fname.split('_')[-1]
image = Image.open(os.path.join(opt.orig_img_dir, original_fname))

bbox = get_bbox(original_fname[:-4]+'.png') # replace jpg with png
assert(bbox is not None), 'Cannot find file %s in dictionary' % fname
image = crop_and_resize(image, bbox, 'RGB')

mask = Image.open(os.path.join(opt.orig_mask_dir, original_fname[:-4]+'.png')) # replace jpg with png
mask = crop_and_resize(mask, bbox, 'L')

# 3) Read in generated image and mask
image_prime = Image.open(os.path.join(opt.gen_img_dir, fname))
mask_prime = Image.open(os.path.join(opt.gen_mask_dir, (fname[:-4]+'.png')))

# 5) Get the face+hair bbox on original and generated masks
# face 11
# hair 2
# hat 1
# glasses 3
# 5-1) For original
head_mask = np.logical_or(np.array(mask)==11, np.array(mask)==2)
head_mask = np.logical_or(head_mask, np.array(mask)==1)
head_mask = np.logical_or(head_mask, np.array(mask)==3)
y_i, x_i = np.nonzero(head_mask)
y1, x1, y2, x2 = polygon_bbox(y_i, x_i)
head_image = image.crop((x1, y1, x2, y2)) # (left, upper, right, lower)-tuple
head_bbox_w, head_bbox_h = head_image.size
# 5-2) For generated
head_mask_prime = np.logical_or(np.array(mask_prime)==11, np.array(mask_prime)==2)
head_mask_prime  = np.logical_or(head_mask_prime , np.array(mask_prime)==1)
head_mask_prime  = np.logical_or(head_mask_prime , np.array(mask_prime)==3)
y_i, x_i = np.nonzero(head_mask_prime)
y1_prime, x1_prime, y2_prime, x2_prime = polygon_bbox(y_i, x_i)
head_image_prime = image_prime.crop((x1_prime, y1_prime, x2_prime, y2_prime))
head_bbox_w_prime, head_bbox_h_prime = head_image_prime.size

############ Overlap bbox #############
min_head_bbox_w = np.min(np.array([head_bbox_w, head_bbox_w_prime]))
min_head_bbox_h = np.min(np.array([head_bbox_h, head_bbox_h_prime]))
# Align original and generated head bounding boxes to center
overlap_y1 = int((y1+y2)/float(2) - min_head_bbox_h/float(2))
overlap_y2 = int((y1+y2)/float(2) + min_head_bbox_h/float(2))
overlap_x1 = int((x1+x2)/float(2) - min_head_bbox_w/float(2))
overlap_x2 = int((x1+x2)/float(2) + min_head_bbox_w/float(2))
overlap_y1_prime = int((y1_prime+y2_prime)/float(2) - min_head_bbox_h/float(2))
overlap_y2_prime = int((y1_prime+y2_prime)/float(2) + min_head_bbox_h/float(2))
overlap_x1_prime = int((x1_prime+x2_prime)/float(2) - min_head_bbox_w/float(2))
overlap_x2_prime = int((x1_prime+x2_prime)/float(2) + min_head_bbox_w/float(2))
print('original w: %f h: %f' % ((overlap_x2-overlap_x1), (overlap_y2-overlap_y1)))
print('generated w: %f h: %f' % ((overlap_x2_prime-overlap_x1_prime), (overlap_y2_prime-overlap_y1_prime)))
# Compute ratio of overlap area to original area
original_area = head_bbox_w * head_bbox_h
overlap_area = min_head_bbox_w * min_head_bbox_h
overlap_ratio = overlap_area/float(original_area)
print('Overlap ratio: %f' % overlap_ratio)

# Compute center position
original_center_pos_x, original_center_pos_y = (x1+x2)/float(2), (y1+y2)/float(2)
prime_center_pos_x, prime_center_pos_y =  (x1_prime+x2_prime)/float(2), (y1_prime+y2_prime)/float(2)
print('Original center position x: %f y: %f' % (original_center_pos_x, original_center_pos_y))
print('Generated center position x: %f y: %f' % (prime_center_pos_x, prime_center_pos_y))


if overlap_ratio >= 0.6:
    if (abs(original_center_pos_x - prime_center_pos_x) <=10) and (abs(original_center_pos_y - prime_center_pos_y) <=10):
        print('Simple overwrites')
        # Version I:
        overlap_mask = np.zeros_like(np.array(mask), dtype=bool)
        # print(overlap_mask.shape)
        overlap_mask[overlap_y1_prime: overlap_y2_prime, overlap_x1_prime: overlap_x2_prime] = 1
        overlap_head_mask = np.logical_and(overlap_mask, head_mask)
        image_prime = np.array(image_prime)
        image_prime[overlap_head_mask] = np.array(image)[overlap_head_mask]
        # Image.fromarray(np.uint8(image_prime)).save('result/%s' % fname)

    else:
        # Version III
        overlap_mask = np.zeros_like(np.array(mask), dtype=bool)
        overlap_mask[overlap_y1: overlap_y2, overlap_x1: overlap_x2] = 1
        overlap_head_mask = np.logical_and(overlap_mask, head_mask)
        image_prime = np.array(image_prime)
        image_prime[overlap_head_mask] = np.array(image)[overlap_head_mask]


else:
    print('Complete overwrites')
    ############ Original image completely overwrites #############
    # Version I
    overwrite_mask = np.zeros_like(np.array(mask), dtype=bool)
    overwrite_mask[y1: y2, x1: x2] = 1
    overwrite_head_mask = np.logical_and(overwrite_mask, head_mask)
    image_prime = np.array(image_prime)
    image_prime[overwrite_head_mask] = np.array(image)[overwrite_head_mask]

if fname.startswith('001'):
    new_fname = 'reconstructed_' + original_fname
    Image.fromarray(np.uint8(image_prime)).save(os.path.join(opt.result_dir, new_fname))
    print('Saving results to %s' % os.path.join(opt.result_dir, new_fname))
else:
    Image.fromarray(np.uint8(image_prime)).save(os.path.join(opt.result_dir, fname))
    print('Saving results to %s' % os.path.join(opt.result_dir, fname))
