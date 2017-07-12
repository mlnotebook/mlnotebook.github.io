import numpy as np
import tensorflow as tf
import scipy.misc

#IMAGE LOAD WRAPPER
""" Loads the image and crops it to 'image_size'

PARAMETERS
is_crop:    whether to crop the image or not [True or False]

INPUTS
image_path: location of the image
image_size: size (in pixels) of the output image

RETURNS
- the cropped image
"""
def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)



#IMAGE READER FUNCTION
""" Reads in the image (part of get_image function)

INPUT
path: location of the image
"""
def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)



#CREATE IMAGE ARRAY FUNCTION
""" Takes a set of 'images' and creates an array from them.

INPUT
images:     the set of input images
size:       [height, width] of the array

RETURNS
- image array as a single image
""" 
def merge(images, size):



# IMAGE CROPPING FUNCTION
""" Crops the input image at the centre pixel

INPUTS
x:      the input image
crop_h: the height of the crop region
crop_w: if None crop width = crop height
resize_w: the width of the resized image

RETURNS
- the cropped image
"""
def center_crop(x, crop_h, crop_w=None, resize_w=64):


# TRANSFORM/CROPPING WRAPPER
""" Transforms the image by cropping and resizing

INPUT
image:      the image to be transformed
npx:        the size of the transformed image [npx x npx]
is_crop:    whether to preform cropping too [True or False]

RETURNS
- the cropped, transformed image
"""
def transform(image, npx=64, is_crop=True):



