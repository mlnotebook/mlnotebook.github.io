import numpy as np
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
""" Transforms the image by cropping and resizing and 
normalises intensity values between -1 and 1

INPUT
image:      the image to be transformed
npx:        the size of the transformed image [npx x npx]
is_crop:    whether to preform cropping too [True or False]

RETURNS
- the cropped, normalised image
"""
def transform(image, npx=64, is_crop=True):





#CREATE IMAGE ARRAY FUNCTION
""" Takes a set of 'images' and creates an array from them.

INPUT
images:     the set of input images
size:       [height, width] of the array

RETURNS
- image array as a single image
""" 
def merge(images, size):


#ARRAY TO IMAGE FUNCTION
""" Takes a set of `images` and calls the merge function. Converts
the array to image data.

INPUT
images: the set of input images
size:   [height, width] of the array
path:   the save location

RETURNS
- an image array
"""
def imsave(images, size, path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))

#SAVE IMAGE FUNCTION
""" takes an image and saves it to disk. Redistributes
intensity values [-1 1] from [0 255]


"""
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

#INVERSE TRANSFORMATION OF INTENSITITES
""" This turns the intensities from [-1 1] to [0 1]

INPUTS
images:     the image to be transformed

RETURNS
-the transformed image
"""
def inverse_transform(images):
    return (images+1.)/2.
    
