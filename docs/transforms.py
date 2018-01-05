
###########################################################
# DATA AUGMENTATION FUNCTIONS FOR N-DIMENSIONAL IMAGES
# author: r.robinson16@imperial.ac.uk
# Part of the 'Data Augmentations for n-Dimensional Image
# Input to CNNs' tutorial at https://mlnotebook.github.io
###########################################################

"""
For each function:
image   = array of pixel intensities in 2 or 3 dimensions
returns = array of pixel intensities same shape as `image`
"""

from scipy.ndimage import rotate, interpolation
import numpy as np
import matplotlib.pyplot as plt

def scaleit(image, factor, isseg=False):
    order = 0 if isseg == True else 3

    height, width, depth= image.shape
    zheight             = int(np.round(factor * height))
    zwidth              = int(np.round(factor * width))
    zdepth              = depth

    if factor < 1.0:
        newimg  = np.zeros_like(image)
        row     = int((height - zheight) / 2)
        col     = int((width - zwidth) / 2)
        layer   = int((depth - zdepth) / 2)
        newimg[row:row+zheight, col:col+zwidth, layer:layer+zdepth] = interpolation.zoom(image, (float(factor), float(factor), 1.0), order=order, mode='nearest')[0:zheight, 0:zwidth, 0:zdepth]

        return newimg

    elif factor > 1.0:
        row     = (zheight - height) // 2
        col     = (zwidth - width) // 2
        layer   = (zdepth - depth) // 2

        newimg = interpolation.zoom(image[row:row+zheight, col:col+zwidth, layer:layer+zdepth], (float(factor), float(factor), 1.0), order=order, mode='nearest')  
        
        extrah = (newimg.shape[0] - height) // 2
        extraw = (newimg.shape[1] - width) // 2
        extrad = (newimg.shape[2] - depth) // 2
        newimg = newimg[extrah:extrah+height, extraw:extraw+width, extrad:extrad+depth]

        return newimg

    else:
        return image


def resampleit(image, dims, isseg=False):
    order = 0 if isseg == True else 5

    image = interpolation.zoom(image, np.array(dims)/np.array(image.shape, dtype=np.float32), order=order, mode='nearest')

    if image.shape[-1] == 3: #rgb image
        return image
    else:
        return image if isseg else (image-image.min())/(image.max()-image.min()) 
    

def translateit(image, offset, isseg=False):
    order = 0 if isseg == True else 5

    return interpolation.shift(image, (int(offset[0]), int(offset[1]), 0), order=order, mode='nearest')


def rotateit(image, theta, isseg=False):
    order = 0 if isseg == True else 5
        
    return rotate(image, float(theta), reshape=False, order=order, mode='nearest')


def flipit(image, axes):
    
    if axes[0]:
        image = np.fliplr(image)
    if axes[1]:
        image = np.flipud(image)
    
    return image


def intensifyit(image, factor):

    return image*float(factor)


def cropit(image, seg=None, margin=5):

    fixedaxes = np.argmin(image.shape[:2])
    trimaxes  = 0 if fixedaxes == 1 else 1
    trim    = image.shape[fixedaxes]
    center  = image.shape[trimaxes] // 2
    if seg is not None:

        hits = np.where(seg!=0)
        mins = np.argmin(hits, axis=1)
        maxs = np.argmax(hits, axis=1)

        if center - (trim // 2) > mins[0]:
            while center - (trim // 2) > mins[0]:
                center = center - 1
            center = center + margin

        if center + (trim // 2) < maxs[0]:
            while center + (trim // 2) < maxs[0]:
                center = center + 1
            center = center + margin
    
    top    = max(0, center - (trim //2))
    bottom = trim if top == 0 else center + (trim//2)

    if bottom > image.shape[trimaxes]:
        bottom = image.shape[trimaxes]
        top = image.shape[trimaxes] - trim
  
    if trimaxes == 0:
        image   = image[top: bottom, :, :]
    else:
        image   = image[:, top: bottom, :]

    if seg is not None:
        if trimaxes == 0:
            seg   = seg[top: bottom, :, :]
        else:
            seg   = seg[:, top: bottom, :]

        return image, seg
    else:
        return image    
