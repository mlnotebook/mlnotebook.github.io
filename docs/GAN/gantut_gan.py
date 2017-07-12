from __future__ import division
import os
import time
import math
import itertools
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange

#IMPORT OUR IMAGE AND DATA FUNCTIONS
from gantut_datafuncs import *
from gantut_imgfuncs import *

#A LITTLE HOUSEKEEPING TO CATCH ERRORS IN THE DATA
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

""" Returns the list of all SUPPORTED image files in the directory
"""
def dataset_files(root):   
    return list(itertools.chain.from_iterable(
    glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))
    
#DEFINE THE GAN CLASS
"""this class instantiates the GAN. It has methods:

build_model:    initialises the model variables and the graph for the GAN
generator:      the reverse CNN for the generator
discriminator:  the CNN for the discriminator
train:          trains the network (defines the optimisers etc.)
save:           saves the current configuration
load:           loads a saved configuration

RETURNS
- a DCGAN object with the above methods
"""
class DCGAN(object):
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64, lowres=8, z_dim=100,
                 gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1):
    
    
    def build_model(self):
        
        
    def generator(self, z):
        
        
    def discriminator(self, image, reuse=False):
        
    
    def train(self, config):
        
        
    def save(self, checkpoint_dir, step):
        
        
    def load(self, checkpoint_dir):
        
    
    