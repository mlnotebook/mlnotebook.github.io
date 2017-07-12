import numpy as np
import tensorflow as tf
import scipy.misc

from tensorflow.python.framework import ops

#BATCH NORMALISATION
"""Batch normalisation function to standardise the input
Initialises an object with all of the batch norm properties
When called, performs batch norm on input 'x'

PARAMETERS
epsilon:    'small float added to variance [of the input data] to avoid division by 0'
momentum:   'decay value for the moving average, usually 0.999, 0.99, 0.9'

INPUTS
x:      the set of input images to be normalised
train:  whether or not the network is in training mode [True or False]

RETURNS
 - A batch_norm 'object' on instantiation
 - A tensor representing the output of the batch_norm operation
"""
class batch_norm(object):
    def __init__(self, epsilon=1e-5, momentum = 0.9, name="batch_norm"):
    
    
    def __call__(self, x, train):    
    
#BINOMIAL CROSS ENTROPY LOSS FUNCTION
"""Computes the cross entropy (aka logistic loss) between 'preds' and 'targets'
this is more suitable than mean-squared error etc. for binary classification
i.e. real or fake

INPUTS
preds:      the current predictions of the model
targets:    the true values

RETURNS
- Value of the cross-entropy
"""
def binary_cross_entropy(preds, targets, name=None):


#CONVOLUTION FUNCTION
"""Defines how to perform the convolution for the discriminator,
i.e. traditional conv rather than reverse conv for the generator

INPUTS
input_:     the input images (full batch)
output_dim: the number of kernels/filters to be learned
k_h, k_w:   height and width of the kernels to be learned
d_h, d_w:   stride of the kernel horizontally and vertically
stddev:     standard deviation for the normal func in weight-initialiser

RETURN
- the convolved images x number of kernels
"""
def conv2d(input_, output_dim, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02, name="conv2d"):
    
    
#REVERSE CONVOLUTION FUNCTION
"""Deconv isn't an accurate word, but is a handy shortener,
so we'll use that. This is for the generator that has to make
the image from some randomly sampled data

INPUTS
input_:         a vector (of noise) with dim=batch_size x z_dim
output_shape:   the final shape of the generated image
k_h, k_w:       the height and width of the kernels
d_h, d_w:       the stride of the kernel horiz and vert.    

RETURNS
- an image (upscaled from the initial data)
"""
def conv2d_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                     name="conv2d_transpose", with_w=False):
    
    
#NON-LINEARITY FUNCTION
"""Neural nets need this non-linearity to build complex functions

PARAMETERS
leak:   the 'leakiness' of the lrelu

INPUTS
x: some data with a wide range

RETURNS
- the transformed input data
"""
def lrelu(x, leak=0.2, name="lrelu"):

#LINEAR FUNCTION
"""For the final layer of the discriminator network to get the
full detail (probabilities etc.) from the output

PARAMETERS
stddev:     standard deviation for weight initialiser
bias_start: for the bias initialiser (constant value)
with_w:     return the weight matrix (and biases) as well as the output if True

INPUTS
input_:         input data (shape is used to define weight/bias matrices)
output_size:    desired output size of the linear layer 
"""
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):

    
    
