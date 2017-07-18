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
        with tf.variable_scope(name):
            self.epsilon = epsilon
            self.momentum = momentum

            self.name = name

    def __call__(self, x, train):
        return tf.contrib.layers.batch_norm(x, decay=self.momentum, updates_collections=None, epsilon=self.epsilon,
                                            center=True, scale=True, is_training=train, scope=self.name)   
    
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
#def binary_cross_entropy(preds, targets, name=None):


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
    with tf.variable_scope(name):
        w = tf.get_variable('w', [k_h, k_w, input_.get_shape()[-1], output_dim],
                            initializer=tf.truncated_normal_initializer(stddev=stddev))
        conv = tf.nn.conv2d(input_, w, strides=[1, d_h, d_w, 1], padding='SAME')

        biases = tf.get_variable('biases', [output_dim], initializer=tf.constant_initializer(0.0))
        # conv = tf.reshape(tf.nn.bias_add(conv, biases), conv.get_shape())
        conv = tf.nn.bias_add(conv, biases)

        return conv   
    
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
    with tf.variable_scope(name):
        # filter : [height, width, output_channels, in_channels]
        w = tf.get_variable('w', [k_h, k_w, output_shape[-1], input_.get_shape()[-1]],
                            initializer=tf.random_normal_initializer(stddev=stddev))

        try:
            deconv = tf.nn.conv2d_transpose(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        # Support for verisons of TensorFlow before 0.7.0
        except AttributeError:
            deconv = tf.nn.deconv2d(input_, w, output_shape=output_shape,
                                strides=[1, d_h, d_w, 1])

        biases = tf.get_variable('biases', [output_shape[-1]], initializer=tf.constant_initializer(0.0))
        # deconv = tf.reshape(tf.nn.bias_add(deconv, biases), deconv.get_shape())
        deconv = tf.nn.bias_add(deconv, biases)

        if with_w:
            return deconv, w, biases
        else:
            return deconv    
    
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
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
        
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
    shape = input_.get_shape().as_list()

    with tf.variable_scope(scope or "Linear"):
        matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32,
                                 tf.random_normal_initializer(stddev=stddev))
        bias = tf.get_variable("bias", [output_size],
            initializer=tf.constant_initializer(bias_start))
        if with_w:
            return tf.matmul(input_, matrix) + bias, matrix, bias
        else:
            return tf.matmul(input_, matrix) + bias
    
    
