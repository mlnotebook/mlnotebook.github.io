+++
date = "2017-07-17T09:37:58+01:00"
title = "Generative Adversarial Network (GAN) in TensorFlow - Part 4"
tags = ["GAN", "CNN", "machine learning", "generative", "tensorflow"]
description = "The GAN Class and Data Functions"
topics = ['tutorial']
featured_image="/img/featgan4.png"
social = true
+++

Now that we're able to import images into our network, we really need to build the GAN iteself. This tuorial will build the GAN `class` including the methods needed to create the generator and discriminator. We'll also be looking at some of the data functions needed to make this work.


<!--more-->

*Note: This table of contents does not follow the order in the post. The contents is grouped by the methods in the GAN `class` and the functions in `gantut_imgfuncs.py`.

<div id="toctop"></div>

1. [Introduction][1]
2. [The GAN][2]
	* [dataset_files()][3]
	* [GAN Class][4]
		* [\_\_init\_\_()][5]
		* [discriminator()][7]
		* [generator()][11]
		* [build_model()][13]
		* [save()][14]
		* [load()][15]
		* [train()][16]				
	* [Data Functions][6]
		* [batch_norm()][6]	
		* [conv2d()][8]
		* [relu()][9]
		* [linear()][10]
		* [conv2d_transpose()][12]				
3. [Conclusion][17]

[100]:{{< relref "#toctop" >}}
[1]:{{< relref "#intro" >}}
[2]:{{< relref "#gan" >}}
[3]:{{< relref "#datasetfiles" >}}
[4]:{{< relref "#dcgan" >}}
[5]:{{< relref "#init" >}}
[6]:{{< relref "#batchnorm" >}}
[7]:{{< relref "#discriminator" >}}
[8]:{{< relref "#conv2d" >}}
[9]:{{< relref "#relu" >}}
[10]:{{< relref "#linear" >}}
[11]:{{< relref "#generator" >}}
[12]:{{< relref "#conv2dtrans" >}}
[13]:{{< relref "#buildmodel" >}}
[14]:{{< relref "#save" >}}
[15]:{{< relref "#load" >}}
[16]:{{< relref "#train" >}}
[17]:{{< relref "#conclusion" >}}

<h2 id="intro"> Introduction </h2>

In the last tutorial, we build the functions in `gantut_imgfuncs.py`which allow us to import data into our networks. The completed file is [here](/docs/GAN/gantut_imgfuncs_complete.py "gantut_imgfuncs_complete.py"). In this tutorial we will be working on the final two code skeletons:

* [`gantut_gan.py`](/docs/GAN/gantut_gan.py "gantut_gan.py")
* [`gantut_datafuncs.py`](/docs/GAN/gantut_datafuncs.py "gantut_datafuncs.py")

First, let's take a look at the various parts of our GAN in the `gantut_gan.py` file and see what they're going to do.

<h2 id="gan"> The GAN </h2>

We're going to import a number of modules for this file including those from our own `gantut_datafuncs.py` and `gantut_imgfuncs.py`:

```python
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
```

<h3 id="datasetfiles"> dataset_files() </h3>

The initial part of this file is a little housekeeping - ensuring that we are only dealing with supported filetypes. This way of doing things I liked in [B. Amos blog](http://bamos.github.io/2016/08/09/deep-completion/#ml-heavy-generative-adversarial-net-gan-building-blocks "B. Amos"). We define accepted file-extensions and then return a list of all of the possible files we can use for training purposes. the `itertools.chain.from_iterable` function is useful for create a single `list` of all of the files found in the folders and subfolders of a particular `root` with an appropriate `ext`. Notice that it doesn't really matter what we call the images, so this will work for all datasets.

```python
SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg"]

""" Returns the list of all SUPPORTED image files in the directory
"""
def dataset_files(root):
    return list(itertools.chain.from_iterable(
    glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))
```

<hr>

<h3 id="dcgan"> DCGAN() </h3>

This is where the hard work begins. We're going to build the DCGAN `class` (i.e. Deep Convolutional Generative Adversarial Network). The skeleton code already has the necessary method names for our model, let's have a look at what we've got to create:

* `__init__`:  &emsp;to initialise the model and set parameters
* `build_model`: &emsp;creates the model (or 'graph' in TensorFlow-speak) by calling...
* `generator`: &emsp;defines the generator network
* `discriminator`: &emsp;defines the discriminator network
* `train`: &emsp;is called to begin the training of the network with data
* `save`: &emsp;saves the TensorFlow checkpoints of the GAN
* `load`: &emsp;loads the TensorFlow checkpoints of the GAN

We create an instance of our GAN class with `DCGAN(args)` and be returned a DCGAN object with the above methods. Let's code.

<h4 id="init"> __init__() </h4>

To initialise our GAN object, we need some initial parameters. It looks like this:

```python
def __init__(self, sess, image_size=64, is_crop=False, batch_size=64, sample_size=64, z_dim=100,
             gf_dim=64, df_dim=64, gfc_dim=1024, dfc_dim=1024, c_dim=3, checkpoint_dir=None, lam=0.1):
 ```

The parameters are:

* `sess`: &emsp; the TensorFlow session to run in
* `image_size`: &emsp; the width of the images, which should be the same as the height as we like square inputs
* `is_crop`: &emsp; whether to crop the images or leave them as they are
* `batch_size`: &emsp; number of images to use in each run
* `sample_size`: &emsp; number of z samples to take on each run, should be equal to batch_size
* `z_dim`: &emsp; number of samples to take for each z
* `gf_dim`: &emsp; dimension of generator filters in first conv layer
* `df_dim`: &emsp; dimenstion of discriminator filters in first conv layer
* `gfc_dim`: &emsp; dimension of generator units for fully-connected layer
* `dfc_gim`: &emsp; dimension of discriminator units for fully-connected layer
* `c_dim`: &emsp; number of image cannels (gray=1, RGB=3)
* `checkpoint_dir`: &emsp; where to store the TensorFlow checkpoints
* `lam`: &emsp;small constant weight for the sum of contextual and perceptual loss

These are the controllable parameters for the GAN. As this is the initialising function, we need to transfer these inputs to the `self` of the class so they are accessible later on. We will also add two new lines:

* Let's add a check that the `image_size` is a power of 2 (to make the convolution work well). This clever 'bit-wise-and' operator `&` will do the job for us. It uses the unique property of all power of 2 numbers have only one bit set to `1` and all others to `0`. Let's also check that the image is bigger than $[8  \times 8]$ to we don't convolve too far:

* Get the `image_shape` which is the width and height of the image along with the number of channels (gray or RBG).

```python
#image_size must be power of 2 and 8+
assert(image_size & (image_size - 1) == 0 and image_size >= 8)

self.sess = sess
self.is_crop = is_crop
self.batch_size = batch_size
self.image_size = image_size
self.sample_size = sample_size
self.image_shape = [image_size, image_size, c_dim]

self.z_dim = z_dim
self.gf_dim = gf_dim
self.df_dim = df_dim        
self.gfc_dim = gfc_dim
self.dfc_dim = dfc_dim

self.lam = lam
self.c_dim = c_dim
```

Later on, we will want to do 'batch normalisation' on our data to make sure non of our images are extremely different to the others. We will need a batch-norm layer for each of the conv layers in our generator and discriminator. We will initialise the layers here, but define them in our `gantut_datafuncs.py` file shortly.

```python
#batchnorm (from funcs.py)
self.d_bns = [batch_norm(name='d_bn{}'.format(i,)) for i in range(4)]

log_size = int(math.log(image_size) / math.log(2))
self.g_bns = [batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]
```

This shows that we will be using 4 layers in our discriminator. But we will need more in our generator: our generator starts with a simple vector *z* and needs to upscale to the size of `image_size`. It does this by a factor of 2 in each layer, thus $\log(\mathrm{image \ size})/\log(2)$ is equal to the number of upsamplings to be done i.e. $2^{\mathrm{num \ of \ layers}} = 64$ in our case. Also note that we've created these objects (layers) with an iterator so that each has the name `g_bn1`, `g_bn1` etc.

To finish `__init__()` we set the checkpoint directory for TensorFlow saves, instruct the class to build the model and name it 'DCGAN.model'.

```python
self.checkpoint_dir = checkpoint_dir
self.build_model()

self.model_name="DCGAN.model"
```

<hr>

<h4 id="batchnorm"> batch_norm() </h4>

This is the first of our `gantut_datafuncs.py` functions.

If some of our images are very different to the others then the network will not learn the features correctly. To avoid this, we add batch normalisation (as described in [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift - Ioffe & Szegedy (2015)](http://arxiv.org/abs/1502.03167 "Batch Normalization: Sergey Ioffe, Christian Szegedy"). We effectively redistribute the intensities of the images around a common mean with a set variance.

This is a `class` that will be instantiated with set parameters when called. Then, the method will perform batch normalisation whenever the object is called on the set of images `x`. We are using Tensorflow's built-in [tf.contrib.layers.batch_norm()](https://www.tensorflow.org/api_docs/python/tf/contrib/layers/batch_norm "tf.contrib.layers.batch_norm") layer for this which implements the method from the paper above.

*Parameters*

* `epsilon`:    'small float added to variance [of the input data] to avoid division by 0'
* `momentum`:   'decay value for the moving average, usually 0.999, 0.99, 0.9'

*Inputs*

* `x`:      the set of input images to be normalised
* `train`:  whether or not the network is in training mode [True or False]

*Returns*

* A batch_norm 'object' on instantiation
* A tensor representing the output of the batch_norm operation

```python
"""Batch normalisation function to standardise the input
Initialises an object with all of the batch norm properties
When called, performs batch norm on input 'x'
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
```

<hr>

<h4 id="discriminator"> discriminator() </h4>

As the discriminator is a simple [convolutional neural network (CNN)](/post/CNN1 "MLNotebook: Convolutional Neural Network") this will not take many lines. We will have to create a couple of wrapper functions that will perform the actual convolutions, but let's get the method written in `gantut_gan.py` first.

We want our discriminator to check a real `image`, save varaibles and then use the same variables to check a fake `image`. This way, if the images are fake, but fool the discriminator, we know we're on the right track. Thus we use the variable `reuse` when calling the `discriminator()` method - we will set it to `True` when we're using the fake images.

We add `tf.variable_scope()` to our functions so that when we visualise our graph in TensorBoard we can recognise the various pieces of our GAN.

Next are the definitions of the 4 layers of our discriminator. each one takes in the images, the kernel (filter) dimensions and has a name to identify it later on. Notice that we also call our `d_bns` objects which are the batch-norm objects that were set-up during instantiation of the GAN. These act on the result of the convolution before being passed through the non-linear `lrelu` function. The last layer is just a `linear` layer that outputs the unbounded results from the network.

As this is a classificaiton task (real or fake) we finish by returning the probabilities in the range $[0 \ 1]$ by applying the sigmoid function. The full output is also returned.

```python
def discriminator(self, image, reuse=False):
	with tf.variable_scope("discriminator") as scope:
	    if reuse:
		scope.reuse_variables()
	   	    
	    h0 = lrelu(conv2d(image, self.df_dim, name='d_h00_conv'))
	    h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim*2, name='d_h1_conv'), self.is_training))
	    h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim*4, name='d_h2_conv'), self.is_training))
	    h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim*8, name='d_h3_conv'), self.is_training))
	    h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
	    
	    return tf.nn.sigmoid(h4), h4
```

This method calls a couple of functions that we haven't defined yet: `cov2d`, `lrelu` and `linear` so lets do those now.

<hr>

<h4 id="conv2d"> conv2d() </h4>

This function we've seen before in our [CNN](/post/CNN1 "MLNotebook: Convolutional Neural Networks") tutorial. We've defined the weights `w` for each kernel which is `[k_h x k_w x number of images x number of kernels]`not forgetting that different weights are learned for different images. We've initialised these weights using a standard, random sampling from a normal distribution with standard deviation `stddev`.

The convolution is done by TensorFlow's [tf.nn.conv2d]( "tf.nn.conv2d") function using the weights `w` we've already defined. The padding option `SAME` makes sure that we end up with output that is the same size as the input. Biases are added (the same size as the number of kernels and initialised at a constant value) before the result is returned.

*Inputs*

* `input_`:     the input images (full batch)
* `output_dim`: the number of kernels/filters to be learned
* `k_h`, `k_w`:   height and width of the kernels to be learned
* `d_h`, `d_w`:   stride of the kernel horizontally and vertically
* `stddev`:     standard deviation for the normal func in weight-initialiser

*Returns*

* the convolved images for each kernel

```python
"""Defines how to perform the convolution for the discriminator,
i.e. traditional conv rather than reverse conv for the generator
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
```



<hr>

<h4 id="relu"> relu() </h4>

The network need to be able to learn complex functions, so we add some non-linearity to the output of our convolution layers. We've seen this before in our tutorial on [transfer functions](/post/transfer_functions "Transfer Functions"). Here we use the leaky rectified linear unit (lReLU).

*Parameters*

* `leak`:   the 'leakiness' of the lrelu

*Inputs*

* `x`: some data with a wide range

*Returns*

* the transformed input data

```python
"""Neural nets need this non-linearity to build complex functions
"""
def lrelu(x, leak=0.2, name="lrelu"):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)
```

<hr>

<h4 id="linear"> linear() </h4>

This linear layer takes the outputs from the convolution and does a linear transform using some randomly initialised weights. This does not have the same non-linear property as the `lrelu` function because we will use this output to calcluate probabilities for classification. We return the result of `input_ x matrix` by default, but if we also need the weights, we also output `matrix` and `bias` through the `if` statement.

*Parameters*

* `stddev`:     standard deviation for weight initialiser
* `bias_start`: for the bias initialiser (constant value)
* `with_w`:     return the weight matrix (and biases) as well as the output if True

*Inputs*

* `input_`:         input data (shape is used to define weight/bias matrices)
* `output_size`:    desired output size of the linear layer 

```python
"""For the final layer of the discriminator network to get the
full detail (probabilities etc.) from the output
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
```

<hr>

<h4 id="generator"> generator() </h4>

Finally! We're going to write the code for the generative part of the GAN. This method will take a single input - the randomly-sampled vector $z$ from the well known distribution $p_z$.

Remember that the generator is effectively a reverse discriminator in that it is a CNN that works backwards. Thus we start with the 'values' and must perform the linear transformation on them before feeding them through the other layers of the network. As we do not know the weights or biases yet in this network, we need to make sure we output these from the linear layer with `with_w=True`.

This first hidden layer `hs[0]` needs reshaping to be the small image-shaped array that we can send through the network to become the upscaled $[64 \times 64]$ image at the end. So we take the linearly-transformed z-values and reshape to $[4 x 4 x num_kernels]$. Don't forget the `-1` to do this for all images in the batch. As before, we must batch-norm the result and pass it through the non-linearity.

The number of layers in this network has been calculated earlier (using the logarithm ratio of image size to downsampling factor. We can therefore do the next part of the generator in a loop.

In each loop/layer we are going to:

1. give the layer a name
2. perform the *inverse* convolution
3. apply non-linearity

1 and 3 are self-explanatory, but the inverse convolution function still needs to be written. This is the function that will take in the small square image and upsample it to a larger image using some weights that are being learnt. We start at layer `i=1` where we want the image to go to `size=8` from `size=4` at layer `i=0`. This will increase by a factor of 2 at each layer. As with a regular CNN we want to learn fewer kernels on the larger images, so we need to decrease the `depth_mul` by a factor of 2 at each layer. Note that the `while` loop will terminate when the size gets to the size of the input images `image_size`.

The final layer is added which takes the last output and does the inverse convolution to get the final fake image (that will be tested with the discriminator.

```python
def generator(self, z):
	with tf.variable_scope("generator") as scope:
	    self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)

	    hs = [None]
	    hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
	    hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))
	    
	    i=1             #iteration number
	    depth_mul = 8   #depth decreases as spatial component increases
	    size=8          #size increases as depth decreases
	    
	    while size < self.image_size:
		hs.append(None)
		name='g_h{}'.format(i)
		hs[i], _, _ = conv2d_transpose(hs[i-1], [self.batch_size, size, size, self.gf_dim*depth_mul],
		                                name=name, with_w=True)
		hs[i] = tf.nn.relu(self.g_bns[i](hs[i], self.is_training))
		
		i += 1
		depth_mul //= 2
		size *= 2
		
	    hs.append(None)
	    name = 'g_h{}'.format(i)
	    hs[i], _, _ = conv2d_transpose(hs[i-1], [self.batch_size, size, size, 3], name=name, with_w=True)
	    
	    return tf.nn.tanh(hs[i])           
```

<hr>

<h4 id="conv2dtrans"> conv2d_transpose() </h4>

The inverse convolution function looks very similar to the forward convolution function. We've had to make sure that different versions of TensorFlow work here - in newer versions, the correct function is located at [tf.nn.conv2d_transpose](https://www.tensorflow.org/api_docs/python/tf/nn/conv2d_transpose "tf.nn.conv2d_transpose") where as in older ones we must use `tf.nn.deconv2d`.

*Inputs*

* `input_`:         a vector (of noise) with dim=batch_size x z_dim
* `output_shape`:   the final shape of the generated image
* `k_h`, `k_w`:       the height and width of the kernels
* `d_h`, `d_w`:       the stride of the kernel horiz and vert.    

*Returns*

* an image (upscaled from the initial data)

```python
"""Deconv isn't an accurate word, but is a handy shortener,
so we'll use that. This is for the generator that has to make
the image from some randomly sampled data
"""
def conv2d_transpose(input_, output_shape, k_h=5, k_w=5, d_h=2, d_w=2, stddev=0.02,
                     name="conv2d_transpose", with_w=False):
    with tf.variable_scope(name):
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
```


<hr>

<h4 id="buildmodel"> build_model() </h4>

The `build_model()` method bring together the image data and the generator and discriminator methods. This is the 'graph' for TensorFlow to follow. It contains some `tf.placeholder` pieces which we must supply attributes to when we finally train the model.

We will need to know whether the model is in training or inference mode throughout our code, so we have a placeholder for that variable. We also need a placeholder for the image data itself because there will be a different batch of data being injected at each epoch. These are our `real_images`.

When we inject the `z` vectors into the GAN (served by another palceholder) we will also produce some monitoring output for TensorBoard. By adding `tf.summary.histogram()` we are able to keep track of how the different `z` vectors look at each epoch.

```python
    def build_model(self):
        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        self.lowres_images = tf.reduce_mean(tf.reshape(self.images,
            [self.batch_size, self.lowres_size, self.lowres,
             self.lowres_size, self.lowres, self.c_dim]), [2, 4])
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.z_sum = tf.summary.histogram("z", self.z)
```

Next, lets tell the graph to take the injected `z` vector an turn it into an image with our `generator`. We'll also produce a lowres version of this image. Now, put the 'real\_images' into the `discriminator`, which gives back our probabilities and the final-layer data (the logits). We then `reuse` the same discriminator parameters to test the fake image from the generator. Here we also output some histograms of the probabilities of the 'real\_image' and the fake image. We will also output the current fake image from the generator to TensorBoard.

```python
        self.G = self.generator(self.z)
        self.lowres_G = tf.reduce_mean(tf.reshape(self.G,
            [self.batch_size, self.lowres_size, self.lowres,
             self.lowres_size, self.lowres, self.c_dim]), [2, 4])
        self.D, self.D_logits = self.discriminator(self.images)

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.G_sum = tf.summary.image("G", self.G)
```

Now for some of the necessary calculations needed to be able to update the network. Let's find the 'loss' on the current outputs. We will utilise a very efficient loss function here the [tf.nn.sigmoid_cross_entropy_with_logits](https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits "tf.nn.sigmoid_cross_entropy_with_logits"). We want to calculate a few things:

1. how well did the discriminator do at letting *true* images through (i.e. comparing `D` to `1`)
2. how often was the discriminator fooled by the generator  (i.e. comparing `D_` to `1`)
3. how often did the generator fail at making realistic images (i.e. comparing `D_` to `0`).

We'll add the discriminator losses up (1 + 2) and create a TensorBoard summary statistic (a `scalar` value) for the discriminator and generator losses in this epoch. These are what we will optimise during training.

To keep everything tidy, we'll group the discriminator and generator variables into `d_vars` and `g_vars` respectively.

```python
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]
```

We don't want t lose our progress, so lets make sure we setup the `tf.Saver()` function just keeping the most recent variables each time.

```python
        self.saver = tf.train.Saver(max_to_keep=1)
```

<hr>

<h4 id="save"> save() </h4>

When we want to save a checkpoint (i.e. save all of the weights we've learned) we will call this function. It will check whether the output directory exists, if not it will create it. Then it wll call the [`tf.train.Saver.save()`]( https://www.tensorflow.org/api_docs/python/tf/train/Saver#save "tf.train.Saver.save") function which takes in the current session `sess`, the save directory, model name and keeps track of the number of steps that've been done.

```python
    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
            
        self.saver.save(self.sess, os.path.join(checkpoint_dir, self.model_name), global_step=step)
```

<hr>

<h4 id="load"> load() </h4>

Equally, if we've already spent a long time learning weights, we don't want to start from scratch every time we want to push the network further. This function will load the most recent checkpoint in the save directory. TensorFlow has build-in functions for checking out the most recent checkpoint. If there is no checkpoint available, the function returns false and the appropriate action is taken by the main method that called it.

```python
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")
        
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
```

<hr>

<h4 id="train"> train() </h4>

The all-important `train()` method. This is where the magic happens. When we call `DCGAN.train(config)` the networks will begin their fight and train. We will discuss the `config` argument later on, but succinctly: it's a list of all hyperparameters TensorFlow will use in the network. Here's how `train()` works:

First we give the trainer the data (using our `dataset_files` function) and make sure that it's randomly shuffled. We want to make sure that the images next to each other have nothing in common so that we can truly randomly sample them. There's also a check here ``assert(len(data) > 0)` to make sure that we don't pass in an empty directory... that wouln't be useful to learn from.

```python
def train(self, config):
	data = dataset_files(config.dataset)
	np.random.shuffle(data)
	assert(len(data) > 0)
```

We're going to use the adaptive non-convex optimization method [`tf.train.AdamOptimizer()`](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer "tf.train.AdamOptimizer") from [Kingma *et al* (2014)](https://arxiv.org/pdf/1412.6980.pdf "Adam: A Method for Stochastic Optimization") to train out networks. Let's set this up for the discriminator (`d_optim`) and the generator (`g_optim`).

```python
	d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.d_loss, var_list=self.d_vars)
	g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1).minimize(self.g_loss, var_list=self.g_vars)
```

Next we will initialize all variables in the network (depending on TensorFlow version) and generate some `tf.summary` variables for TensorBoard which group together all of the summaries that we want to keep track of.

```python
	try:
	    tf.global_variables_initializer().run()
	except:
	    tf.initialize_all_variables().run()
	    
	self.g_sum = tf.summary.merge([self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
	self.d_sum = tf.summary.merge([self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
	self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
```

So here's the part where we now sample this well-known distribution $p_z$ to get the noise vector $z$. We're using a `np.random.uniform` distribution. Keep a look out for this when we're watching the network in TensorBoard, we told the GAN `class` to output the histogram of $z$ vectors that are sampled from $p_z$. So they should all approximate to a uniform distribution.

We're also going to sample the input *real* image files we shuffled earlier taking `sample_size` images through to the training process. We will use these later on to assess the loss functions every now and again when we output some examples.

We need to load in the data using the function `get_image()` that we wrote into `gantut_imgfuncs.py` during the [last tutorial](/post/GAN3 "MLNotebook: GAN3"). After loading the images, lets make sure that they're all in one `np.array` ready to be used.

```python
	sample_z = np.random.uniform(-1, 1, size=(self.sample_size, self.z_dim))

	sample_files = data[0:self.sample_size]
	sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
	sample_images = np.array(sample).astype(np.float32)
```

Set the epoch counter and get the start time (it can be frustrating if we can't see how long things are taking). We also want to be sure to load any previous checkpoints from TensorFlow before we start again from scratch.

```python
	counter = 1
	start_time = time.time()

	if self.load(self.checkpoint_dir):
	    print(""" An existing model was found - delete the directory or specify a new one with --checkpoint_dir """)
	else:
	    print(""" No model found - initializing a new one""")
```

Here's the actual training bit taking place.  `For` each `epoch` that we've assigned in `config`, we create two minibatches: a sampling of real images, and those generated from the $z$ vector. We then update the `discriminator` network before updating the `generator`. We also write these loss values to the TensorBoard summary. There are two things to notice:

* By calling `sess.run()` with specified variables in the first (or `fetch` attribute) we are able to keep the generator steady whilst updating the discriminator, and vice versa.

* The generator is updated twice. This is to make sure that the discriminator loss function does not just converge to zero very quickly.


```python
	for epoch in xrange(config.epoch):
	    data = dataset_files(config.dataset)
	    batch_idxs = min(len(data), config.train_size) // self.batch_size
	    
	    for idx in xrange(0, batch_idxs):
		batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
		batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
		batch_images = np.array(batch).astype(np.float32)
		
		batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)
		
		#update D network
		_, summary_str = self.sess.run([d_optim, self.d_sum],
		                               feed_dict={self.images: batch_images, self.z: batch_z, self.is_training: True})
		self.writer.add_summary(summary_str, counter)
		
		#update G network
		_, summary_str = self.sess.run([g_optim, self.g_sum],
		                               feed_dict={self.z: batch_z, self.is_training: True})
		self.writer.add_summary(summary_str, counter)
		
		#run g_optim twice to make sure that d_loss does not go to zero
		_, summary_str = self.sess.run([g_optim, self.g_sum],
		                               feed_dict={self.z: batch_z, self.is_training: True})
		self.writer.add_summary(summary_str, counter)

```

To get the errors needed for backpropagation, we evaluate `d_loss_fake`, `d_loss_real` and `g_loss`. We run the $z$ vector through the graph to get the fake loss and the generator loss, and use the real `batch_images` for the real loss.

```python
		errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
		errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
		errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})
```

Let's get some output to `stdout` for the user. The current epoch and progress through the minibatches is output at each new minibatch. Every 100 minibatches we're going to evaluate the current generator `self.G` and calculate the loss against the small set of images we sampled earlier. We will output the result of the generator and use our `save_images()` function to create that image array we worked on in the last tutorial.

```python
		counter += 1
		print("Epoch [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}".format(
		        epoch, idx, batch_idxs, time.time() - start_time, errD_fake + errD_real, errG))
		
		if np.mod(counter, 100) == 1:
		    samples, d_loss, g_loss = self.sess.run([self.G, self.d_loss, self.g_loss], 
		                                            feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False})
		    save_images(samples, [8,8], './samples/train_{:02d}-{:04d}.png'.format(epoch, idx))
		    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
```
Finally, we need to save the current weights from our networks.

```python 
		if np.mod(counter, 500) == 2:
		    self.save(config.checkpoint_dir, counter)
```

<h2 id="conclusion"> Conclusion </h2>

That's it! We've completed the `gantut_gan.py` and `gantut_datafuncs.py` files. Checkout the completed files below:

Completed versions of:

* [gantut_trainer.py](/docs/GAN/gantut_trainer.py "gantut_trainer.py")
* [gantut_imgfuncs_complete.py](/docs/GAN/gantut_imgfuncs_complete.py "gantut_imgfuncs_complete.py")
* [gantut_datafuncs_complete.py](/docs/GAN/gantut_datafuncs_complete.py "gantut_datafuncs_complete.py")
* [gantut_gan_complete.py](/docs/GAN/gantut_gan_complete.py "gantut_gan_complete.py")

 By following this tutorial series we should now have:

1. A background in how GANs work
2. Necessary data, fullly pre-processed and ready to use
3. The `gantut_imgfuncs.py` for loading data into the neworks
4. A GAN `class` with the necessary methods in `gantut_gan.py` and the `gantut_datafuncs.py` we need to do the computations.

In the final part of the series, we will run this network and take a look at the outputs in TensorBoard.