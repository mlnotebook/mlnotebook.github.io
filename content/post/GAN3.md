+++
topics = ["tutorial"]
date = "2017-07-13T09:16:32+01:00"
title = "Generative Adversarial Network (GAN) in TensorFlow - Part 3"
tags = ["GAN", "machine learning", "CNN", "generative", "tensorflow"]
description = "Image Import and Export Functions"
social=true
featured_image="/img/featgan3.png"
+++

We're ready to code! In [Part 1](/content/post/GAN1 "GAN Tutorial - Part 1) we looked at how GANs work and [Part 2](/content/post/GAN2 "GAN Tutorial - Part 2) showed how to get the data ready. In this Part, we will begin creating the functions that handle the image data including some pre-procesing and data normalisation.

<!--more-->

<div id="toctop"></div>

1. [Introduction][1]
2. [Image Functions][2]
	1. [Importing Functions][3]
		* [imread()][6]
		* [transform()][7]
		* [center_crop()][8]
		* [get_image()][9]
	2. [Saving Functions][4]
		* [inverse_transform][10]
		* [merge()][11]
		* [imsave()][12]
		* [save_images()][13]
5. [Conclusion][5]

[100]:{{< relref "#toctop" >}}
[1]:{{< relref "#intro" >}}
[2]:{{< relref "#imagefuncs" >}}
[3]:{{< relref "#importfuncs" >}}
[4]:{{< relref "#savingfuncs" >}}
[5]:{{< relref "#conclusion" >}}
[6]:{{< relref "#imread" >}}
[7]:{{< relref "#transform" >}}
[8]:{{< relref "#centercrop" >}}
[9]:{{< relref "#getimage" >}}
[10]:{{< relref "#invtransform" >}}
[11]:{{< relref "#merge" >}}
[12]:{{< relref "#imsave" >}}
[13]:{{< relref "#saveimages" >}}

<h2 id="intro"> Introduction </h2> 

In the [previous post](/content/post/GAN2 "GAN Tutorial - Part 2) we downloaded and pre-processed our training data. There were also links to the skeleton code we will be using in the remainder of the tutorial, here they are again:

1. [`gantut_imgfuncs.py`](/docs/GAN/gantut_imgfuncs.py "gantut_imgfuncs.py"): holds the image-related functions
2. [`gantut_datafuncs.py`](/docs/GAN/gantut_datafuncs.py "gantut_datafuncs.py"): contains the data-related functions
3. [`gantut_gan.py`](/docs/GAN/gantut_gan.py "gantut_gan.py"): is where we define the GAN `class`
4. [`gantut_trainer.py`](/docs/GAN/gantut_trainer.py "gantut_trainer.py"): is the script that we will call in order to train the GAN

Again, the code is based from other sources, particularly the respository by [carpedm20](https://github.com/carpedm20/DCGAN-tensorflow "carpedm20/DCGAN-tensorflow") and [B. Amos](http://bamos.github.io/2016/08/09/deep-completion/#ml-heavy-generative-adversarial-net-gan-building-blocks "bamos.github.io").

Now, if your folder structure that looks something like this then we're ready to go:

```bash
~/GAN
  |- raw
    |-- 00001.jpg
    |-- ...
  |- aligned
    |-- 00001.jpg
    |-- ...
  |- gantut_imgfuncs.py
  |- gantut_datafuncs.py
  |- gantut_gan.py
  |- gantut_trainer.py
```

<h2 id="imagefuncs"> Image Functions </h2>

We're going to want to be able to read-in a set of images. We will also want to be able to output some generated images. We will also add in a fail-safe cropping/transformation procedure in-case we want to make sure we have the right input format. The skeleton code `gantut_imgfuncs.py` contains the definition headers for these functions, we will fill them in as we go along.

<h3 id="importfuncs"> Importing Functions </h3>

These are the functions needed to get the data from the hard-disk into our network. They are called like this:

1. `get_image` which calls
2. `imread` and 
3. `transform` which calls
4. `center_crop`

<h4 id="imread"> imread() </h4>

We are dealing with standard image files and our GAN will support `.jpg`, `.jpeg` and `.png` as input. For these kind of files, Python already has well-developed tools: specifically we can use the [scipy.misc.imread](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.misc.imread.html "imread documentation") function from the `scipy.misc` library. This is a one-liner and is already written in the skeleton code.


*Inputs*

* `path`: location of the image

*Returns*

* the image

```python
""" Reads in the image (part of get_image function)
"""
def imread(path):
    return scipy.misc.imread(path, mode='RGB').astype(np.float)
```

<hr>

<h4 id="transform"> transform() </h4>
[to top][100]

This function we will have to write into the skeleton. We are including this to make sure that the image data are all of the same dimensions. So this function will need to take in the image, the desired width (the output will be square) and whether to perform the cropping or not. We may have already cropped our images (as we have) because we've done some registration/alignment etc.

We do a check on whether we want to crop the image, if we do then call the `center_crop` function, other wise, just take the `image` as it is.

Before returning our cropped (or uncropped) image, we are going to perform normalisation. Currently the pixels have intensity values in the range $[0 \ 255]$ for each channel (reg, green, blue). It is best not to have this kind of skew on our data, so we will normalise our images to have intensity values in the range $[-1 \ 1]$ by dividing by the mean of the maximum range (127.5) and subtracting 1. i.e. image/127.5 - 1. 

We will define the cropping function next, but note that the returned image is a simply a `numpy` array.

*Inputs*

* `image`:      the image data to be transformed
* `npx`:        the size of the transformed image [`npx` x `npx`]
* `is_crop`:    whether to preform cropping too [`True` or `False`]

*Returns*

* the cropped, normalised image

```python
""" Transforms the image by cropping and resizing and 
normalises intensity values between -1 and 1
"""
def transform(image, npx=64, is_crop=True):
    if is_crop:
        cropped_image = center_crop(image, npx)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.
```

<hr>

<h4 id="centercrop"> center_crop() </h4>

Lets perform the cropping of the images (if requested). Usually we deal with square images, say $[64 \times 64]$. We can add a quick option to change that with short `if` statements looking at the `crop_w` argument to this function. We take the current height and width (`h` and `w`) from the `shape` of the image `x`.

To find the location of the centre of the image around which to take the square crop, we take half the result of `h - crop_h` and `w - crop_w`, making sure to round both to get a definite pixel value. However, it's not guaranteed (depending on the image dimensions) that we will end up with a nice $[64 \times 64]$ image. Let's fix that at the end.

As before, `scipy` has some efficient functions that we may as well use. [`scipy.misc.imresize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.misc.imresize.html "imresize documentation") takes in an image array and the desired size and outputs a resized image. We can give it our array, which may not be a nice square image due to the initial image dimensions, and `imresize` will perform interpolation (bilinear by default) to make sure we get a nice square image at the end.

*Inputs*

* `x`:      the input image
* `crop_h`: the height of the crop region
* `crop_w`: if None crop width = crop height
* `resize_w`: the width of the resized image

*Returns*

* the cropped image

```python
""" Crops the input image at the centre pixel
"""
def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h
    h, w = x.shape[:2]
    j = int(round((h - crop_h)/2.))
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])
```

<hr>

<h4 id="getimage"> get_image() </h4>

The `get_image` function is a wrapper that will call the `imread` and `transform` functions. It is the function that we'll call to get the data rather than doing two separate function calls in the main GAN `class`. This is a one-liner and is already written in the skeleton code.

*Parameters*

* `is_crop`:    whether to crop the image or not [True or False]

*Inputs*

* `image_path`: location of the image
* `image_size`: width (in pixels) of the output image

*Returns*

* the cropped image

```python
""" Loads the image and crops it to 'image_size'
"""
def get_image(image_path, image_size, is_crop=True):
    return transform(imread(image_path), image_size, is_crop)
```


<hr>

<h3 id="savingfuncs"> Saving Functions </h3>

When we're training our network, we will want to see some of the results. The previous functions all deal with getting images from storage *into* the networks. We now want to take some images *out*. The functions are called like this:

1. `save_images` which calls
2. `inverse_transform` and
3. `imsave` which calls
4. `merge`


<h4 id="invtransform"> inverse_transform() </h4>

Firstly, let's put the intensities back into the skewed range, we'll just go from $[-1 \ 1]$ to $[0 \ 1]$ here. 

*Inputs*

* `images`:     the image to be transformed

*Returns*

* the transformed image

```python
""" This turns the intensities back to a normal range
"""
def inverse_transform(images):
    return (images+1.)/2.
```

<hr>
 
<h4 id="merge"> merge() </h4>

We will create an array of several example images from the network which we can output every now and again to see how things are progressing. We need some `images` to go in and a `size` which will say how many images in width and height the array should be.

First get the height `h` and width `w` of the `images` from their `shape` (we assume they're all the same size becuase we will have already used our previous functions to make this happen). **Note** that `images` is a collection of images where each `image` has the same `h` and `w`.

We define `img` to be the final image array and initialise it to all zeros. Notice that there is a '3' on the end to denote the number of channels as these are RGB images. This will still work for grayscale images.

Next we will iterate through each `image` in `images` and put it into place. The `%` operator is the modulo which returns the remainder of the division between two numbers. `//` is the floor division operator which returns the integer result of division rounded down. So this will move along the top row of the array (remembering Python indexing starts at 0) and move down placing the image at each iteration.

*Inputs*

* `images`:     the set of input images
* `size`:       [height, width] of the array

*Returns*

* an array of images as a single image

```python
""" Takes a set of 'images' and creates an array from them.
""" 
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((int(h * size[0]), int(w * size[1]), 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image
        
    return img
```

<hr>
    
<h4 id="imsave"> imsave() </h4>

Our image array `img` now has intensity values in $[0 \ 1]$ lets make this the proper image range $[0 \ 255]$ before getting the integer values as an image array with [`scipy.misc.imsave`](https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.misc.imsave.html "imsave documentation").

*Inputs*

* `images`: the set of input images
* `size`:   [height, width] of the array
* `path`:   the save location

*Returns*

* an image saved to disk

```python
""" Takes a set of `images` and calls the merge function. Converts
the array to image data and saves to disk.
"""
def imsave(images, size, path):
    img = merge(images, size)
    return scipy.misc.imsave(path, (255*img).astype(np.uint8))
```

<hr>

<h4 id="saveimages"> save_images() </h4>

Finally, let's create the wrapper to pull this together:

*Inputs*

* `images`: the images to be saves
* `size`: the size of the img array [width height]
* `image_path`: where the array is to be stored on disk

```python
""" takes an image and saves it to disk. Redistributes
intensity values [-1 1] from [0 255]
"""
def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)
```

<h3 id="conclusion"> Conclusion </h3>

In this post, we've dealt with all of the functions that are needed to import image data into our network and also some that will create outputs so we can see what's going on. We've made sure that we can import any image-size  and it will be dealt with correctly.

Make sure that we've imported `scpipy.misc` and `numpy` to this script:

```python
import numpy as np
import scipy.misc
```

The complete script can be found [here](/docs/GAN/gantut_imgfuncs_complete.py "gantut_imgfuncs_complete.py"). In the next post, we will be working on the GAN itself and building the `gantut_datafuncs.py` functions as we go.




















