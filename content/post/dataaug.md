+++
tags = ['CNN', 'neural network', 'deep learning']
draft = true
description = "Getting the most of our training data and improving performance"
date = "2018-01-04T10:13:20Z"
title = "Data Augmentations for n-Dimensional Image Input to CNNs"
topics = ['convolutional neural network', 'data augmentation']
social=true
featured_image=''
+++

One of the greatest limiting factors for training effective deep learning frameworks is the availability, quality and organisation of the *training data*. To be good at classification tasks, we need to show our CNNs *etc.* as many examples as we possibly can. However, this is not always possible especially in situations where the training data is hard to collect e.g. medical image data. In this post, we will learn how to apply *data augmentation* strategies to n-Dimensional images get the most of our limited number of examples.

<!--more-->

<h2 id="intro"> Introduction </h2>

If we take any image, like our little Android below, and we shift all of the data in the image to the right by a single pixel, you may struggle to see any difference visually. However, numerically, this may as well be a completely different image! Imagine taking a stack of 10 of these images, each shifted by a single pixel compared to the previous one. Now consider the pixels in the images at [20, 25] or some arbitrary location. Focusing on that point, each pixel has a different colour, different average surrounding intensity etc. A CNN take these values into account when performing convolutions and deciding upon weights. If we supplied this set of 10 images to a CNN, it would effectively be making it learn that it should be invariant to these kinds of translations.

Of course, translations are not the only way in which an image can change, but still *visually* be the same image. Consider rotating the image by even a single degree, or 5 degrees. It's still an Android. Traning a CNN without including translated and rotated versions of the image may cause the CNN to **overfit** and assume that all images of Androids have to be perfectly upright and centered.

Providing deep learning frameworks with images that are translated, rotated, scaling, intensified and flipped is what we mean when we talk about *data augmentation*.

In this post we'll look at how to apply these transformations to an image, even in 3D and see how it affects the performance of a deep learning framework.


<h2 id="augs"> Augmentations </h2>

As usual, we are going to write our augmentation functions in python. We'll just be using simple functions from `numpy` and `scipy`.

<h3 id="translate"> Translation </h3>

In our functions, `image` is a 2 or 3D array - if it's a 3D array, we need to be careful about specifying our translation directions in the argument called `offset`. We don't really want to move images in the `z` direction for a couple of reasons: firstly, if it's a 2D image, the third dimension will be the colour channel, if we move the image through this dimension, will either become all red, all blue or all black if we move it `-2`, `2` or greater than these respectively; second, in a full 3D image, the 3rd dimension is often the smallest as is the case in medical scans. In our translation function below, the `offset` is given as a length 2 array defining the shift in the `x` and `y` directions. We hard-code z-direction to`0` but you're welcome to change this if your use-case demands it.

~~~python
def translateit(image, offset, isseg=False):
    if isseg:
        order=0
    else:
        order=3
    return scipy.ndimage.interpolation.shift(image, (offset[0], offset[1], 0), order=order, mode='nearest')
~~~

Here we have also provided the option for what kind of interpolation we want to perform: `order=0` means to just use the nearest-neighbour pixel intensity and `order=5` means to perform bspline interpolation with order 5 (taking into account many pixels around the target). This is triggered with a Boolean argument to the `scaleit` function called `isseg` so named because when dealing with image-segmentations, we want to keep their integer class numbers and not get a result which is a float with a value between two classes. This is not a problem with the actual image as we want to retain as much visual smoothness as possible. Similarly, when we move our image, we will leave a gap around the edges from which it's moved. We need a way to fill in this gap: by default `shift` will use a contant value set to `0`. This may not be helpful in some case, so it's best to set the `mode` to `'nearest'` which takes the cloest pixel-value and replicates it.

<h3 id="scale"> Scaling </h3>

When scaling an image, i.e. zooming in and out, we want to increase or decrease the area our image takes up whilst keeping the image dimensions the same. We scale our image by a certain `factor`. A `factor > 1.0` means the image scales-up, and `factor < 1.0` scales the image down. Note that we should provide a factor for each dimension: if we want to keep the same number of layers or slices in our image, we should set last value to `1.0`. To determine the intensity of the resulting image at each pixel, we are taking the lattice (grid) on which each pixel sits and using this to perform *interpolation* of the surrounding pixel intensities. `scipy` provides a handy function for this:

~~~python
def scaleit(image, factor, isseg=False):
    if isseg:
        order=0
    else:
        order=5
    return scipy.ndimage.interpolation.zoom(image, (1.0, factor, factor), order=order, mode='nearest') 
~~~


<h3 id='resample'> Resampling </h3>

It may be the case that we want to change the dimensions of our image such that they fit nicely into the input of our CNN. For example, most images and photographs have one dimension larger than the other or may be of different resolutions. This may not be the case in our training set, but most CNNs prefer to have inputs that are square and of identical sizes. We can use the same `interpolation` function from our `scaleit` function to do this:

~~~python
def resampleit(image, dims, isseg=False):
    if isseg:
        order=0
    else:
        order=3
    image = scipy.ndimage.interpolation.zoom(image, np.array(dims)/np.array(image.shape, dtype=np.float32), order=order)

    if not isseg:
        return (image-np.min(image))/(np.max(image)-np.min(image)) 
    else:
        return image
~~~

The key part here is that we've replaced the `factor` argument with `dims` of type `list`. `dims` should have length equal to the number of dimensions of our image i.e. 2 or 3. We are calculating the factor that each dimension needs to change by in order to change the image to the target `dims`.

In this step, we are also changing the intensities of the image to use the full range from `0.0` to `1.0`. This ensures that all of our image intensities fall over the same range - one fewer thing for the network to be biased against. Again, note that we don't want to do this for our segmentations as the pixel 'intensities' are actually labels.

<h3 id="rotate"> Rotation </h3>

This function utilises another `scipy` function called `rotate`. It takes a `float` for the `theta` argument which specifies the number of degrees of the roation (negative numbers rotate anti-clockwise). We want the returned image to be of the same shape as the input `image` so `reshape=False` is used. Again we need to specify the `order` of the interpolation on the new lattice. The rotate function handles 3D images by rotating each slice by the same `theta`.

~~~python
def rotateit(image, theta, isseg=False):
    order=3
    if isseg:
        order=0
        
    return rotate(image, theta, reshape=False, order=order)
    ~~~

<h3 id="intensify"> Intensity Changes </h3>

The final augmentation we can perform is a scaling in the intensity of the pixels. This effectively brightens or dims the image by appling a blanket increase or decrease across all pixels. We specify the amount by a factor: `factor < 1.0` will dim the image, and `factor > 1.0` will brighten it. Note that we don't want a `factor = 0.0` as this will blank the image.

~~~python
def intensifyit(image, factor):

    return image*factor
~~~

<h3 id="flip"> Flipping </h3>

One of the most common image augmentation procedures for natural images (dogs, cats, landscapes etc.) is to do flipping. The premise being that a dog is a dog no matter which was it's facing. Or it doesn't matter if a tree is on the right or the left of an image, it's still a tree.

We can do horizontal flipping, left-to-right or vertical flipping, up and down. It may make sense to do only one of these (if we know that dogs don't walk on their heads for example). In this case, we can specify a `list` of 2 boolean values: if each is `1` then both flips are performed. We use the `numpy` functions `fliplr` and `flipup` for these.

~~~python
def flipit(image, axes):
    
    if axes[0]:
        image = np.fliplr(image)
    if axes[1]:
        image = np.flipud(image)
    
    return image
~~~

<h3 id="application"> Application </h3>

We should be careful about how we apply our transformations. For example, if we apply multiple transformations to the same image we need to make sure that we don't apply 'resampling' after 'intensity changes' because this is reset the range of the image, defeating the point of the intensification. We also want to make sure that we are not being over zealous with the augmentations either - we need to set limits for our factors and other arguments.