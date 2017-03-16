+++
date = "2017-03-01T19:27:27Z"
title = "Surface Distance Function"
description = "Surface distance evaluation function in Python"
topics = ["python"]
tags = ["python", "segmentation","evaluation","surface distance"]
social=true
featured_image="/img/segs.png"
+++

Surface Distance measures are a good way of evaluating the accuracy of an image-segmentation if we already know the ground truth (GT). The problem is that there is no nicely packaged function in Python to do this directly. In this post, we'll write a surface distance function in Python which uses numpy and scipy. It'll help us to calculate Mean Surface Distance (MSD), Residual Mean-Square Error (RMS) and the Hausdorff Distance (HD).
<!--more-->

## Background
Recently, I have been doing a **lot** of segmentation evaluation - seeing how good a segmentation done by a machine compares with one that's done manual, a 'ground truth' (GT). Traditionally, such verification is done by comparing the overlap between the two e.g. Dice Simlarity Coefficient (DSC) [1]. There are a few different calculations that can be done (there'll be a longer post on just that) and 'surface distance' calculations are one of them.

## Method
For this calculation, we need to be able to find the outline of the segmentation and compare it to the outline of the GT. We can then take measurements of how far each segmentation pixel is from its corresponding pixel in the GT outline.

Let's take a look at the maths. Surface distance metrics estimate the error between the outer surfaces $S$ and $S^{\prime}$ of the segmentations $X$ and $X^{\prime}$. The distance between a point $p$ on surface $S$ and the surface $S^{\prime}$ is given by the minimum of the Euclidean norm:

<div>$$ d(p, S^{\prime}) = \min_{p^{\prime} \in S^{\prime}} \left|\left| p - p^{\prime} \right|\right|_{2} $$</div>

Doing this for all pixels in the surface gives the total surface distance between $S$ and $S^{\prime}$: $d(S, S^{\prime})$:


Now I've seen MATLAB code that can do this, though often its not entirely accurate. Plus I wanted to do this calculation on-the-fly as part of my program which was written in Python. So I came up with this function:

<pre><code class="python"
>import numpy as np
from scipy.ndimage import morphology

def surfd(input1, input2, sampling=1, connectivity=1):
    
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
    

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)

    
    dta = morphology.distance_transform_edt(~S,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
       
    
    return sds
</code></pre>

Lets go through it bit-by-bit. The function _surfd_ is defined to take in four variables:

* _input1_ - the segmentation that has been created. It can be a multi-class segmentation, but this function will make the image binary. We'll talk about how to use this function on individual classes later.
* _input2_ - the GT segmentation against which we wish to compare _input1_
* _sampling_ - the pixel resolution or pixel size. This is entered as an _n_-vector where _n_ is equal to the number of dimensions in the segmentation i.e. 2D or 3D. The default value is 1 which means pixels (or rather voxels) are 1 x 1 x 1 mm in size.
* _connectivity_ - creates either a 2D (3 x 3) or 3D (3 x 3 x 3) matrix defining the neighbourhood around which the function looks for neighbouring pixels. Typically, this is defined as a six-neighbour kernel which is the default behaviour of this function.

First we'll be making use of simple numpy operations, but we'll also need the _morphology_ module from _scipy_'s _dnimage_ package. These are imported first. More information on this module can be found [here](https://docs.scipy.org/doc/scipy-0.18.1/reference/ndimage.html "Scipy _ndimage_ package")

<pre><code class="python"
>import numpy as np
from scipy.ndimage import morphology
</code></pre>

The two inputs are checked for their size and made binary. Any value greater than zero is made 1 (true).

<pre><code class="python"
>    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
</code></pre>

We use the the _morphology.generate\_binary\_structure_ function, along with the number of dimensions of the segmentation, to create the kernel that will be used to detect the edges of the segmentations. This could be done just by hard-coding the kernel itself: `[[0 0 0],[0 1 0],[0 0 0]; [0 1 0], [1 1 1], [0 1 0]; [0 0 0], [0 1 0], [0 0 0]]`. This kernel '_conn_' is supplied to the _morphology.binary\_erosion_ function which strips the outermost pixel from the edge of the segmentation. Subtracting this result from the segmentation itself leaves only the single-pixel-wide surface.

<pre><code class="python"
>    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    S = input_1 - morphology.binary_erosion(input_1, conn)
    Sprime = input_2 - morphology.binary_erosion(input_2, conn)
</code></pre>

Next we again use the _morphology_ module. This time we give the _distance\_transform\_edt_ function our pixel-size (_samping_) and also the inverted surface-image. The inversion is used such that the surface itself is given the value of zero i.e. any pixel at this location, will have zero surface-distance. The transform increases the value/error/penalty of the remaining pixels with increasing distance away from the surface.

Each pixel of the opposite segmentation-surface is then laid upon this 'map' of penalties and both results are concatenated into a vector which is as long as the number of pixels in the surface of each segmentation. This vector of _surface distances_ is returned. Note that this is technically the _symmetric_ surface distance as we are not assuming that just doing this for _one_ of the surfaces is enough. It may be that the distance between a pixel in A and in B is not the same as between the pixel in B and in A. i.e. $d(S, S^{\prime}) \neq d(S^{\prime}, S)$

<pre><code class="python"
>    dta = morphology.distance_transform_edt(~input1_border,sampling)
    dtb = morphology.distance_transform_edt(~Sprime,sampling)
    
    sds = np.concatenate([np.ravel(dta[Sprime!=0]), np.ravel(dtb[S!=0])])
        
    return sds
</code></pre>

## How is it used?
The function example below takes two segmentations (which both have multiple classes). The sampling vector is a typical pixel-size from an MRI scan and the 1 indicated I'd like a 6 neighbour (cross-shaped) kernel for finding the edges.

<pre><code class="python"
>    surface_distance = surfd(test_seg, GT_seg, [1.25, 1.25, 10],1)
</code></pre>

By specifcing the value of the voxel-label I'm interested in (assuming we're talking about classes which are contiguous and not spread out), we can find the surface accuracy of that class.

<pre><code class="python"
>    surface_distance = surfd(test_seg(test_seg==1), \
    		       GT_seg(GT_seg==1), [1.25, 1.25, 10],1)
</code></pre>

## What do the results mean?
The returned surface distances can be used to calculate:

* _Mean Surface Distance (MSD)_ - the mean of the vector is taken. This tell us how much, on average, the surface varies between the segmentation and the GT (in mm).  

<div>$$ 	\text{MSD} = \frac{1}{n_{S} + n_{S^{\prime}}} \left( \sum_{p = 1}^{n_{S}} d(p, S^{\prime}) + \sum_{p^{\prime}=1}^{n_{S^{\prime}}} d(p^{\prime}, S) \right) $$ </div>

* _Residual Mean Square Distance (RMS)_ - as it says, the mean is taken from each of the points in the vector, these residuals are squared (to remove negative signs), summated, weighted by the mean and then the square-root is taken. Measured in mm.  

<div>$$ \text{RMS} = \sqrt{\frac{1}{n_{S} + n_{S^{\prime}}} \left( \sum_{p = 1}^{n_{S}} d(p, S^{\prime})^{2} + \sum_{p^{\prime}=1}^{n_{S^{\prime}}} d(p^{\prime}, S)^{2} \right) }\ $$</div>

* _Hausdorff Distance (HD)_ - the maximum of the vector. The largest difference between the surface distances. Also measured in mm. We calculate the _symmetric Hausdorff distance_ as:  

<div>$$\text{HD} = \max \left[ d(S, S^{\prime}) , d(S^{\prime}, S) \right]$$</div>

Or in Python:
<pre><code class="python"
>    msd = surface_distance.mean()
    rms = np.sqrt((surface_distance**2).mean())
    hd  = surface_distance.max()
</code></pre>

---
### References

[1] 	Dice, L. R. (1945). Measures of the Amount of Ecologic Association Between Species. Ecology, 26(3), 297–302. https://doi.org/10.2307/1932409