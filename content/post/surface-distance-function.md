+++
date = "2017-03-01T19:27:27Z"
title = "Surface Distance Function"
description = "Performing segmentation evaluation with surface distances is a good way of measuring the overall accuracy of the segmentation. This function is written in Python to help with just that!"
topics = ["segmentation"]
tags = ["python", "segmentation","evaluation","surface distance"]

+++

## Background
Recently, I have been doing a **lot** of segmentation evaluation - seeing how good a segmentation done by a machine compares with one that's done manual, a 'ground truth' (GT). Traditionally, such verification is done by comparing the overlap between the two e.g. Dice Simlarity Coefficient (DSC) [1]. There are a few different calculations that can be done (there'll be a longer post on just that) and 'surface distance' calculations are one of them.

## Method
For this calculation, we need to be able to find the outline of the segmentation and compare it to the outline of the GT. We can then take measurements of how far each segmentation pixel is from its corresponding pixel in the GT outline.

Now I've seen MATLAB code that can do this, though often its not entirely accurate. Plus I wanted to do this calculation on-the-fly as part of my program which was written in Python. So I came up with this function:

{{< highlight python >}}
import numpy as np
from scipy.ndimage import morphology

def surfd(input1, input2, sampling=1, connectivity=1):
    
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
    

    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    input1_border = input_1 - morphology.binary_erosion(input_1, conn)
    input2_border = input_2 - morphology.binary_erosion(input_2, conn)

    
    dta = morphology.distance_transform_edt(~input1_border,sampling)
    dtb = morphology.distance_transform_edt(~input2_border,sampling)
    
    sds = np.concatenate([np.ravel(dta[input2_border!=0]), \
    	  np.ravel(dtb[input1_border!=0])])
       
    
    return sds
{{< /highlight >}}

Lets go through it bit-by-bit. The function _surfd_ is defined to take in four variables:

* _input1_ - the segmentation that has been created. It can be a multi-class segmentation, but this function will make the image binary. We'll talk about how to use this function on individual classes later.
* _input2_ - the GT segmentation against which we wish to compare _input1_
* _sampling_ - the pixel resolution or pixel size. This is entered as an _n_-vector where _n_ is equal to the number of dimensions in the segmentation i.e. 2D or 3D. The default value is 1 which means pixels (or rather voxels) are 1 x 1 x 1 mm in size.
* _connectivity_ - creates either a 2D (3 x 3) or 3D (3 x 3 x 3) matrix defining the neighbourhood around which the function looks for neighbouring pixels. Typically, this is defined as a six-neighbour kernel which is the default behaviour of this function.

First we'll be making use of simple numpy operations, but we'll also need the _morphology_ module from _scipy_'s _dnimage_ package. These are imported first. More information on this module can be found [here](https://docs.scipy.org/doc/scipy-0.18.1/reference/ndimage.html "Scipy _ndimage_ package")

{{< highlight python >}}
import numpy as np
from scipy.ndimage import morphology
{{< /highlight >}}

The two inputs are checked for their size and made binary. Any value greater than zero is made 1 (true).

{{< highlight python >}}
    input_1 = np.atleast_1d(input1.astype(np.bool))
    input_2 = np.atleast_1d(input2.astype(np.bool))
{{< /highlight >}}

We use the the _morphology.generate\_binary\_structure_ function, along with the number of dimensions of the segmentation, to create the kernel that will be used to detect the edges of the segmentations. This could be done just by hard-coding the kernel itself: `[[0 0 0],[0 1 0],[0 0 0]; [0 1 0], [1 1 1], [0 1 0]; [0 0 0], [0 1 0], [0 0 0]]`. This kernel '_conn_' is supplied to the _morphology.binary\_erosion_ function which strips the outermost pixel from the edge of the segmentation. Subtracting this result from the segmentation itself leaves only the single-pixel-wide surface.

{{< highlight python >}}
    conn = morphology.generate_binary_structure(input_1.ndim, connectivity)

    input1_border = input_1 - morphology.binary_erosion(input_1, conn)
    input2_border = input_2 - morphology.binary_erosion(input_2, conn)
{{< /highlight >}}

Next we again use the _morphology_ module. This time we give the _distance\_transform\_edt_ function our pixel-size (_samping_) and also the inverted surface-image. The inversion is used such that the surface itself is given the value of zero i.e. any pixel at this location, will have zero surface-distance. The transform increases the value/error/penalty of the remaining pixels with increasing distance away from the surface.

Each pixel of the opposite segmentation-surface is then laid upon this 'map' of penalties and both results are concatenated into a vector which is as long as the number of pixels in the surface of each segmentation. This vector of _surface distances_ is returned. Note that this is technically the _symmetric_ surface distance as we are not assuming that just doing this for _one_ of the surfaces is enough. It may be that the distance between a pixel in A and in B is not the same as between the pixel in B and in A.

{{< highlight python >}}
    dta = morphology.distance_transform_edt(~input1_border,sampling)
    dtb = morphology.distance_transform_edt(~input2_border,sampling)
    
    sds = np.concatenate([np.ravel(dta[input2_border!=0]), \
    	  np.ravel(dtb[input1_border!=0])])
        
    return sds
{{< /highlight >}}

## How is it used?
The function example below takes two segmentations (which both have multiple classes). The sampling vector is a typical pixel-size from an MRI scan and the 1 indicated I'd like a 6 neighbour (cross-shaped) kernel for finding the edges.

{{< highlight python >}}
    surface_distance = surfd(test_seg, GT_seg, [1.25, 1.25, 10],1)
{{< /highlight >}}

By specifcing the value of the voxel-label I'm interested in (assuming we're talking about classes which are contiguous and not spread out), we can find the surface accuracy of that class.

{{< highlight python >}}
    surface_distance = surfd(test_seg(test_seg==1), \
    		       GT_seg(GT_seg==1), [1.25, 1.25, 10],1)
{{< /highlight >}}

## What do the results mean?
The returned surface distances can be used to calculate:

* _Mean Surface Distance (MSD)_ - the mean of the vector is taken. This tell us how much, on average, the surface varies between the segmentation and the GT (in mm).
* _Residual Mean Square Distance (RMS)_ - as it says, the mean is taken from each of the points in the vector, these residuals are squared (to remove negative signs), summated, weighted by the mean and then the square-root is taken. Measured in mm.
* _Hausdorff Distance (HD)_ - the maximum of the vector. The largest difference between the surface distances. Also measured in mm.

{{< highlight python >}}
    msd = surface_distance.mean()
    rms = np.sqrt((surface_distance**2).mean())
    hd  = (surface_distance.max())
{{< /highlight >}}

---
### References

[1] 	Dice, L. R. (1945). Measures of the Amount of Ecologic Association Between Species. Ecology, 26(3), 297–302. https://doi.org/10.2307/1932409