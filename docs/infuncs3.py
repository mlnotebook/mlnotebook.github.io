import os
import csv
import sys

import numpy        as np
import SimpleITK    as sitk
import tensorflow   as tf

from sklearn.model_selection    import train_test_split
from glob                       import glob
from transforms                 import *


def get_split(dataroot, filename='data.npy', labelColumn=1, takeFactor=0.1):
    ### FUNCTION: Return balanced set of train/valid/test ids and labels
    # dataroot:     Root where datafile is stored.
    # filename:     Filename of the datafile
    # labelColumn:  Number of the column from which to take labels
    # takeFactor:   the fraction of the data to be used in validation/testing
    #               (e.g. takeFactor=0.1: training=0.8, validation=0.1, testing=0.1)
    # datafile columns: column 1    = IDs of the samples (i.e. filenames for images)
    #                   column 2+   = labels to be trained on (select only one) 
    # RETURNS [trainingIDs, trainingLabels, valIDs, valLabels, testIDs, testLabels]

    """ Load in the datafile """
    filepath = os.path.join(os.path.abspath(dataroot),filename)
    assert os.path.exists(filepath), "File does not exist: {}".format(filepath)
    if filename[-4:] == '.npy':
        filedata            = np.load(filepath)
    else:
        reader              = reader = csv.reader(open(filepath, 'rb'))
        filedata            = np.array(list(reader)[2:])

    ids                 = np.array(filedata[:,0], dtype=int)
    labels              = np.array(filedata[:,labelColumn], dtype=int)

    """ Sort the data by class into new arrays """
    newlabels           = []
    newids              = []
 
    for i, label in enumerate(np.unique(labels)):
        newlabels.append(labels[labels==label])
        newids.append(ids[labels==label])
 
    newlabels   = np.asarray(newlabels)
    newids      = np.asarray(newids)

    """ Perform one-hot encoding """
    nclass      = len(newlabels)
    for i, part in enumerate(newlabels):
        newlabels[i] = np.eye(nclass)[newlabels[i]]

    """ Shuffle the data in place. (Will only shuffle within each class) """
    seed = 42
    np.random.seed(seed)
    np.random.shuffle(newlabels)
    np.random.seed(seed)
    np.random.shuffle(newids)

    """ To ensure fully balanced classes, find class with lowest number of samples.
     Take only this number from each class"""
    hist        = np.histogram(labels, 2)
    mincounts   = np.min(hist[0])
    totake      = int(takeFactor*mincounts)

    X_test  =  np.concatenate([newids[i][:totake]               for i in range(0,len(np.unique(labels)))])
    X_val   =  np.concatenate([newids[i][totake:2*totake]       for i in range(0,len(np.unique(labels)))])
    X_train =  np.concatenate([newids[i][2*totake:mincounts]    for i in range(0,len(np.unique(labels)))])

    y_test  =  np.concatenate([newlabels[i][:totake]            for i in  range(0,len(np.unique(labels)))])
    y_val   =  np.concatenate([newlabels[i][totake:2*totake]    for i in  range(0,len(np.unique(labels)))])
    y_train =  np.concatenate([newlabels[i][2*totake:mincounts] for i in  range(0,len(np.unique(labels)))])

    return  X_train, y_train, X_val, y_val, X_test, y_test


def get_batch(ims, labels, numAugs, dims, dataroot='./', imagename='sa_ED.nii.gz', segname='label_sa_ED.nii.gz', cropem=False, maskout=False, splitseg=False):
    ### FUNCTION: takes in a list of IDs (folders), loads, processes and returns the images and labels. Primary function for returning data to a generator
    # ims:      list of IDs (folders) - folders should contain image and segmentation
    # labels:   list of labels, one for each folder in ims
    # numAugs:  0 if no data augmentation, otherwise performd numAugs augmentations of each image in ims
    # dims:     dimensions for the returned images (gets passed to the cropping function too)
    # dataroot: the folder where the image-data is stored i.e. root where each ID of ims is located
    # imagename: REQUIRED - name of the image file (same for each folder ID)
    # segname:   REQUIRED - name of the segmenatation file (same for ecah folder ID)
    # cropem:   whether to perform cropping around ROI (segmentation) or to return the full images. will be resampled back to dims after cropping
    # maskout:  whether to zero all data outside of the segmentation region in the image or to return the full image data
    # splitseg: turns an n-class segmentation of shape [h, w, d] into [h, w, d, n] where each channel n is segmentation of different class.
    # Returns:  set of processed images and labels

    assert imagename    is not None, "imagename cannot be blank: should be the name of the image file (same for each folder ID)"
    assert segname      is not None, "segname cannot be blank: should be the name of the segmentation file (same for each folder ID)"

    ifiles      = [os.path.join(dataroot,str(id_),imagename)    for id_ in ims if os.path.exists(os.path.join(dataroot,str(id_),imagename)) and os.path.exists(os.path.join(dataroot,str(id_),segname))]
    gfiles      = [os.path.join(dataroot,str(id_),segname)      for id_ in ims if os.path.exists(os.path.join(dataroot,str(id_),imagename)) and os.path.exists(os.path.join(dataroot,str(id_),segname))]

    images      = load_and_resample(ifiles, dims)
    segs        = load_and_resample(gfiles, dims, True)

    ids     = [i for i, seg in enumerate(segs) if np.sum(seg) != 0]
    images  = images[ids]
    segs    = segs[ids]

    nclass          = len(np.unique(labels, 1))

    if numAugs  !=0:
        ''' sends the loaded image files to the augmenter
        (can be any function so long as it returns arrays of images and segmetnations) '''
        images, segs    = do_aug(images, segs, numAugs)
        auglabels       = np.zeros([len(images), nclass])
        ''' if augmentations have been done, expand the labels to match '''
        for idx in range(len(labels)):
            auglabels[idx*(numAugs+1):(idx+1)*(numAugs+1),:nclass] = labels[idx]
        ''' return cropped images if cropem = True '''
        return (cropbatch(images, segs, dims, maskout, splitseg=splitseg), auglabels) if cropem else (images, auglabels)
    else:
        return (cropbatch(images, segs, dims, maskout, splitseg=splitseg), labels) if cropem else (images, labels)


def load_and_resample(ilist, dims, isseg=False):
    ### FUNCTION: Take a list of filenames, load and resample them
    # ilist:    list of filenames (images)
    # dims:     dimensions of output images
    # isseg:    if list is segmentations, change interpolation to nearest neighbour

    i_ = [resampleit(sitk.GetArrayFromImage(sitk.ReadImage(i)).transpose(1,2,0), dims, isseg) for i in ilist]

    return np.array([i for i in i_])


def get_masks(seg, onehot=True):
    ### FUNCTION:  Take segmetnation with n classes and returns each class as a separate channel
    # seg:      segmetation of shape        [h, w, d]
    # onehot:   if True, returns segmentation as series of one-hot vectors (recommended)
    # Returns:  segmentation masks of shape [h, w, d, n] where n is number of classes
    #           with one-hot, each 'pixel' is now [1 x n] vector e.g. [1,0,0,0] = background for n=4
    
    nclass      = len(np.unique(seg))
    s1, s2, s3  = seg.shape
    a           = np.array(seg.reshape([np.prod(seg.shape)]), dtype=int)
    b           = np.zeros((np.prod(seg.shape),nclass))

    ''' Make sure classes run consecutively i.e. [0,1,2,3] and not [0,1,2,4] '''
    for i, class_ in enumerate(np.unique(seg)):
        a[np.where(a==class_)]  = i

    ''' Go through all rows of b and at a in each row, turn element to 1 '''
    b[np.arange(b.shape[0]),a]  =1

    ''' Retrun to image dimensions'''
    return b.reshape([s1, s2, s3, 4])


def cropbatch(images, segs, resample_dims = [128,128,8], maskout=False, margin=10, splitseg=False):
    ### FUNCTION: Crops a batch of images using their segmentations as region of interest
    # images:           array of n images [n, h, w, d]
    # segs:             array of n segmentations (same dimetions as images)
    # resample_dims:    required dimensions of the output images
    # maskout:          returns full image data within ROI if False, else sets data outside of segmentation to zero
    # margin:           margin around the segmetation to keep in pixels
    # Returns:          array of n images [n, resample_dims]

    assert images.shape == segs.shape, "images and segmentations must be the same shape"
    x_ = np.zeros([images.shape[0]] + resample_dims)
    s_ = np.zeros([images.shape[0]] + resample_dims)
    
    ''' np.squeeze incase images are passed with channel [n, h, w, d, c] and pass the assertion test above '''
    ims = np.squeeze(images) if images.shape[0] > 1 else np.expand_dims(np.squeeze(images),0)
    ss = np.squeeze(segs) if segs.shape[0] > 1 else np.expand_dims(np.squeeze(segs),0)
    for idx, im in enumerate(ims):        
        xtmp, stmp  = cropit(im, seg=ss[idx], margin=margin)
        x_[idx,...] = resampleit(xtmp, resample_dims, False)
        s_[idx,...] = resampleit(stmp, resample_dims, True)

    ''' Perform masking if required'''
    if maskout:
    	x_[s_==0] = 0

    if splitseg:
        ss_   = np.zeros(list(s_.shape) + [1+len(np.unique(s_))])
        for idx, seg in enumerate(s_):
            ss_[idx,:,:,:,1:] = get_masks(seg)
        ss_[:,:,:,:,0] = x_
        return ss_ if images.ndim == 5 else x_
    
    return np.expand_dims(x_,-1)


def do_aug(images, segs, numAugs, squeeze=False):
    ### FUNCTION: takes in a set of image arrays and returns them along with augmentations. If numAugs = 0, just images are returned
    # images:   an array of image data [n, h, w, d]
    # segs:     an array of segmentations [n, h, w, d]
    # numAugs:  the number of augmentations per image to perform
    # squeeze:  this function auto adds a 'channel' dimension for use with Tensorflow etc. Set this true to return np.squeeze()
    # Returns:  2 arrays of [images, segmentation] with augmentations
    ''' Augmentations are: rotation, scaling (zoom), flipping, translation and intensity shifting [0, 1, 2, 3, 4] '''

    ''' placeholders for the augmented images (including a +1 channel dimension at the end for returning to TensorFlow) '''
    theseims    = np.zeros([images.shape[0]*numAugs+images.shape[0]] + list(images.shape[1:]) +[1])
    thesesegs   = np.zeros([images.shape[0]*numAugs+images.shape[0]] + list(images.shape[1:]) +[1])

    ''' set counter '''
    imageCount = 0
    for idx, originalim in enumerate(images):
        ''' set the image/seg to be augmented. Is reset to this after each augmentation '''
        theseims[imageCount*(numAugs+1), :,:,:,0]     = originalim
        thesesegs[imageCount*(numAugs+1), :,:,:,0]    = segs[idx]
        
        if numAugs > 0:
            ''' set counter to ensure correct number of augmentations. Some seg augs may fail, so do while until correct number is reached '''
            augCount = 0           
            while augCount < numAugs:
                ''' determine where to put this augmentation in the array. Reset images '''
                augID = (imageCount * (numAugs+1) ) + augCount + 1
                thisim      = originalim
                thisseg     = segs[idx]

                ''' set the default factors for the augmentations '''
                theta   = 0.0
                factor  = 1.0
                offset  = 0.0
                axes    = [0, 0] # for flipping
                scale   = 1.0
                
                ''' determine how many augmentations to perform and which ones '''
                numTrans        = np.random.randint(2, 6, size=1)        
                allowedTrans    = [0, 1, 2, 3, 4, 5]
                whichTrans      = np.random.choice(allowedTrans, numTrans, replace=False)

                '''Rotation between [-theta and theta] '''
                if 0 in whichTrans:
                    theta   = float(np.around(np.random.uniform(-15.0,15.0, size=1), 2))
                    thisim  = rotateit(thisim, theta)
                    thisseg = rotateit(thisseg, theta, isseg=True)
                
                ''' Scaling (zooming) between [-scale and scale] '''
                if 1 in whichTrans:
                    scale  = float(np.around(np.random.uniform(0.85, 1.15, size=1), 2))
                    thisim  = scaleit(thisim, scale)
                    thisseg = scaleit(thisseg, scale, isseg=True)
                
                ''' Flipping along axes '''
                if 2 in whichTrans:
                    axes    = list(np.random.choice([0,1], 1, replace=True))
                    thisim  = flipit(thisim, axes+[0])
                    thisseg = flipit(thisseg, axes+[0])
                
                ''' Translation in x and y planes only '''
                if 3 in whichTrans:
                    offset  = list(np.random.randint(-8,8, size=2))
                    #currseg = thisseg
                    thisim  = translateit(thisim, offset)
                    thisseg = translateit(thisseg, offset, isseg=True)
                
                ''' Check if the segmetnation has been successfully augmented. If failed (all zeros) discard this augmentation and try again '''
                if int(thisseg.sum())==0:
                    continue                            
                
                ''' resample the augmentation back to the correct dimensions. May increase with rotation etc. '''
                thisim  = resampleit(thisim, list(images.shape[1:]))
                thisseg = resampleit(thisseg, list(images.shape[1:]), isseg=True)
                
                ''' Check if the segmetnation has been successfully augmented. If failed (all zeros) discard this augmentation and try again '''
                if int(thisseg.sum())== 0:
                    continue

                ''' Perform random translation for some of the slices in the 3D image (simulates motion)'''
                if 5 in whichTrans:
                    thisim   = sliceshift(thisim, shift_min = -3, shift_max = 3, fraction=0.5)
                    thisseg  = sliceshift(thisseg, shift_min = -3, shift_max = 3, fraction=0.5, isseg=True)

                ''' Intensity shifts: has to be done after the resampling or else normalisation would cancel this out '''
                ''' Note: no intensity change on the segmentation '''
                if 4 in whichTrans:
                    factor  = float(np.around(np.random.uniform(0.8, 1.2, size=1), 2))
                    thisim  = intensifyit(thisim, factor)

                theseims[augID, :,:,:,0]    = thisim
                thesesegs[augID, :,:,:,0]   = thisseg
                
                ''' Successful augmentation, so increase counter and continue '''
                augCount += 1
        ''' After each image has been augmentation numAug times, go to the next image'''       
        imageCount+=1 

    return (np.squeeze(theseims), np.squeeze(thesesegs)) if squeeze else (theseims, thesesegs)
    
    
    
    
    
    
    
    
    
    