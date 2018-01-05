import SimpleITK as sitk
import numpy as np
import sys
import os

from transforms import *

def plotit(images):
    f, axes = plt.subplots(len(images)/2, 2)
    for axis, image in zip(axes.ravel(), images):
        if image.shape[-1] == 3:
            axis.imshow(image[:,:, :]) 
        else:
            axis.imshow(image[:,:, image.shape[-1]//2], cmap='gray')
    plt.show()
    
imfile = sys.argv[1]

withseg = False

try:
    thisim      = sitk.GetArrayFromImage(sitk.ReadImage(imfile))
    thisim = thisim.transpose([1,2,0]) if thisim.shape[-1] > 3 else thisim
    originalim  = thisim
except:
    sys.stdout.write("File does not exist: {}\n".format(imfile))
    sys.stdout.flush()

dims = [224,224,8]

if len(sys.argv) > 2:
    segfile = sys.argv[2]
    withseg = True
    try:
        thisseg     = sitk.GetArrayFromImage(sitk.ReadImage(segfile)).transpose([1,2,0])
        originalseg = thisseg
    except:
        sys.stdout.write("File does not exist: {}\n".format(segfile))
        sys.stdout.flush()     


np.random.seed()
numTrans     = np.random.randint(1, 6, size=1)        
allowedTrans = [0, 1, 2, 3, 4]
whichTrans   = np.random.choice(allowedTrans, numTrans, replace=False)

theta = 0.0
scalefactor = 1.0
factor = 1.0
offset = [0.0, 0.0]

if 0 in whichTrans:
    theta   = float(np.around(np.random.uniform(-10.0,10.0, size=1), 2))
    theta =-10.0
    thisim  = rotateit(thisim, theta)
    thisseg = rotateit(thisseg, theta, isseg=True) if withseg else np.zeros_like(thisim)

if 1 in whichTrans:
    scalefactor  = float(np.around(np.random.uniform(0.9, 1.1, size=1), 2))
    thisim  = scaleit(thisim, scalefactor)
    thisseg = scaleit(thisseg, scalefactor, isseg=True) if withseg else np.zeros_like(thisim)

if 2 in whichTrans:
    factor  = float(np.around(np.random.uniform(0.8, 1.2, size=1), 2))
    thisim  = intensifyit(thisim, factor)
    #no intensity change on segmentation

if 3 in whichTrans:
    axes    = list(np.random.choice(2, 1, replace=True))
    thisim  = flipit(thisim, axes+[0])
    thisseg = flipit(thisseg, axes+[0]) if withseg else np.zeros_like(thisim)

if 4 in whichTrans:
    offset  = list(np.random.randint(-5,5, size=2))
    thisim  = translateit(thisim, offset)
    thisseg = translateit(thisseg, offset, isseg=True) if withseg else np.zeros_like(thisim)

if withseg:
    thisim, thisseg = cropit(thisim, thisseg)
    thisseg         = resampleit(thisseg, dims, isseg=True) if withseg else np.zeros_like(thisim)
else:
    thisim = cropit(thisim)

thisim          = resampleit(thisim, dims)

nameTrans = ['Rotation', 'Scaling', 'Intensity', 'Flipping', 'Translation']
params = [theta, scalefactor, factor, 'lr', offset]
sys.stdout.write("Transformations\n{}\n".format([[nameTrans[x], params[x]] for x in whichTrans]))
sys.stdout.flush()

if withseg:
    plotit([originalim, originalseg, thisim, thisseg])
else:
    plotit([originalim, thisim])

### Uncomment this section if you wish to save the image/segmentation ###
# suff='crop'
# dirname = '/home/'
# import scipy

# if dims[-1] == 3:
#     sitk.WriteImage(sitk.GetImageFromArray(thisim, isVector=True), '/home/rdr16/mln/static/img/augmentation/{}{}.png'.format(os.path.basename(imfile)[:-4],suff))
# else:
#     scipy.misc.imsave('{}/{}{}.png'.format(dirname, os.path.basename(imfile)[:-7],suff), thisim[:,:,4])
#     scipy.misc.imsave('{}/{}{}.png'.format(dirname, os.path.basename(segfile)[:-7],suff), thisseg[:,:,4])
