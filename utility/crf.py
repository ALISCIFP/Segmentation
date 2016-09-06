"""
Author: Lengyue Chen
Date: 08/31/2016
Adapted from the inference.py to demonstate the usage of the util functions.
"""

import sys
import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image
from skimage.io import imread, imsave
import glob
imwrite = imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian








def crf(fn_im,fn_anno,fn_output):
    ##################################
    ### Read images and annotation ###
    ##################################
    img = imread(fn_im)
    print imread(fn_anno).size
    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_rgb = imread(fn_anno).astype(np.uint32)


    #anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    anno_lbl = anno_rgb

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)

    # And create a mapping back from the labels to 32bit integer colors.
    # But remove the all-0 black, that won't exist in the MAP!
    colors = colors[1:]
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16




    # Compute the number of classes in the label image.
    # We subtract one because the number shouldn't include the value 0 which stands
    # for "unknown" or "unsure".
    n_labels = len(set(labels.flat)) - 1
    print(n_labels, " labels and \"unknown\" 0: ", set(labels.flat))

    ###########################
    ### Setup the CRF model ###
    ###########################
    use_2d = False
    # use_2d = True
    if use_2d:
        print("Using 2D specialized functions")

        # Example using the DenseCRF2D code
        if n_labels==1:
            n_labels+=1
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)
    else:
        print("Using generic 2D functions")

        # Example using the DenseCRF class and the util functions
        d = dcrf.DenseCRF(img.shape[1] * img.shape[0], n_labels)

        # get unary potentials (neg log probability)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=True)
        d.setUnaryEnergy(U)

        # This creates the color-independent features and then add them to the CRF
        feats = create_pairwise_gaussian(sdims=(3, 3), shape=img.shape[:2])
        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This creates the color-dependent features and then add them to the CRF
        feats = create_pairwise_bilateral(sdims=(80, 80), schan=(13, 13, 13),
                                          img=img, chdim=2)
        d.addPairwiseEnergy(feats, compat=10,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)


    ####################################
    ### Do inference and compute MAP ###
    ####################################

    # Run five inference steps.

    Q = d.inference(5)

    # Find out the most probable class for each pixel.
    MAP = np.argmax(Q, axis=0)

    # Convert the MAP (labels) back to the corresponding colors and save the image.
    MAP = colorize[MAP,:]

    ####################################
    ###     Convert to greyscale     ###
    ####################################

    imwrite(fn_output, MAP.reshape(img.shape))
    tmp = Image.open(fn_output).convert('L')
    tmp.save(fn_output)

    # Just randomly manually run inference iterations
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)


#/home/alisc/SegNet/ADEChallengeData2016/annotations/validation

if len(sys.argv) != 4:
    print("Usage: python {} IMAGE ANNO OUTPUT".format(sys.argv[0]))
    print("")
    print("IMAGE and ANNO are inputs image folders and OUTPUT is the folder where the result should be written.")
    sys.exit(1)

fn_im_folder_path = sys.argv[1]
fn_anno_folder_path = sys.argv[2]
fn_output_folder_path = sys.argv[3]

original_images =  sorted(glob.glob(fn_im_folder_path+"/*.jpg"))
anno_images =  sorted(glob.glob(fn_anno_folder_path+"/*.png"))

for i in range(len(original_images)):
    print original_images[i]
    crf(original_images[i],anno_images[i], fn_output_folder_path +"/ADE_val_"+str(i)+".png")
    