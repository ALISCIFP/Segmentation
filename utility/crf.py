"""
Author: Lengyue Chen
Date: 09/17/2016
"""

import sys
import numpy as np
import pydensecrf.densecrf as dcrf
from PIL import Image
from skimage.io import imread, imsave
from skimage import color
import glob
imwrite = imsave
from pydensecrf.utils import unary_from_labels, create_pairwise_bilateral, create_pairwise_gaussian

import cv2



def crf(fn_im,fn_anno,fn_output,colorful_fn_output):
    ##################################
    ### Read images and annotation ###
    ##################################
    img = imread(fn_im)
    #truth_img = imread(truth_image).astype(np.uint8)
    # Convert the annotation's RGB color to a single 32-bit integer color 0xBBGGRR
    anno_rgb = imread(fn_anno)


    #anno_lbl = anno_rgb[:,:,0] + (anno_rgb[:,:,1] << 8) + (anno_rgb[:,:,2] << 16)
    anno_lbl = anno_rgb

    # Convert the 32bit integer color to 1, 2, ... labels.
    # Note that all-black, i.e. the value 0 for background will stay 0.
    colors, labels = np.unique(anno_lbl, return_inverse=True)


    # And create a mapping back from the labels to 32bit integer colors.
    # But remove the all-0 black, that won't exist in the MAP!
    colors = colors[1:]

    colorize = np.empty((len(colors), 1), np.uint8)

    colorize[:,0] = (colors & 0x0000FF)
    #colorize[:,1] = (colors & 0x00FF00) >> 8
    #colorize[:,2] = (colors & 0xFF0000) >> 16




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


    crf_img = MAP.reshape(anno_lbl.shape)
    ########change to rgb########
    
    label = anno_lbl
    
    
    ind = crf_img
    
    
    r = ind.copy()
    g = ind.copy()
    b = ind.copy()

    r_gt = label.copy()
    g_gt = label.copy()
    b_gt = label.copy()


    wall = [139,181,248]
    building = [251,209,244]
    sky = [44,230,121]
    floor = [156,40,149]
    tree = [166,219,98]
    ceiling = [35,229,138]
    road = [143,56,194]
    bed  = [144,223,70]
    windowpane = [200,162,57]
    grass = [120,225,199]
    cabinet = [87,203,13]
    sidewalk = [185,1,136]
    person = [16,167,16]
    earth = [29,249,241]
    door = [17,192,40]
    table = [199,44,241]
    mountain = [193,196,159]
    plant = [241,172,78]
    curtain = [56,94,128]
    chair = [231,166,116]
    car = [50,209,252]
    water = [217,56,227]
    painting = [168,198,178]
    sofa = [77,179,188]
    shelf = [236,191,103]
    house = [248,138,151]
    sea = [214,251,89]
    mirror = [208,204,187]
    rug = [115,104,49]
    field = [29,202,113]
    armchair = [159,160,95]
    seat = [78,188,13]
    fence = [83,203,82]
    desk = [8,234,116]
    rock = [80,159,200]
    wardrobe = [124,194,2]
    lamp = [192,146,237]
    bathtub = [64,3,73]
    railing = [17,213,58]
    cushion = [106,54,105]
    base = [125,72,155]
    box = [202,36,231]
    column = [79,144,4]
    signboard = [118,185,128]
    chest = [138,61,178]
    counter = [23,182,182]
    sand = [154,114,4]
    sink = [201,0,83]
    skyscraper = [21,134,53]
    fireplace = [194,77,237]
    refrigerator = [198,81,106]
    grandstand = [37,222,181]
    path = [203,185,14]
    stairs = [134,140,113]
    runway = [220,196,79]
    case = [64,26,68]
    pooltable = [128,89,2]
    pillow = [199,228,65]
    screen = [62,215,111]
    stairway = [124,148,166]
    river = [221,119,245]
    bridge = [68,57,158]
    bookcase = [80,47,26]
    blind = [143,59,56]
    coffeetable = [14,80,215]
    toilet = [212,132,31]
    flower = [2,234,129]
    book = [134,179,44]
    hill = [53,21,129]
    bench = [80,176,236]
    countertop = [154,39,168]
    stove = [221,44,139]
    palm = [103,56,185]
    kitchenisland = [224,138,83]
    computer = [243,93,235]
    swivelchair = [80,158,63]
    boat = [81,229,38]
    bar = [116,215,38]
    arcademachine = [103,69,182]
    hovel = [66,81,5]
    bus = [96,157,229]
    towel = [164,49,170]
    light = [14,42,146]
    truck = [164,67,44]
    tower = [108,116,151]
    chandelier = [144,8,144]
    awning = [85,68,228]
    streetlight = [16,236,72]
    booth = [108,7,86]
    television = [172,27,94]
    airplane = [119,247,193]
    dirttrack = [155,240,152]
    apparel = [49,158,204]
    pole = [23,193,204]
    land = [228,66,107]
    bannister = [69,36,163]
    escalator = [238,158,228]
    ottoman = [202,226,35]
    bottle = [194,243,151]
    buffet = [192,56,76]
    poster = [16,115,240]
    stage = [61,190,185]
    van = [7,134,32]
    ship = [192,87,171]
    fountain = [45,11,254]
    conveyerbelt = [179,183,31]
    canopy = [181,175,146]
    washer = [13,187,133]
    plaything = [12,1,2]
    swimmingpool = [63,199,190]
    stool = [221,248,32]
    barrel = [183,221,51]
    basket = [90,111,162]
    waterfall = [82,0,6]
    tent = [40,0,239]
    bag = [252,81,54]
    minibike = [110,245,152]
    cradle = [0,187,93]
    oven = [163,154,153]
    ball = [134,66,99]
    food = [123,150,242]
    step = [38,144,137]
    tank = [59,180,230]
    tradename = [144,212,16]
    microwave = [132,125,200]
    pot = [26,3,35]
    animal = [199,56,92]
    bicycle = [83,223,224]
    lake = [203,47,137]
    dishwasher = [74,74,251]
    screen = [246,81,197]
    blanket = [168,130,178]
    sculpture = [136,85,200]
    hood = [186,147,103]
    sconce = [170,21,85]
    vase = [104,52,182]
    trafficlight = [166,147,202]
    tray = [103,119,71]
    ashcan = [74,161,165]
    fan = [14,9,83]
    pier = [129,194,43]
    crtscreen = [7,100,55]
    plate = [13,12,170]
    monitor = [30,21,22]
    bulletinboard = [224,189,139]
    shower = [40,77,25]
    radiator = [194,14,94]
    glass = [178,8,231]
    clock = [234,166,8]
    flag = [248,25,7]   
    unlabelled = [0,0,0]

    label_colours = np.array([wall,building,sky,floor,tree,ceiling,road,bed ,windowpane,grass,cabinet,sidewalk,person,earth,door,table,mountain,plant,curtain,chair,car,water,painting,sofa,shelf,house,sea,mirror,rug,field,armchair,seat,fence,desk,rock,wardrobe,lamp,bathtub,railing,cushion,base,box,column,signboard,chest,counter,sand,sink,skyscraper,fireplace,refrigerator,grandstand,path,stairs,runway,case,pooltable,pillow,screen,stairway,river,bridge,bookcase,blind,coffeetable,toilet,flower,book,hill,bench,countertop,stove,palm,kitchenisland,computer,swivelchair,boat,bar,arcademachine,hovel,bus,towel,light,truck,tower,chandelier,awning,streetlight,booth,television,airplane,dirttrack,apparel,pole,land,bannister,escalator,ottoman,bottle,buffet,poster,stage,van,ship,fountain,conveyerbelt,canopy,washer,plaything,swimmingpool,stool,barrel,basket,waterfall,tent,bag,minibike,cradle,oven,ball,food,step,tank,tradename,microwave,pot,animal,bicycle,lake,dishwasher,screen,blanket,sculpture,hood,sconce,vase,trafficlight,tray,ashcan,fan,pier,crtscreen,plate,monitor,bulletinboard,shower,radiator,glass,clock,flag,unlabelled])
    
    for l in range(0,150):
        r[ind==l] = label_colours[l,0]
        g[ind==l] = label_colours[l,1]
        b[ind==l] = label_colours[l,2]
        r_gt[label==l] = label_colours[l,0]
        g_gt[label==l] = label_colours[l,1]
        b_gt[label==l] = label_colours[l,2]
        # r_truth[truth_img==l] = label_colours[l,0]
        # g_truth[truth_img==l] = label_colours[l,1]
        # b_truth[truth_img==l] = label_colours[l,2]


    rgb = np.zeros((ind.shape[0], ind.shape[1], 3))
    rgb[:,:,0] = r
    rgb[:,:,1] = g
    rgb[:,:,2] = b
    rgb_gt = np.zeros((label.shape[0], label.shape[1], 3))

    rgb_gt[:,:,0] = r_gt
    rgb_gt[:,:,1] = g_gt
    rgb_gt[:,:,2] = b_gt

    
    cv2.imwrite(colorful_fn_output,rgb)






    ########change to rgb########
    imsave(fn_output,crf_img)
    # Just randomly manually run inference iterations
    Q, tmp1, tmp2 = d.startInference()
    for i in range(5):
        print("KL-divergence at {}: {}".format(i, d.klDivergence(Q)))
        d.stepInference(Q, tmp1, tmp2)





fn_im_folder_path = sys.argv[1]
fn_anno_folder_path = sys.argv[2]
fn_output_folder_path = sys.argv[3]
colorful_fn_output_folder_path = sys.argv[4]

original_images =  sorted(glob.glob(fn_im_folder_path+"/*.jpg"))
anno_images =  sorted(glob.glob(fn_anno_folder_path+"/*.png"))


for i in range(len(original_images)):
    print original_images[i]
    crf(original_images[i],anno_images[i], fn_output_folder_path +"/ADE_val_0000"+str('%04d'%(i+1))+".png", colorful_fn_output_folder_path +"/ADE_val_0000"+str('%04d'%(i+1))+".png")
    