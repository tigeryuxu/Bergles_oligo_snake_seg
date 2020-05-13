# -*- coding: utf-8 -*-
"""
Created on February 15th, 2020
============================================================

Runs neural network for training snake-seg based reconstruction of oligodendrocytes

- updated UNet3D to accomodate for 80 x 80 images

@author: Tiger
"""



""" ALLOWS print out of results on compute canada """
#from keras import backend as K
#cfg = K.tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
#K.set_session(K.tf.Session(config=cfg))

import matplotlib
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 
matplotlib.use('Agg')
import matplotlib.pyplot as plt


""" Libraries to load """
import tensorflow as tf
import cv2 as cv2
from matplotlib import *
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from natsort import natsort_keygen, ns
#from skimage import measure
import pickle as pickle
import os

from random import randint

from plot_functions import *
from data_functions import *
from data_functions_3D import *
#from post_process_functions import *
from UNet import *
from UNet_3D import *
import glob, os
#from tifffile import imsave

#from UnetTrainingFxn_0v6_actualFunctions import *
from random import randint


# Initialize everything with specific random seeds for repeatability
#input_path = 'E:/7) Bergles lab data/Traces files/TRAINING FORWARD PROP SCALED FINISHED/'
input_path = '/scratch/yxu233/TRAINING FORWARD PROP ONLY SCALED FINAL/'
               
""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))

natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)
""" FOR CLAHE """
examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop','_DILATE_truth_crop'), seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop')) for i in images]

input_size = 80
num_truth_class = 1 + 1 # ALWAYS MINUS 1 for the background
multiclass = 0
if num_truth_class > 2:
     multiclass = 1
     
depth = 16


for i in range(len(examples)):
        """ Load input image """
        input_name = examples[i]['input']
        input_im = open_image_sequence_to_3D(input_name, width_max=input_size, height_max=input_size, depth=depth)

        
        """ Combine image + cell nucleus mask """
        seed_crop_name = examples[i]['seed_crop']
        seed_crop_mask = open_image_sequence_to_3D(seed_crop_name, width_max=input_size, height_max=input_size, depth=depth)
        
    
        """ Load truth image, either BINARY or MULTICLASS """
        truth_name = examples[i]['truth']   
        truth_im, weighted_labels = load_class_truth_3D(truth_name, num_truth_class, width_max=input_size, height_max=input_size,
                                                        depth=depth, spatial_weight_bool=0, delete_seed=seed_crop_mask, skip_class=2, resized_check=1)

      
          
        """ Check to see if truth includes splits, if so allow 80% of the time
        
             ==> since compute canada can't do skeletonize (skimage functions), maybe just...
                         more than 3 means branchpoint
                 == 2 means skeleton normal point
                 == 1 means endpoint
        """
        test_split = np.copy(truth_im[:, :, :, 1])
        
        degrees, coordinates = bw_skel_and_analyze(test_split)
        

        # if rand_num <= 2 and np.count_nonzero(degrees == 3) == 0:     # 20% of the time want one with NO splits
        #      rand_num = randint(0, 10)
            
               
        # elif rand_num > 2 and np.count_nonzero(degrees == 3) > 0:
        #      rand_num = randint(0, 10)
                          
        # else:
        #      continue;   # can skip altogether!

        save_name = input_name.split('.tif')[0]
             
        #input_im = convert_matrix_to_multipage_tiff(input_im)     
        """ Size based selection """
        if np.count_nonzero(degrees == 3) == 0:     # 20% of the time want one with NO splits
             #imsave(save_name + '_LARGE.tif', np.asarray(input_im, dtype=np.uint8))
             print('no branching')
             
        elif np.count_nonzero(degrees == 3) > 0:
             
             imsave(save_name + '_BRANCHED.tif', np.asarray(input_im, dtype=np.uint8)) 
             


        #save_name = input_name.split('.tif')[0]
             
        #input_im = convert_matrix_to_multipage_tiff(input_im)     
        #""" Size based selection """
        #if np.count_nonzero(test_split) > 500:     # 20% of the time want one with NO splits
        #     imsave(save_name + '_LARGE.tif', np.asarray(input_im, dtype=np.uint8))
             
        #elif np.count_nonzero(test_split) <= 500:
             
        #     imsave(save_name + '_SMALL.tif', np.asarray(input_im, dtype=np.uint8)) 


