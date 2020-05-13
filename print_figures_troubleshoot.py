# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???
 
 
 Try reducing network size
 

@author: Tiger
"""

""" Libraries to load """


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



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

from data_functions import *
from data_functions_3D import *
import glob, os
#from tifffile import imsave


# Initialize everything with specific random seeds for repeatability

"""  Network Begins:
"""
#s_path = './Checkpoints/'
s_path = './'
input_path = '/scratch/yxu233/Training_data_Bergles_lab_512x512/'

#val_path = '../Validation/'
val_path = 0

""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*input_512x512_RESIZED.tif'))
examples = [dict(input=i,truth=i.replace('input_512x512_RESIZED','truth'), cell_mask=i.replace('input_512x512_RESIZED','input_cellMASK_512x512_RESIZED')) for i in images]
#examples = [dict(input=i,truth=i.replace('input.tif','_pos_truth.tif')) for i in images]  # if want to load only positive examples

counter = list(range(len(examples)))  # create a counter, so can randomize it
counter = np.array(counter)

input_counter = counter

""" Saving the objects """
save_pkl(examples, s_path, 'examples.pkl')
examples
save_pkl(input_counter, s_path, 'input_counter.pkl')


input_size = 512
depth = 96
num_truth_class = 2 + 1

for i in range(len(input_counter)):
        
        # degrees rotated:
        rand = randint(0, 360)      
        """ Load input image """
        input_name = examples[input_counter[i]]['input']
        #print(input_name)
        #input_im = np.asarray(Image.open(input_name), dtype=np.float32)
        input_im = open_image_sequence_to_3D(input_name, input_size, depth)
        
        
        """ Combine image + cell nucleus mask """
        cell_mask_name = examples[input_counter[i]]['cell_mask']
        cell_mask = open_image_sequence_to_3D(cell_mask_name, input_size, depth)
        temp = np.zeros(np.shape(input_im) + (2,))
        temp[:, :, :, 0] = input_im
        temp[:, :, :, 1] = cell_mask
                             
        input_im = temp
        cell_mask = []; temp = [];
        
        
        """ Load truth image, either BINARY or MULTICLASS """
        truth_name = examples[input_counter[i]]['truth']   
        truth_name_load = truth_name

        split = input_name.split('.')   # takes away the ".tif" filename at the end
        truth_name = split[0:-1]
        truth_name = '.'.join(truth_name)
            
        split = truth_name.split('/')   # takes away the "path // stuff" 
        truth_name = split[-1]
        
        #truth_name = glob.glob(os.path.join(input_path, 'Output//' + truth_name + '*.tif'))
        #truth_name = s_path + '/' + truth_name
        
        #print(truth_name)    
        #print(examples)
        truth_im, weighted_labels = load_class_truth_3D(truth_name_load, num_truth_class, input_size, depth, spatial_weight_bool=0)

        
        #imsave(s_path + '/' + truth_name + '_channel_0_analysis_output.tif', truth_im[:, :, :, 0])
        #imsave(s_path + '/' + truth_name + '_channel_1_analysis_output.tif', truth_im[:, :, :, 1])
        #imsave(s_path + '/' + truth_name + '_channel_2_analysis_output.tif', truth_im[:, :, :, 2])
        
        
        truth_im = np.argmax(truth_im, axis = -1)
        truth_im = np.amax(truth_im, axis= 0)
        
        input_im = np.amax(input_im, axis = 0)
        #input_im_val = np.amax(input_im_val, axis = 0)  
                 
        plt.figure(num=5, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(221); plt.imshow(truth_im[:, :]); plt.title('Truth Train');
   
        plt.subplot(222); plt.imshow(input_im[:, :, 1]); plt.title('Cell mask');
        plt.subplot(223); plt.imshow(input_im[:, :, 0]); plt.title('Input');
   
        plt.savefig(s_path + '_' + str(i) + '_max_project_output.png')
        
        
        
        
        
        
        
        
        
        
