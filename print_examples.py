# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???
 
 
 Try reducing network size
 

@author: Tiger
"""

""" Libraries to load """
from matplotlib import *
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from natsort import natsort_keygen, ns
from skimage import measure
import pickle as pickle
import os

from random import randint

from data_functions import *
import glob, os
#from tifffile import imsave


# Initialize everything with specific random seeds for repeatability
tf.set_random_seed(2);

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
np.random.shuffle(counter)

val_size = 0;
val_idx_sub = round(len(counter) * val_size)
if val_idx_sub == 0:
    val_idx_sub = 1
validation_counter = counter[-1 - val_idx_sub : len(counter)]
input_counter = counter[0: -1 - val_idx_sub]

""" Saving the objects """
save_pkl(examples, s_path, 'examples.pkl')
examples
save_pkl(validation_counter, s_path, 'val_counter.pkl')
save_pkl(input_counter, s_path, 'input_counter.pkl')



s_path = './Checkpoints/6_Bergles_single_oligo_512x512/'        
        
# Getting back the objects:
with open(s_path + 'val_counter.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loaded = pickle.load(f)
    validation_counter = loaded[0]     
        
# Getting back the objects:
with open(s_path + 'input_counter.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    loaded = pickle.load(f)
    input_counter = loaded[0]  
    
examples[validation_counter[0]]
examples[validation_counter[1]]
examples
