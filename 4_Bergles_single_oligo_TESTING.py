# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???
 
 
 Try reducing network size
 

@author: Tiger
"""



""" ALLOWS print out of results on compute canada """
from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt



""" Libraries to load """
import tensorflow as tf
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

from plot_functions import *
from data_functions import *
from data_functions_3D import *
#from post_process_functions import *
from UNet import *
from UNet_3D import *
import glob, os
#from tifffile import imsave


# Initialize everything with specific random seeds for repeatability
tf.set_random_seed(2);

"""  Network Begins:
"""
#s_path = './Checkpoints/'
s_path = './Checkpoints/4_Bergles_single_oligo_RESIZED_deep_net_FLOAT32/'

#s_path = './Checkpoints/test/'

#input_path = '../20190713_FINAL_raw_images_with_all_classes/patches_skel/'
input_path = './Raw data/'

#input_path = './Training/RESIZED/'

input_path = '/scratch/yxu233/Training_data_Bergles_lab/RESIZED/'


#val_path = '../Validation/'
val_path = 0

""" load mean and std """  
mean_arr = load_pkl('', 'mean_arr.pkl')
std_arr = load_pkl('', 'std_arr.pkl')
               
""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*input_RESIZED.tif'))
examples = [dict(input=i,truth=i.replace('input_RESIZED','truth'), cell_mask=i.replace('input_RESIZED','input_cellMASK_RESIZED')) for i in images]
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


""" Make other validation counter for a separate folder """
if val_path:
    input_counter = counter
    images_val = glob.glob(os.path.join(val_path,'*input_*.tif'))
    examples_val = [dict(input=i,truth=i.replace('_input_','_truth_')) for i in images_val]
    counter_val = list(range(len(examples_val)))  # create a counter, so can randomize it
    counter_val = np.array(counter_val)
    np.random.shuffle(counter_val)
    
    validation_counter = counter_val

""" SO LAYERS MUST BE 2 x 2 x 2 x 1 for depth convolutions"""

#input_size = 2048 
#input_size = 1024
input_size = 960
num_truth_class = 2 + 1 # ALWAYS MINUS 1 for the background
#alter_images = 1
depth = 80   # ***OR can be 160

""" original == 60 * 320 * 320, now == 2100 * 150 * 150    so about 7.5 x larger image """

x_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, 2], name='3D_x') 
y_3D_ = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name='3D_CorrectLabel')
weight_matrix_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name = 'weighted_labels')
training = tf.placeholder(tf.bool, name='training')

""" Creates network and cost function"""
depth_filter = 5
height_filter = 5
width_filter = 5
kernel_size = [depth_filter, height_filter, width_filter]
y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits_3D, softMaxed = create_network_3D(x_3D, y_3D_, kernel_size, training, num_truth_class)
#y_3D, y_b_3D, L1, L2, L3, L8, L9, L9_conv, L10, L11, logits_3D, softMaxed = create_network_3D_smaller(x_3D, y_3D_,kernel_size, training, num_truth_class)

accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y_3D, y_b_3D, logits_3D, weight_matrix_3D, train_rate=1e-5, epsilon=1e-8, weight_mat=True)

sess = tf.InteractiveSession()

""" TO LOAD OLD CHECKPOINT """
saver = tf.train.Saver()

# Read in file names
onlyfiles_check = [ f for f in listdir(s_path) if isfile(join(s_path,f))]   
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
onlyfiles_check.sort(key = natsort_key1)

 
""" Find last checkpoint """   
last_file = onlyfiles_check[-13]
split = last_file.split('.')
checkpoint = split[0]
num_check = checkpoint.split('_')
num_check = int(num_check[1])
saver.restore(sess, s_path + checkpoint)
    
#tf.global_variables_initializer().run()
#tf.local_variables_initializer().run()   # HAVE TO HAVE THESE in order to initialize the newly created layers

""" Prints out all variables in current graph """
tf.trainable_variables()


# Required to initialize all
batch_x = []; batch_y = [];
weights = [];

for i in range(1, len(input_counter)):
        
        # degrees rotated:
        rand = randint(0, 360)      
        """ Load input image """
        input_name = examples[input_counter[i]]['input']
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
        truth_im, weighted_labels = load_class_truth_3D(truth_name, num_truth_class, input_size, depth, spatial_weight_bool=1)


        """ maybe remove normalization??? """
        input_im_save = np.copy(input_im)
        input_im = normalize_im(input_im, mean_arr, std_arr) 


        """ set inputs and truth """
        batch_x.append(input_im)
        batch_y.append(truth_im)
        weights.append(weighted_labels)
        
        
        """ Plot for debug """
        feed_dict = {x_3D:batch_x, y_3D_:batch_y, training:0, weight_matrix_3D:weights}   
        output_train = softMaxed.eval(feed_dict=feed_dict)
        seg_train = np.argmax(output_train, axis = -1)[0]              
                            
        plt.figure(num=10, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
        plt.subplot(211); plt.imshow(seg_train[40, :, :]); plt.title('Output Train');                        
        plt.subplot(212); plt.imshow(input_im[40, :, :, 0]); plt.title('Input image train');
        plt.savefig(s_path + '_' + str(i) + '_output.png')
              
        plt.close(10)
    
        split = input_name.split('.')   # takes away the ".tif" filename at the end
        truth_name = split[0:-1]
        truth_name = '.'.join(truth_name)
        
        split = truth_name.split('/')   # takes away the "path // stuff" 
        truth_name = split[-1]
    
        #truth_name = glob.glob(os.path.join(input_path, 'Output//' + truth_name + '*.tif'))
        truth_name = input_path + 'Outputs/' + truth_name
            
        seg_train = np.asarray(seg_train, dtype=np.uint16)
        imsave(truth_name + '_argmax_analysis_output.tif', seg_train)
            
        print(np.unique(seg_train))

#        channel_1 = np.copy(seg_train)
#        channel_1[channel_1 != 0] = -1
#        channel_1[channel_1 == 0] = 1
#        channel_1[channel_1 == -1] = 0
#        channel_1 = np.asarray(channel_1, dtype=np.uint8)
#        imsave(truth_name + 'channel_1_analysis_output.tif', channel_1)
#    
#        channel_2 = np.copy(seg_train)
#        channel_2[channel_2 != 1] = 0
#        channel_2[channel_2 == 1] = 1
#        channel_2 = np.asarray(channel_2, dtype=np.uint8)
#        imsave(truth_name + 'channel_2_analysis_output.tif', channel_2)    
#    
#    
#        channel_3 = np.copy(seg_train)
#        channel_3[channel_3 != 2] = 0
#        channel_3[channel_3 == 2] = 1
#        channel_3 = np.asarray(channel_3, dtype=np.uint8)
#        imsave(truth_name + 'channel_3_analysis_output.tif', channel_3)
        
        input_im_tmp = np.asarray(input_im_save[:, :, :, 0], dtype=np.uint8)
        imsave(truth_name + 'cropped_input_channel_1.tif', input_im_tmp)  

        input_im_tmp = np.asarray(input_im_save[:, :, :, 1], dtype=np.uint8)
        imsave(truth_name + 'cropped_input_channel_2.tif', input_im_tmp)  
        
        
        batch_x = []; batch_y = [];
        weights = [];
        
#        """ Feed into training loop """
#        if len(batch_x) == batch_size:
#           feed_dict_TRAIN = {x_3D:batch_x, y_3D_:batch_y, training:1, weight_matrix_3D:weights}
#                                 
#           train_step.run(feed_dict=feed_dict_TRAIN)
#
#           batch_x = []; batch_y = []; weights = [];
#           epochs = epochs + 1           
#           print('Trained: %d' %(epochs))
#           
#           
#           if epochs % plot_every == 0:
#               
#              """ Load validation """
#              batch_x_val = []
#              batch_y_val = []
#              batch_weights_val = []
#              for batch_i in range(len(validation_counter)):
#            
#                  # degrees rotated:
#                  rand = randint(0, 360)   
#                  
#                  # select random validation image:
#                  rand_idx = randint(0, len(validation_counter)- 1)
#                     
#                  """ Load input image """
#                  if val_path == 0:
#                      input_name = examples[validation_counter[rand_idx]]['input']
#                  else:
#                      input_name = examples_val[validation_counter[rand_idx]]['input']
#                
#                  #input_im_val = np.asarray(Image.open(input_name), dtype=np.float32)
#                  input_im_val = open_image_sequence_to_3D(input_name, input_size, depth)
#            
#            
#                  """ Combine image + cell nucleus mask """
#                  if val_path == 0:
#                      cell_mask_name = examples[validation_counter[rand_idx]]['cell_mask']
#                  else:
#                      cell_mask_name = examples_val[validation_counter[rand_idx]]['cell_mask']
#                  cell_mask_val = open_image_sequence_to_3D(cell_mask_name, input_size, depth)
#                  temp = np.zeros(np.shape(input_im_val) + (2,))
#                  temp[:, :, :, 0] = input_im_val
#                  temp[:, :, :, 1] = cell_mask_val
#                                         
#                  input_im_val = temp
#                  cell_mask_val = []; temp = [];
#                  
#                  """ maybe remove normalization??? """
#                  input_im_val = normalize_im(input_im_val, mean_arr, std_arr) 
#
#            
#                  """ Load truth image """
#                  if val_path == 0:
#                      truth_name = examples[validation_counter[rand_idx]]['truth']                  
#                  else:
#                      truth_name = examples_val[validation_counter[rand_idx]]['truth']                  
#                      
#                  truth_im_val, weighted_labels_val = load_class_truth_3D(truth_name, num_truth_class, input_size, depth, spatial_weight_bool=1)
#
#                  """ set inputs and truth """
#                  batch_x_val.append(input_im_val)
#                  batch_y_val.append(truth_im_val)
#                  batch_weights_val.append(weighted_labels_val)
#                  
#                  if len(batch_x_val) == batch_size:
#                      break             
#              feed_dict_CROSSVAL = {x_3D:batch_x_val, y_3D_:batch_y_val, training:0, weight_matrix_3D:batch_weights_val}      
              
 

                            