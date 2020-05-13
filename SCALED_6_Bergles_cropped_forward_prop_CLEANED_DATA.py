# -*- coding: utf-8 -*-
"""
Created on February 15th, 2020
============================================================

Runs neural network for training snake-seg based reconstruction of oligodendrocytes

- updated UNet3D to accomodate for 80 x 80 images

@author: Tiger
"""



""" ALLOWS print out of results on compute canada for CEDAR CLUSER (maybe Graham???) NOT BELUGA """
from keras import backend as K
cfg = K.tf.ConfigProto()
cfg.gpu_options.allow_growth = True
K.set_session(K.tf.Session(config=cfg))

import matplotlib
#matplotlib.rc('xtick', labelsize=8) 
#matplotlib.rc('ytick', labelsize=8) 
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

from UnetTrainingFxn_0v6_actualFunctions import *
from random import randint


# Initialize everything with specific random seeds for repeatability
tf.set_random_seed(2);

"""  Network Begins:
"""
s_path = './Checkpoints/SCALED_6_Bergles_cropped_forward_prop_CLEANED_DATA/'

input_path = '/scratch/yxu233/TRAINING_FORWARD_PROP_SCALED_FINISHED_CLEANED_DATA/'
#input_path = 'E:/7) Bergles lab data/Traces files/TRAINING FORWARD PROP SCALED FINISHED/'

#input_path = './Traces files/TRAINING FORWARD PROP ONLY SCALED/'



val_path = 0

""" load mean and std """  
mean_arr = load_pkl('', 'mean_val_SCALED_NOCLAHE_SCALED.pkl')
std_arr = load_pkl('', 'std_val_SCALED_NOCLAHE_SCALED.pkl')


#mean_arr = load_pkl('', 'mean_arr.pkl')
#std_arr = load_pkl('', 'std_arr.pkl')
               
""" Load LARGE samples """
images_large = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop_BRANCHED.tif'))

natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images_large.sort(key = natsort_key1)
examples_large = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop_BRANCHED','_DILATE_truth_crop'), seed_crop=i.replace('_NOCLAHE_input_crop_BRANCHED','_DILATE_seed_crop')) for i in images_large]

counter_large = list(range(len(examples_large)))  # create a counter, so can randomize it
counter_large = np.array(counter_large)
np.random.shuffle(counter_large)
               

""" Load samples with paranodes """
images_para = glob.glob(os.path.join(input_path,'*_crop_pos.tif'))

images_para.sort(key = natsort_key1)
examples_para = [dict(input=i.replace('_DILATE_truth_class_3_crop_pos','_NOCLAHE_input_crop'),truth=i.replace('_DILATE_truth_class_3_crop_pos','_DILATE_truth_crop'), 
                       seed_crop=i.replace('_DILATE_truth_class_3_crop_pos','_DILATE_seed_crop')) for i in images_para]

counter_para= list(range(len(examples_para)))  # create a counter, so can randomize it
counter_para = np.array(counter_para)
np.random.shuffle(counter_para)



""" Load filenames from folder """
images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))

images.sort(key = natsort_key1)
""" FOR CLAHE """
examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop','_DILATE_truth_crop'), seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop')) for i in images]

counter = list(range(len(examples)))  # create a counter, so can randomize it
counter = np.array(counter)
np.random.shuffle(counter)

val_size = 0.1;
val_idx_sub = round(len(counter) * val_size)
if val_idx_sub == 0:
    val_idx_sub = 1
validation_counter = counter[-1 - val_idx_sub : len(counter)]
input_counter = counter[0: -1 - val_idx_sub]


""" SO LAYERS MUST BE 2 x 2 x 2 x 1 for depth convolutions"""
input_size = 80
num_truth_class = 1 + 1 # ALWAYS MINUS 1 for the background
multiclass = 0
if num_truth_class > 2:
     multiclass = 1
     
depth = 16

""" variable declaration """
x_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, 2], name='3D_x') 
y_3D_ = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name='3D_CorrectLabel')
weight_matrix_3D = tf.placeholder('float32', shape=[None, depth, input_size, input_size, num_truth_class], name = 'weighted_labels')
training = tf.placeholder(tf.bool, name='training')

""" Creates network and cost function"""
depth_filter = 7
height_filter = 7
width_filter = 7
kernel_size = [depth_filter, height_filter, width_filter]
y_3D, y_b_3D, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits_3D, softMaxed = create_network_3D_COMPLEX_REAL_NO_POOL_Z(x_3D, y_3D_, kernel_size, training, num_truth_class)
accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original = costOptm(y_3D, y_b_3D, logits_3D, weight_matrix_3D, 
                                                                                            train_rate=1e-5, epsilon=1e-8, weight_mat=True, optimizer='adam', multiclass=multiclass)

""" TO LOAD OLD CHECKPOINT """
sess = tf.InteractiveSession()
saver = tf.train.Saver()

# Read in file names
onlyfiles_check = glob.glob(os.path.join(s_path,'check_*0.index'))
onlyfiles_check.sort(key = natsort_key1)


""" If no old checkpoint then starts fresh FROM PRE-TRAINED NETWORK """
if not onlyfiles_check:  
    tf.global_variables_initializer().run()
    tf.local_variables_initializer().run() 
    plot_cost = []; plot_cost_val = []; plot_jaccard = []; plot_jaccard_val = [];
    plot_jaccard_single = []; plot_loss_single = [];
    num_check= 0;
    
    if multiclass:
      for i in range(num_truth_class - 1):
           plot_jaccard.append([])
           plot_jaccard_val.append([])
    
else:   
    """ Find last checkpoint """       
    last_file = onlyfiles_check[-1]
    split = last_file.split('check_MAIN')[-1]
    num_check = split.split('.')
    checkpoint = num_check[0]
    checkpoint = 'check_MAIN' + checkpoint
    num_check = int(num_check[0])
    
    #checkpoint = 'check_36400' 
    saver.restore(sess, s_path + checkpoint)
    
    def load_plot_pickles(filename):
         # Getting back the objects:
         with open(filename, 'rb') as f:  # Python 3: open(..., 'rb')
             loaded = pickle.load(f)
             loaded_value = loaded[0] 
         return loaded_value
         
    # Getting back the objects:
    plot_cost = load_plot_pickles(s_path + 'loss_global.pkl')
    plot_cost_val = load_plot_pickles(s_path + 'loss_global_val.pkl')
    plot_jaccard = load_plot_pickles(s_path + 'jaccard.pkl')
    plot_jaccard_val = load_plot_pickles(s_path + 'jaccard_val.pkl')
    validation_counter = load_plot_pickles(s_path + 'val_counter.pkl')
    input_counter = load_plot_pickles(s_path + 'input_counter.pkl')



""" Prints out all variables in current graph """
tf.trainable_variables()


# Required to initialize all
batch_size = 1; 
save_epoch = 5000;
plot_every = 10;
epochs = num_check;

batch_x = []; batch_y = [];
weights = [];

alter_images = 1;

rand_num = randint(0, 10)
for P in range(8000000000000000000000):
    np.random.shuffle(validation_counter)
    np.random.shuffle(input_counter)

    np.random.shuffle(counter_large)
    np.random.shuffle(counter_para)
    
    large_idx = 0
    para_idx = 0
    for i in range(len(input_counter)):


        """ Size based selection """
        if rand_num <= 3:     # 30% of the time want random one
             rand_num = randint(1, 10)
             filenum = input_counter[i]
             input_name = examples[filenum]['input']
             seed_crop_name = examples[filenum]['seed_crop']
             truth_name = examples[filenum]['truth']  
             #print(np.count_nonzero(test_split))
             #print('small')
              
        elif rand_num > 3 and rand_num <= 8:    # 40% of the time show split ==> of total 12% are split     % changed to 50% at 170,000
             rand_num = randint(1, 10)
             if large_idx > len(counter_large) - 1:
                  large_idx = 0;   # reset if too largr
             filenum = counter_large[large_idx]
             input_name = examples_large[filenum]['input']
             seed_crop_name = examples_large[filenum]['seed_crop']
             truth_name = examples_large[filenum]['truth'] 
             large_idx += 1
             #print('large')
             
        elif rand_num > 8:  # 30% of time show paranodes     % changed to 20% at 170,000
             rand_num = randint(1, 10)
             if para_idx > len(counter_para) - 1:
                  para_idx = 0;   # reset if too largr
             filenum = counter_para[large_idx]
             input_name = examples_para[filenum]['input']
             seed_crop_name = examples_para[filenum]['seed_crop']
             truth_name = examples_para[filenum]['truth'] 
             para_idx += 1
             #print('para')            
             
             



        #input_name = examples[input_counter[i]]['input']
        #seed_crop_name = examples[input_counter[i]]['seed_crop']
        #truth_name = examples[input_counter[i]]['truth']   
        
        
        # degrees rotated:
        rand = randint(0, 360)      
        """ Load input image """
        #input_name = examples[input_counter[i]]['input']
        input_im = open_image_sequence_to_3D(input_name, width_max=input_size, height_max=input_size, depth=depth)

        
        """ Combine image + cell nucleus mask """
        #seed_crop_name = examples[input_counter[i]]['seed_crop']
        seed_crop_mask = open_image_sequence_to_3D(seed_crop_name, width_max=input_size, height_max=input_size, depth=depth)
        seed_crop_mask[seed_crop_mask > 0] = 1

        """ Added because need to check to make sure no objects leaving out of frame during small crop """
        seed_crop_mask = np.expand_dims(seed_crop_mask, axis=-1)
        seed_crop_mask = check_resized(seed_crop_mask, depth, width_max=input_size, height_max=input_size)
        seed_crop_mask = seed_crop_mask[:, :, :, 0]


        temp = np.zeros(np.shape(input_im) + (2,))
        temp[:, :, :, 0] = input_im
        seed_crop_mask[seed_crop_mask > 0] = 100
        temp[:, :, :, 1] = seed_crop_mask
                             
        input_im = temp
    
        """ Load truth image, either BINARY or MULTICLASS """
        #truth_name = examples[input_counter[i]]['truth']   
        truth_im, weighted_labels = load_class_truth_3D(truth_name, num_truth_class, width_max=input_size, height_max=input_size,
                                                        depth=depth, spatial_weight_bool=1, delete_seed=seed_crop_mask, skip_class=2, resized_check=1)
        

        if alter_images:
            perform_alteration = randint(0, 10)
            random = 1
            amount = random
            
            altered_im = input_im/255  # SCALE TO between -1 and 1
            altered_truth = truth_im
            altered_weighted = weighted_labels
            #plt.figure(); plt.imshow(altered_im[30, :, :, :]); plt.title("Original")
    
            """ (1) Add noise """
            if perform_alteration > 10:  # ONLY PERFORMS TRANSFORMATION 20% of the time
                kind = randint(1,3)
                altered_im = changeNoise(im=altered_im,kind=kind,amount=amount, random=1)    # kind: 1=gaussian, 2=speckle, 3=salt and pepper
                                                    # amount = variation of noise (for gaussian and speckle) or amount of 
                                                    # pixels to replace with noise (for salt and pepper); 
                                                    # random = if true, applies noise btwn 0 and amount
            perform_alteration = randint(0, 10)
            #perform_alteration = 10
            #plt.figure(); plt.imshow(altered_im[30, :, :, :]); plt.title("Add noise")
    
            """ (2) Blur """
            if perform_alteration > 10:  # ONLY PERFORMS TRANSFORMATION 20% of the time
                altered_im = changeBlur(im=altered_im,amount=amount,random=random)    # amount = proportional to sigma of gaussian; generally btwn 0 and 1
                                                                    # random = if true, will apply some blur btwn 0 and amount
                                                                    # returns blurred image
                                                                # note - gaussian function has other optional parameters including wrap         
            perform_alteration = randint(0, 10)
            #perform_alteration = 10
            #plt.figure(); plt.imshow(altered_im[30, :, :, :]); plt.title("Blur")
    
    
            """ (3) Change contrast ==> doesn't work with CLAHE at the moment? """
            if perform_alteration > 10:  # ONLY PERFORMS TRANSFORMATION 20% of the time
               kind = randint(1,3)
               if kind == 2:
                   print("skip CLAHE for now")    
               else:
                   altered_im = changeContrast(im=altered_im,kind=kind,amount=amount,random=random)# kind: 1 = linear equalize, 2 = adaptive equalize, 3 = linear defined
                                                # amount does nothing for linear equalization, sets the clip limit for CLAHE,
                                                    # and sets the magnitude of floor raising / ceiling lowering for linear defined
                                                # random = if true, randomly sets amount to be btwn 0 and amount for CLAHE; for
                                                    # linear defined, adds a random number between 0 and half the image range * amount,
                                                    # and subtracts different random number from ceiling
            perform_alteration = randint(0, 10)
            #perform_alteration = 10
            #plt.figure(); plt.imshow(altered_im[30, :, :, :]); plt.title("Change contrast")
    
    
            """ (4) Flip image """
            if perform_alteration > 2:  # ONLY PERFORMS TRANSFORMATION 50% of the time
                kind = randint(1,3)
                altered_im = changeFlip(im=altered_im,kind=kind,random=0)    # kind = (1 = reflect X axis, 2 = reflect Y, 3 = reflect both)
                                                            # random = if true, ignores kind and chooses a random reflection

                altered_truth = changeFlip(im=altered_truth,kind=kind,random=0)    # kind = (1 = reflect X axis, 2 = reflect Y, 3 = reflect both)
                altered_weighted = changeFlip(im=altered_weighted,kind=kind,random=0)    # kind = (1 = reflect X axis, 2 = reflect Y, 3 = reflect both)


            #plt.figure(); plt.imshow(altered_im[30, :, :, :]); plt.title("Change orientation")
            
            input_im = altered_im * 255
            truth_im = altered_truth
            weighted_labels = altered_weighted


        """ normalize data """
        input_im_save = np.copy(input_im)
        input_im = normalize_im(input_im, mean_arr, std_arr) 


        """ set inputs and truth """
        batch_x.append(input_im)
        batch_y.append(truth_im)
        weights.append(weighted_labels)

        """ Feed into training loop """
        if len(batch_x) == batch_size:
           
           feed_dict_TRAIN = {x_3D:batch_x, y_3D_:batch_y, training:1, weight_matrix_3D:weights}
                                 
           train_step.run(feed_dict=feed_dict_TRAIN)

           batch_x = []; batch_y = []; weights = [];
           epochs = epochs + 1           
           print('Trained: %d' %(epochs))
           
           
           if epochs % plot_every == 0:
               
              """ Load validation """
              #plt.close(2)
              #plt.close(18)
              #plt.close(19)
              #plt.close(21)
              batch_x_val = []
              batch_y_val = []
              batch_weights_val = []
              for batch_i in range(len(validation_counter)):
            
                  # degrees rotated:
                  rand = randint(0, 360)   
                  
                  # select random validation image:
                  rand_idx = randint(0, len(validation_counter)- 1)
                     
                  """ Load input image """
                  if val_path == 0:
                      input_name = examples[validation_counter[rand_idx]]['input']
                  else:
                      input_name = examples_val[validation_counter[rand_idx]]['input']

                  input_im_val = open_image_sequence_to_3D(input_name, width_max=input_size, height_max=input_size, depth=depth)
           
               
                  """ Combine image + cell nucleus mask """
                  if val_path == 0:
                      seed_crop_name = examples[validation_counter[rand_idx]]['seed_crop']
                  else:
                      seed_crop_name = examples_val[validation_counter[rand_idx]]['seed_crop']
                  seed_crop_mask_val = open_image_sequence_to_3D(seed_crop_name, width_max=input_size, height_max=input_size, depth=depth)
                  seed_crop_mask_val[seed_crop_mask_val > 0] = 1


                  """ Added because need to check to make sure no objects leaving out of frame during small crop """
                  seed_crop_mask_val = np.expand_dims(seed_crop_mask_val, axis=-1)
                  seed_crop_mask_val = check_resized(seed_crop_mask_val, depth, width_max=input_size, height_max=input_size)
                  seed_crop_mask_val = seed_crop_mask_val[:, :, :, 0]



                  temp = np.zeros(np.shape(input_im_val) + (2,))
                  temp[:, :, :, 0] = input_im_val
                  seed_crop_mask_val[seed_crop_mask_val > 0] = 100
                  temp[:, :, :, 1] = seed_crop_mask_val
                                   
                                         
                  input_im_val = temp

                  """ maybe remove normalization??? """
                  input_im_val = normalize_im(input_im_val, mean_arr, std_arr) 

            
                  """ Load truth image """
                  if val_path == 0:
                      truth_name = examples[validation_counter[rand_idx]]['truth']                  
                  else:
                      truth_name = examples_val[validation_counter[rand_idx]]['truth']                    
                  truth_im_val, weighted_labels_val = load_class_truth_3D(truth_name, num_truth_class, width_max=input_size, height_max=input_size, depth=depth, 
                                                                          spatial_weight_bool=1,  delete_seed=seed_crop_mask_val, skip_class=2, resized_check=1)
          
                  """ set inputs and truth """
                  batch_x_val.append(input_im_val)
                  batch_y_val.append(truth_im_val)
                  batch_weights_val.append(weighted_labels_val)
                  
                  if len(batch_x_val) == batch_size:
                      break             
              feed_dict_CROSSVAL = {x_3D:batch_x_val, y_3D_:batch_y_val, training:0, weight_matrix_3D:batch_weights_val}      
              
 
              """ Training loss"""
              loss_t = cross_entropy.eval(feed_dict=feed_dict_TRAIN);
              plot_cost.append(loss_t);                 
                             
              """ loss """
              loss_val = cross_entropy.eval(feed_dict=feed_dict_CROSSVAL)
              plot_cost_val.append(loss_val)
                           
              if not multiclass:
                   """ Training Jaccard """
                   jacc_t = jaccard.eval(feed_dict=feed_dict_TRAIN)
                   plot_jaccard.append(jacc_t)                     
                        
                   """ Jaccard """
                   jacc_val = jaccard.eval(feed_dict=feed_dict_CROSSVAL)
                   plot_jaccard_val.append(jacc_val)
                   
                   """ function call to plot """
                   plot_cost_fun(plot_cost, plot_cost_val)
                   plot_jaccard_fun(plot_jaccard, plot_jaccard_val)
                   
                   plt.figure(18); plt.savefig(s_path + 'global_loss.png')
                   plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
                   plt.figure(21); plt.savefig(s_path + 'jaccard.png')
                   plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
                   
              else:
                   """ Get diff jaccard for each class """
                   class_idx = 1
                   for jacc in jaccard:
                        j_val = jacc.eval(feed_dict=feed_dict_TRAIN)
                        plot_jaccard[class_idx - 1].append(j_val)
                        
                        j_val = jacc.eval(feed_dict=feed_dict_CROSSVAL)
                        plot_jaccard_val[class_idx - 1].append(j_val)                        
  
                        plot_jaccard_fun(plot_jaccard[class_idx -1], plot_jaccard_val[class_idx - 1], class_name=' class ' + str(class_idx))                      
                        plt.figure(21); plt.savefig(s_path + 'jaccard class ' + str(class_idx) + '.png')
                        class_idx += 1
              
                   """ function call to plot """
                   plot_cost_fun(plot_cost, plot_cost_val)
                   plt.figure(18); plt.savefig(s_path + 'global_loss.png')
                   plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
                   plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')

               
                           
              plot_depth = 8
              if epochs > 500:
                  if epochs % 100 == 0:
                       plot_trainer_3D(softMaxed, feed_dict_TRAIN, feed_dict_CROSSVAL, input_im, input_im_val, truth_im, truth_im_val,
                                       weighted_labels, weighted_labels_val, s_path, epochs, plot_depth=plot_depth, multiclass=multiclass)

              elif epochs % plot_every == 0:
                       plot_trainer_3D(softMaxed, feed_dict_TRAIN, feed_dict_CROSSVAL, input_im, input_im_val, truth_im, truth_im_val,
                                       weighted_labels, weighted_labels_val, s_path, epochs, plot_depth=plot_depth, multiclass=multiclass)
                                                     
           """ To save (every x epochs) """
           if epochs % save_epoch == 0:                          
              sess_get = tf.get_default_session()   # IF FORGET TO ADD "sess = tf.InteractiveSession
              saver = tf.train.Saver()
       
              save_name = s_path + 'check_MAIN' 
              save_path = saver.save(sess, save_name)
               
              """ Saving the objects """
              save_pkl(plot_cost, s_path, 'loss_global.pkl')
              save_pkl(plot_cost_val, s_path, 'loss_global_val.pkl')
              save_pkl(plot_jaccard, s_path, 'jaccard.pkl')
              save_pkl(plot_jaccard_val, s_path, 'jaccard_val.pkl')
              save_pkl(validation_counter, s_path, 'val_counter.pkl')
              save_pkl(input_counter, s_path, 'input_counter.pkl')   


           """ To save larger timescales but take up less disk space """
           if epochs % save_epoch * 10 == 0:                          
              sess_get = tf.get_default_session()   # IF FORGET TO ADD "sess = tf.InteractiveSession
              saver = tf.train.Saver()
       
              save_name = s_path + 'check_MAIN'  +  str(epochs)
              save_path = saver.save(sess, save_name)
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           