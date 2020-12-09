# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger
"""

""" ALLOWS print out of results on compute canada """
import matplotlib
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8) 
#matplotlib.use('Agg')

""" Libraries to load """
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import glob, os
import datetime
import time
from sklearn.model_selection import train_test_split

from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order


from PYTORCH_dataloader import *
from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.tracker import *


from layers.UNet_pytorch_online import *
from layers.unet_nested import *
from layers.unet3_3D import *
from layers.switchable_BN import *

from losses_pytorch.HD_loss import *
import re
""" optional dataviewer if you want to load it """
#import napari
# with napari.gui_qt():
#     viewer = napari.view_image(seg_val)

torch.backends.cudnn.benchmark = True  ### set these options to improve speed
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    """" path to checkpoints """       
    s_path = './(67) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_4_step_NEURON/'; HD = 1; alpha = 1;
    
    
    """ path to input data """
    input_path = '/media/user/Seagate Portable Drive/Bergles lab data 2021/Su_Jeong_neurons/Training data SOLANGE/TRAINING FORWARD PROP ONLY SCALED crop pads seed 2 COLORED 48 z/TRAINING_FORWARD_NEURON_SOLANGE/'
    
    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), 
                     seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop'),  
                     orig_idx= int(re.search('_origId_(.*)_eId', i).group(1)),
                     filename= i.split('/')[-1].split('_origId')[0].replace(',', ''))
                     for i in images]


    
    # ### REMOVE IMAGE 1 from training data
    idx_skip = []
    for idx, im in enumerate(examples):
        filename = im['input']
        if 'RBP4_HK_5_slice3_40x_stit-Create Image Subset-08-N3_' in filename:
            print('skip')
            idx_skip.append(idx)
    examples = examples[idx_skip[0]:idx_skip[-1]]
    #examples = [i for j, i in enumerate(examples) if j not in idx_skip]
    

    counter = list(range(len(examples)))

    # """ load mean and std for normalization later """  
    mean_arr = np.load('./normalize/' + 'mean_VERIFIED.npy')
    std_arr = np.load('./normalize/' + 'std_VERIFIED.npy')   

    num_workers = 2;
 
    save_every_num_epochs = 1; plot_every_num_epochs = 1; validate_every_num_epochs = 1;      
    
    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
    onlyfiles_check.sort(key = natsort_key1)
    
    """ Find last checkpoint """       
    last_file = onlyfiles_check[-1]
    split = last_file.split('check_')[-1]
    num_check = split.split('.')
    checkpoint = num_check[0]
    checkpoint = 'check_' + checkpoint

    print('restoring weights from: ' + checkpoint)
    check = torch.load(s_path + checkpoint, map_location=device)
    #check = torch.load(s_path + checkpoint, map_location='cpu')
    #check = torch.load(s_path + checkpoint, map_location=device)
        
    
    tracker = check['tracker']

    tracker.idx_valid = counter
    
    #tracker.idx_valid = idx_skip   ### IF ONLY WANT 
    
    
    tracker.idx_train = []

    tracker.batch_size = 1
    tracker.train_loss_per_batch = [] 
    tracker.train_jacc_per_batch = []
    tracker.val_loss_per_batch = []; tracker.val_jacc_per_batch = []
    
    tracker.train_ce_pb = []; tracker.train_hd_pb = []; tracker.train_dc_pb = [];
    tracker.val_ce_pb = []; tracker.val_hd_pb = []; tracker.val_dc_pb = [];
 
    """ Get metrics per epoch"""
    tracker.train_loss_per_epoch = []; tracker.train_jacc_per_epoch = []
    tracker.val_loss_per_eval = []; tracker.val_jacc_per_eval = []
    tracker.plot_sens = []; tracker.plot_sens_val = [];
    tracker.plot_prec = []; tracker.plot_prec_val = [];
    tracker.lr_plot = [];
    tracker.iterations = 0;
    tracker.cur_epoch = 0;
    
    #tracker.


    for check_file in onlyfiles_check:      
        last_file = check_file
        """ Find last checkpoint """       
        #last_file = onlyfiles_check[-1]
        split = last_file.split('check_')[-1]
        num_check = split.split('.')
        checkpoint = num_check[0]
        checkpoint = 'check_' + checkpoint

        print('restoring weights from: ' + checkpoint)
        check = torch.load(s_path + checkpoint, map_location=device)
        #check = torch.load(s_path + checkpoint, map_location='cpu')
        #check = torch.load(s_path + checkpoint, map_location=device)

        # """ Print info """
        # tracker = check['tracker']
        # tracker.print_essential(); 
        # continue;
        
        
        unet = check['model_type']
        unet.load_state_dict(check['model_state_dict']) 
        unet.eval();   unet.to(device)

        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Clean up checkpoint file """
        del check
        torch.cuda.empty_cache()

                
        """ Create datasets for dataloader """
        #training_set = Dataset_tiffs_snake_seg(tracker.idx_train, examples, tracker.mean_arr, tracker.std_arr, sp_weight_bool=tracker.sp_weight_bool, transforms = tracker.transforms)
        val_set = Dataset_tiffs_snake_seg(tracker.idx_valid, examples, tracker.mean_arr, tracker.std_arr, sp_weight_bool=tracker.sp_weight_bool, transforms = 0)
        
        """ Create training and validation generators"""
        val_generator = data.DataLoader(val_set, batch_size=tracker.batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last = True)
    
        # training_generator = data.DataLoader(training_set, batch_size=tracker.batch_size, shuffle=True, num_workers=num_workers,
        #                   pin_memory=True, drop_last=True)
             
        #print('Total # training images per epoch: ' + str(len(training_set)))
        print('Total # validation images: ' + str(len(val_set)))
        
    
        """ Epoch info """
        #train_steps_per_epoch = len(tracker.idx_train)/tracker.batch_size
        validation_size = len(tracker.idx_valid)
        #epoch_size = len(tracker.idx_train)    
       
        

         
         
        """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
        loss_val = 0; jacc_val = 0; val_idx = 0;
        iter_cur_epoch = 0;  ce_val = 0; dc_val = 0; hd_val = 0;
        if tracker.cur_epoch % validate_every_num_epochs == 0:
            
             with torch.set_grad_enabled(False):  # saves GPU RAM
                  unet.eval()
                  for batch_x_val, batch_y_val, spatial_weight in val_generator:
                     
                       """ Transfer to GPU to normalize ect... """
                       inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, tracker.mean_arr, tracker.std_arr)
                       inputs_val = inputs_val[:, 0, ...]   
            
                       # forward pass to check validation
                       output_val = unet(inputs_val)
   
                       """ calculate loss 
                               include HD loss functions """
                       if tracker.HD:
                           loss, tracker, ce_val, dc_val, hd_val = compute_HD_loss(output_val, labels_val, tracker.alpha, tracker, 
                                                                                         ce_val, dc_val, hd_val, val_bool=1)
                       else:
                           if deep_sup:                                                
                               # compute output
                               loss = 0
                               for output in output_val:
                                    loss += loss_function(output, labels_val)
                               loss /= len(output_val)                                
                               output_val = output_val[-1]  # set this so can eval jaccard later                            
                           else:
                           
                               loss = loss_function(output_val, labels_val)       
                               if torch.is_tensor(spatial_weight):
                                      spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                                      weighted = loss * spatial_tensor
                                      loss = torch.mean(weighted)
                               elif dist_loss:
                                      loss  # do not do anything if do not need to reduce
                                   
                               else:
                                      loss = torch.mean(loss)  
   
                       """ Training loss """
                       tracker.val_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                       loss_val += loss.cpu().data.numpy()
                                        
                       """ Calculate jaccard on GPU """
                       jacc = jacc_eval_GPU_torch(output_val, labels_val)
                       jacc = jacc.cpu().data.numpy()
                       
                       jacc_val += jacc
                       tracker.val_jacc_per_batch.append(jacc)   
   
                       val_idx = val_idx + tracker.batch_size
                       print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                       iter_cur_epoch += 1
   
                       #if starter == 50: stop = time.perf_counter(); diff = stop - start; print(diff);  #break;
   
                          
                  tracker.val_loss_per_eval.append(loss_val/iter_cur_epoch)
                  tracker.val_jacc_per_eval.append(jacc_val/iter_cur_epoch)       
                  
                  """ Add to scheduler to do LR decay """
                  #scheduler.step()
                 
        """ Plot metrics every epoch """      
        if tracker.cur_epoch % plot_every_num_epochs == 0:       
             
            
             """ Plot metrics in tracker """
             plot_tracker(tracker, s_path)
             
             """ custom plot """
             # output_train = output_train.cpu().data.numpy()            
             # output_train = np.moveaxis(output_train, 1, -1)              
             # seg_train = np.argmax(output_train[0], axis=-1)  
             
             # convert back to CPU
             # batch_x = batch_x.cpu().data.numpy() 
             # batch_y = batch_y.cpu().data.numpy() 
             batch_x_val = batch_x_val.cpu().data.numpy()
             
             
             batch_y_val = batch_y_val.cpu().data.numpy() 
             output_val = output_val.cpu().data.numpy()            
             output_val = np.moveaxis(output_val, 1, -1)       
             seg_val = np.argmax(output_val[0], axis=-1)  
             
             plot_trainer_3D_PYTORCH_snake_seg(seg_val, seg_val, batch_x_val[0], batch_x_val[0], batch_y_val[0], batch_y_val[0],
                                      s_path, tracker.iterations, plot_depth=8)
                                            
             
        """ To save tracker and model (every x iterations) """
        # if tracker.cur_epoch % save_every_num_epochs == 0:           
        #       tracker.iterations += 1

        #       save_name = s_path + 'check_RERUN_' +  str(tracker.iterations)               
        #       torch.save({
        #        'tracker': tracker,

               
        #        }, save_name)
     
                

                
               
              
              
