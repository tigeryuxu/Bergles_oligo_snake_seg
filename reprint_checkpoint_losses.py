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
import matplotlib.pyplot as plt

""" Libraries to load """
import numpy as np
from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
import glob, os
import datetime
import time
import bcolz
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from UNet_pytorch import *
from UNet_pytorch_online import *
from PYTORCH_dataloader import *

from sklearn.model_selection import train_test_split

from losses_pytorch.boundary_loss import DC_and_HDBinary_loss, BDLoss, HDDTBinaryLoss
from losses_pytorch.dice_loss import FocalTversky_loss, DC_and_CE_loss
from losses_pytorch.focal_loss import FocalLoss

import kornia

from unet_nested import *
from unet3_3D import *
from switchable_BN import *
#import lovasz_losses as L

import napari
# with napari.gui_qt():
#     viewer = napari.view_image(seg_val)

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    """" Input paths """    
    s_path = './(1) Checkpoint_PYTORCH/'
    s_path = './(2) Checkpoint_PYTORCH_spatial_weight/'
    s_path = './(3) Checkpoint_SGD_spatial/'
    s_path = './(4) Checkpoint_AdamW_spatial/'
    #s_path = './(5) Checkpoint_AdamW/'
    #s_path = './(6) Checkpoint_AdamW_FOCALLOSS/'
    #s_path = './(7) Checkpoint_AdamW_spatial_batch_1/'
    #s_path = './(8) Checkpoint_SGD_cyclic_batch_norm/'
    s_path = './(9) Checkpoint_AdamW_batch_norm/'
    #s_path = './(10) Checkpoint_AdamW_batch_norm_SWITCH/'
    #s_path = './(11) Checkpoint_SGD_batch_norm/'
    #s_path = './(12) Checkpoint_AdamW_batch_norm_CYCLIC/'
    #s_path = './(13) Checkpoint_AdamW_batch_norm_DC_and_HDBinary_loss/'
    #s_path = './(14) Checkpoint_AdamW_batch_norm_HDBinary_loss/'
    #s_path = './(15) Checkpoint_AdamW_batch_norm_SPATIALW/'
    #s_path = './(16) Checkpoint_AdamW_batch_norm_DICE_CE/'
    #s_path = './(17) Checkpoint_AdamW_batch_norm_FOCALLOSS/'
    #s_path = './(18) Checkpoint_AdamW_batch_norm_SPATIALW_CYCLIC/'
    #s_path = './(19) Checkpoint_AdamW_batch_norm_HD_and_CE/'
    #s_path = './(20) Checkpoint_AdamW_batch_norm_7x7/'
    #s_path = './(21) Checkpoint_AdamW_batch_norm_3x_branched/'
    #s_path = './(22) Checkpoint_AdamW_batch_norm_3x_branched_SPATIALW_e-6/'
    #s_path = './(23) Checkpoint_nested_unet/'
    #s_path = './(24) Checkpoint_nested_unet_SPATIALW/'
    #s_path = './(25) Checkpoint_nested_unet_SPATIALW_deepsupervision/'
    # s_path = './(26) Checkpoint_AdamW_batch_norm_SPATIALW_transforms/'
    # s_path = './(27) Checkpoint_AdamW_batch_norm_spatialW_1e-6/'
    #s_path = './(28) Checkpoint_nested_unet_SPATIALW_complex/'
    #s_path = './(29) Checkpoint_nested_unet_NO_SPATIALW/'
    #s_path = './(30) Checkpoint_nested_unet_SPATIALW_simple/'
    #s_path = './(31) Checkpoint_nested_unet_SPATIALW_complex_3x3/'
    #s_path = './(32) Checkpoint_nested_unet_SPATIALW_complex_deep_supervision/'
    #s_path = './(33) Checkpoint_UNET3_PLUS_SPATIALW_simpler/'
    #s_path = './(34) Checkpoint_nested_unet_SPATIALW_complex_RETRAIN/'
    s_path = './(35) Checkpoint_nested_unet_SPATIALW_complex_SWITCH_NORM/'
    #s_path = './(36) Checkpoint_nested_unet_SPATIALW_complex_SWITCH_NORM_medium/'
    #s_path = './(37) Checkpoint_nested_unet_SPATIALW_complex/'
    
    #s_path = './(38) Checkpoint_nested_unet_SPATIALW_complex_batch_4_NEW_DATA/'
    
    #s_path = './(39) Checkpoint_nested_unet_SPATIALW_simple_b4_NEW_DATA_SWITCH_NORM/'
    
    
    s_path = './(40) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad/'; deep_supervision = False

    #s_path = './(41) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_crop_pad/'
    
    #s_path = './(42) Checkpoint_nested_unet_SPATIALW_medium_b4_SWITCH_NORM_crop_pad/'
    
    
    #s_path = './(43) Checkpoint_nested_unet_SPATIALW_medium_b4_SWITCH_NORM_crop_pad_deep_sup/';  deep_supervision = True
    
    

    
    
    """ Add Hausdorff + CE??? or + DICE???  + spatial W???"""
    #input_path = '/media/user/storage/Data/(1) snake seg project/Train_SNAKE_SEG_scaled_cleaned/'; dataset = 'old' ### OLD DATA 80 x 80 x16
    #input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED/'; dataset = 'new'  ### NEW DATA 80 x 80 x 32
    input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads/'; dataset = 'new crop pads'
    
    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
    onlyfiles_check.sort(key = natsort_key1)
 

    # """ load mean and std """  
    mean_arr = np.load('./normalize/' + 'mean_VERIFIED.npy')
    std_arr = np.load('./normalize/' + 'std_VERIFIED.npy')   

    num_workers = 2;
 
    save_every_num_epochs = 1;
    plot_every_num_epochs = 1;
    validate_every_num_epochs = 1;      

    #dist_loss = 0
    dist_loss = 0
    both = 0
    branch_bool = 0
    
    
   
    switch_norm = True 
    
    deep_supervision = False
    


    """ Restore per batch """
    train_loss_per_batch = []
    train_jacc_per_batch = []
    val_loss_per_batch = []
    val_jacc_per_batch = []
 
    """ Restore per epoch """
    train_loss_per_epoch = []
    train_jacc_per_epoch = []
    val_loss_per_eval = []
    val_jacc_per_eval = []
    
    """ Find last checkpoint """      
    for check_file in onlyfiles_check:
        last_file = check_file
        split = last_file.split('check_')[-1]
        num_check = split.split('.')
        checkpoint = num_check[0]
        checkpoint = 'check_' + checkpoint
    
        print('restoring weights from: ' + checkpoint)
        check = torch.load(s_path + checkpoint, map_location=lambda storage, loc: storage)
        #check = torch.load(s_path + checkpoint, map_location='cpu')
        #check = torch.load(s_path + checkpoint, map_location=device)
        cur_epoch = check['cur_epoch']
        iterations = check['iterations']
        idx_train = check['idx_train']
        idx_valid = check['idx_valid']
        
        
        unet = check['model_type']
        optimizer = check['optimizer_type']
        scheduler = check['scheduler_type']
        
        
        unet.load_state_dict(check['model_state_dict'])
        unet.to(device)
        optimizer.load_state_dict(check['optimizer_state_dict'])
        scheduler.load_state_dict(check['scheduler'])
      
        # """ Restore per batch """
        # train_loss_per_batch = check['train_loss_per_batch']
        # train_jacc_per_batch = check['train_jacc_per_batch']
        # val_loss_per_batch = check['val_loss_per_batch']
        # val_jacc_per_batch = check['val_jacc_per_batch']
     
        # """ Restore per epoch """
        # train_loss_per_epoch = check['train_loss_per_epoch']
        # train_jacc_per_epoch = check['train_jacc_per_epoch']
        # val_loss_per_eval = check['val_loss_per_eval']
        # val_jacc_per_eval = check['val_jacc_per_eval']
     
        plot_sens = check['plot_sens']
        plot_sens_val = check['plot_sens_val']
        plot_prec = check['plot_prec']
        plot_prec_val = check['plot_prec_val']
     
        lr_plot = check['lr_plot']
    
        # newly added
        mean_arr = check['mean_arr']
        std_arr = check['std_arr']
        
        batch_size = check['batch_size']
        sp_weight_bool = check['sp_weight_bool']
        
        #sp_weight_bool = 0
        
        loss_function = check['loss_function']
        transforms = check['transforms']
    
    
        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        
        
        """ Clean up checkpoint file """
        del check
        torch.cuda.empty_cache()
        
        resume = 1
     
        
     
        
     
        batch_size = 1   
     
        
     
        
     
        
     
    
        """ Load filenames from tiff """
        images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
        images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
        examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop')) for i in images]
        counter = list(range(len(examples)))
        
        all_branch_idx = np.load('./normalize/all_branch_idx.npy')
        
        if not resume:
            idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
            
            
            ### DOUBLES anything that is a branch_idx
            if branch_bool:
                match = set(idx_train) & set(all_branch_idx)
                idx_train = idx_train + list(match) + list(match)
                
    
        """ Create datasets for dataloader """
        training_set = Dataset_tiffs_snake_seg(idx_train, examples, mean_arr, std_arr, sp_weight_bool=sp_weight_bool, transforms = transforms)
        val_set = Dataset_tiffs_snake_seg(idx_valid, examples, mean_arr, std_arr, sp_weight_bool=sp_weight_bool, transforms = 0)
        
        """ Create training and validation generators"""
        val_generator = data.DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                        pin_memory=True, drop_last = True)
    
        training_generator = data.DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                          pin_memory=True, drop_last=True)
             
        
        
        print('Total # training images per epoch: ' + str(len(training_set)))
        print('Total # validation images: ' + str(len(val_set)))
        
    
        """ Epoch info """
        train_steps_per_epoch = len(idx_train)/batch_size
        validation_size = len(idx_valid)
        epoch_size = len(idx_train)    
       
        """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
        loss_val = 0; jacc_val = 0
        precision_val = 0; sensitivity_val = 0; val_idx = 0;
        iter_cur_epoch = 0; 
        if cur_epoch % validate_every_num_epochs == 0:
            
             with torch.set_grad_enabled(False):  # saves GPU RAM
                  unet.eval()
                  for batch_x_val, batch_y_val, spatial_weight in val_generator:
                      
                           
                       """ Transfer to GPU to normalize ect... """
                       inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, mean_arr, std_arr)
            
                       inputs_val = inputs_val[:, 0, ...]   
            
                       # forward pass to check validation
                       output_val = unet(inputs_val)
    
                       if dist_loss:  # for distance loss functions
                           labels_val = labels_val.unsqueeze(1)
                           labels_val = labels_val.permute(0, 1, 3, 4, 2)
                           output_val = output_val.permute(0, 1, 3, 4, 2)                            
                                   
                       """ calculate loss """
                       if deep_supervision:
                                           
                           # compute output
                           loss = 0
                           for output in output_val:
                                loss += loss_function(output, labels_val)
                           loss /= len(output_val)
                           
                           output_val = output_val[-1]  # set this so can eval jaccard later
                       
                       else:
                       
                           loss = loss_function(output_val, labels_val)
               
                       if both:
                           loss_ce = loss_function_2(output_val.permute(0, 1, 4, 2, 3), labels_val.permute(0, 1, 4, 2, 3).squeeze())
                           
                           loss = loss + loss_ce
    
    
    
                       if torch.is_tensor(spatial_weight):
                              spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                              weighted = loss * spatial_tensor
                              loss = torch.mean(weighted)
                       elif dist_loss:
                              loss  # do not do anything if do not need to reduce
                           
                       else:
                              loss = torch.mean(loss)  
                              
                              
                       if dist_loss:  # for distance loss functions
                           labels_val = labels_val.permute(0, 1, 4, 2, 3)
                           output_val = output_val.permute(0, 1, 4, 2, 3)
       
                 
                       """ Training loss """
                       val_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                       loss_val += loss.cpu().data.numpy()
                                        
                       """ Calculate jaccard on GPU """
                       jacc = jacc_eval_GPU_torch(output_val, labels_val)
                       jacc = jacc.cpu().data.numpy()
                       
                       jacc_val += jacc
                       val_jacc_per_batch.append(jacc)
    
                       """ Convert back to cpu """                                      
                       output_val = output_val.cpu().data.numpy()            
                       output_val = np.moveaxis(output_val, 1, -1)
                       
                       """ Calculate sensitivity + precision as other metrics ==> only ever on ONE IMAGE of a batch"""
                       batch_y_val = batch_y_val.cpu().data.numpy() 
                       seg_val = np.argmax(output_val[0], axis=-1)  
    
                       val_idx = val_idx + batch_size
                       print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                       iter_cur_epoch += 1
    
                          
                  val_loss_per_eval.append(loss_val/iter_cur_epoch)
                  val_jacc_per_eval.append(jacc_val/iter_cur_epoch)       
                  
    
                 
                  """ Add to scheduler to do LR decay """
                  scheduler.step()
                      
             if cur_epoch % plot_every_num_epochs == 0:       
                  plot_metric_fun(train_jacc_per_epoch, val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32)
                  plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
                  
                  plot_metric_fun(train_loss_per_epoch, val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
                  plt.figure(33); plt.yscale('log'); plt.savefig(s_path + 'loss_per_epoch.png')          
    
    
                  plot_metric_fun(train_loss_per_epoch, val_loss_per_eval, class_name='', metric_name='loss', plot_num=40)
                  plt.figure(40); plt.savefig(s_path + 'loss_per_epoch_NO_LOG.png')         
                       
                  """ VALIDATION LOSS PER BATCH??? """
                  plot_cost_fun(val_loss_per_batch, val_loss_per_batch)                   
                  plt.figure(18); plt.savefig(s_path + '_VAL_global_loss_VAL.png')
                  plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
                  plt.figure(25); plt.savefig(s_path + '_VAL_global_loss_LOG_VAL.png')
                  plt.close('all')
                    
                  
                  # plot_metric_fun(lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
                  # plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 
    
                  """ Plot negative loss """
                  # if loss < 0:
                  #     plot_metric_fun(train_loss_per_epoch[cur_epoch - 5: -1], val_loss_per_eval[cur_epoch - 5: -1], class_name='', metric_name='loss', plot_num=36)
                  #     plt.figure(36); plt.savefig(s_path + 'loss_per_epoch_NEGATIVE.png')     
                                    
    
                  """ Plot metrics per batch """                
                  # plot_metric_fun(train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
                  # plt.figure(34); plt.savefig(s_path + 'Jaccard_per_batch.png')
                                    
                  # plot_cost_fun(train_loss_per_batch, train_loss_per_batch)                   
                  # plt.figure(18); plt.savefig(s_path + 'global_loss.png')
                  # plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
                  # plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
                  # plt.close('all')
                                     
                  # plot_depth = 8
                  #output_train = output_train.cpu().data.numpy()            
                  #output_train = np.moveaxis(output_train, 1, -1)              
                  #seg_train = np.argmax(output_train[0], axis=-1)  
                  
                  # convert back to CPU
                  #batch_x = batch_x.cpu().data.numpy() 
                  #batch_y = batch_y.cpu().data.numpy() 
                  #batch_x_val = batch_x_val.cpu().data.numpy()
                  
                  #plot_trainer_3D_PYTORCH_snake_seg(seg_train, seg_val, batch_x[0], batch_x_val[0], batch_y[0], batch_y_val[0],
                  #                         s_path, iterations, plot_depth=plot_depth)
                                                 
                  
              
              