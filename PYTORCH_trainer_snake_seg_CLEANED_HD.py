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
from functional.IO_func import *


from layers.UNet_pytorch_online import *
from layers.unet_nested import *
from layers.unet3_3D import *
from layers.switchable_BN import *

from losses_pytorch.HD_loss import *
 
import cIDice_metric as cID_metric
import cIDice_loss as cID_loss

import re
import sps
    
""" optional dataviewer if you want to load it """
# import napari
# with napari.gui_qt():
#     viewer = napari.view_image(seg_val)

torch.backends.cudnn.benchmark = True  ### set these options to improve speed
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    
    """" path to checkpoints """       
    #s_path = './(51) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Hd_loss_balance_repeat_MARCC/'; HD = 1; alpha = 1;
    
    
    # (1)
    #s_path = './(53) Checkpoint_unet_medium_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/'; HD = 1; alpha = 1;
    #s_path = './(54) Checkpoint_nested_unet_medium_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/'; HD = 1; alpha = 1;
    #s_path = './(57) Checkpoint_unet_medium_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step_transform_scale_Z/'; HD = 1; alpha = 1;  resize_z = 1
    #s_path = './(58) Checkpoint_unet_nested_medium_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step_transform_scale_Z/'; HD = 1; alpha = 1;  resize_z = 1
    #s_path = './(62) Checkpoint_unet_LARGE_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step_HISTORY/'; HD = 1; alpha = 1;
    
    resize_z = 0
    skeletonize = 0
    
    #s_path = './(63) Checkpoint_unet_LARGE_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step_SKEL/'; HD = 1; alpha = 1; skeletonize = 1
    # s_path = './(64) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step_HISTORICAL/'; HD = 1; alpha = 1; 
    
    
    # s_path = './(65) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_2_step_REAL_HISTORICAL/'; HD = 1; alpha = 1; 
    
    
    # s_path = './(66) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_4_step_REAL_HISTORICAL_NEURON/'; HD = 1; alpha = 1; 
    
    
    
    # s_path = './(80) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_HD_only_cytosol/'; HD = 1; alpha = 1; sps_bool = 0; im_type = 1; HISTORICAL = 0;
    # s_path = './(81) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_no_HD_only_cytosol/'; HD = 0; alpha = 0; sps_bool = 0; im_type = 1; HISTORICAL = 0;
    # s_path = './(82) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_HD_sps_only_cytosol/'; HD = 1; alpha = 1; sps_bool = 1; im_type = 1; HISTORICAL = 0;
    
    s_path = './(83) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_no_HD_sps_only_cytosol/'; HD = 0; alpha = 0; sps_bool = 1; im_type = 'c'; HISTORICAL = 0;
    
    #s_path = './(84) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_no_HD_sps_CYTOSOL_and_MYELIN/'; HD = 0; alpha = 0; sps_bool = 1; cytosol_only = 0;
    
    
    ### Add spatial weight kernel
    s_path = './(85) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_HD_sps_only_cytosol_sW_kernel/'; HD = 1; alpha = 1; sps_bool = 1; im_type = 'c'; sW_centroid = 1; HISTORICAL = 0;
    
    

    ### add cID_loss and metric
    s_path = './(86) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_HD_sps_only_cytosol_cID_loss/'; cID = 1; HD = 0; alpha = 0; sps_bool = 1; im_type = 'c'; sW_centroid = 0; HISTORICAL = 0;
    

    #s_path = './(86_fast) Checkpoint_unet_SMALL_filt_3x3_b4_type_dataset_NO_1st_im_sps_cytosol_cID_FAST/'; cID = 1; HD = 0; alpha = 0; sps_bool = 1; im_type = 'c'; sW_centroid = 0; HISTORICAL = 0;
        

    
    ### run with NEW Haussdorff loss
    s_path = './(87) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_HD_sps_only_cytosol_NEW_HD_loss/'; cID = 0; HD = 1; alpha = 1; sps_bool = 0; im_type = 'c'; sW_centroid = 0; HISTORICAL = 0;

    s_path = './(88) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_HD_sps_only_cytosol_NEW_HD_loss_YES_SPS/'; cID = 0; HD = 1; alpha = 1; sps_bool = 1; im_type = 'c'; sW_centroid = 0; HISTORICAL = 0;
        
       

    """ path to input data """
    # (2)
    
    #input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads/'; dataset = 'new crop pads'
    #tracker.alpha = 0.5
    
    #input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads seed 5/TRAINING FORWARD PROP ONLY SCALED crop pads seed 5/'; dataset = 'new crop pads'
    # input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads seed 2 COLORED 48 z/TRAINING FORWARD PROP seed 2 COLORED 48 z DATA/'; dataset = 'historical seed 2 z 48'
    
    input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING SCALED crop pads seed 4 COLORED 48 z DENSE LABELS/Training_snake_seg/'; dataset = 'full historical type seed 4 z 48 dataset'

    #input_path = 'E:/7) Bergles lab data/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads/'; 
    #input_path = '/lustre04/scratch/yxu233/TRAINING FORWARD PROP ONLY SCALED crop pads/';  dataset = 'new crop pads'

    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), 
    #                  seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop'),  
    #                  orig_idx= int(re.search('_origId_(.*)_eId', i).group(1)),
    #                  x = int(re.search('_x_(.*)_y_', i).group(1)),
    #                  y = int(re.search('_y_(.*)_z_', i).group(1)),
    #                  z = int(re.search('[^=][^a-z]_z_(.*)_type_', i).group(1)),     ### had to exclude anything that starts with "=0_z" b/c that shows up earlier
    #                  im_type = str(re.search('_type_(.*)_branch_', i).group(1)),    
    #                  filename= i.split('/')[-1].split('_origId')[0].replace(',', ''))
    #                  for i in images]
      
    examples = []
    for i in images:
        type_check = str(re.search('_type_(.*)_branch_', i).group(1))
                         
        if im_type == type_check:
            examples.append(dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), 
                             seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop'),  
                             orig_idx= int(re.search('_origId_(.*)_eId', i).group(1)),
                             x = int(re.search('_x_(.*)_y_', i).group(1)),
                             y = int(re.search('_y_(.*)_z_', i).group(1)),
                             z = int(re.search('[^=][^a-z]_z_(.*)_type_', i).group(1)),     ### had to exclude anything that starts with "=0_z" b/c that shows up earlier
                             im_type = str(re.search('_type_(.*)_branch_', i).group(1)),    
                             filename= i.split('/')[-1].split('_origId')[0].replace(',', '')))
        


    """ Also load in the all_tree_indices file """
    all_trees = []
    if HISTORICAL:
        tree_csv_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads seed 2 COLORED 48 z/'
        all_trees = load_all_trees(tree_csv_path)



    # ### REMOVE IMAGE 1 from training data
    idx_skip = []
    for idx, im in enumerate(examples):
        filename = im['input']
        if '1to1pair_b_series_t1_input' in filename:
            print('skip')
            idx_skip.append(idx)
    
    
    ### USE THE EXCLUDED IMAGE AS VALIDATION/TESTING
    examples_test = examples[0:len(idx_skip)]

    examples = [i for j, i in enumerate(examples) if j not in idx_skip]
          

    """ Shorten for over-fitting """
    # examples = examples[0:500]
    # examples_test = examples_test[0:1000]



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
    
    deep_sup = False
    if not onlyfiles_check:   ### if no old checkpoints found, start new network and tracker
 
        """ Hyper-parameters """
        
        switch_norm = False
        sp_weight_bool = 0
        #transforms = initialize_transforms(p=0.5)
        #transforms = initialize_transforms_simple(p=0.5)
        transforms = 0
        batch_size = 8;      
        test_size = 0.1  
        
        

        """ Initialize network """  
        kernel_size = 7
        pad = int((kernel_size - 1)/2)
        unet = UNet_online(in_channels=2, n_classes=2, depth=5, wf=4, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
                            batch_norm=True, batch_norm_switchable=switch_norm, up_mode='upsample')
        #unet = NestedUNet(num_classes=2, input_channels=2, deep_sup=deep_sup, padding=pad, batch_norm_switchable=switch_norm)
        #unet = UNet_3Plus(num_classes=2, input_channels=2, kernel_size=kernel_size, padding=pad)

        unet.train()
        unet.to(device)
        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Select loss function *** unimportant if using HD loss """
        if not HD and not cID:    
            loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        
        elif cID:
            loss_function = cID_loss.soft_dice_cldice(iter_=3, alpha=0.5, smooth = 1.)
                      
        else:         
            loss_function = 'Haussdorf'
            

        """ Select optimizer """
        lr = 1e-5; milestones = [20, 100]  # with AdamW slow down
        if not sps_bool:
            optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)
            """ Add scheduler """
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
            
        else:
            
            
            """ add useless scheduler """
            scheduler = torch.optim.lr_scheduler.MultiStepLR(torch.optim.AdamW(unet.parameters()), milestones, gamma=0.1, last_epoch=-1)
            
            """ Define SPS optimizer"""
            optimizer = sps.Sps(unet.parameters())

            
        """ initialize index of training set and validation set, split using size of test_size """
        #idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
        
        """ initialize training_tracker """
        idx_valid = idx_skip
        idx_train = counter
        
        tracker = tracker(batch_size, test_size, mean_arr, std_arr, idx_train, idx_valid, deep_sup=deep_sup, switch_norm=switch_norm, alpha=alpha, HD=HD,
                                          sp_weight_bool=sp_weight_bool, transforms=transforms, dataset=input_path, im_type=im_type, cID=cID)

        tracker.resize_z = resize_z

    else:             
        """ Find last checkpoint """       
        last_file = onlyfiles_check[-1]
        split = last_file.split('check_')[-1]
        num_check = split.split('.')
        checkpoint = num_check[0]
        checkpoint = 'check_' + checkpoint

        print('restoring weights from: ' + checkpoint)
        check = torch.load(s_path + checkpoint, map_location=lambda storage, loc: storage)
        #check = torch.load(s_path + checkpoint, map_location='cpu')
        #check = torch.load(s_path + checkpoint, map_location=device)
        
        tracker = check['tracker']
        
        unet = check['model_type']
        scheduler = check['scheduler_type']
        unet.load_state_dict(check['model_state_dict'])
        unet.to(device)
        
        
        """ OPTIMIZER HAS TO BE LOADED IN AFTER THE MODEL!!!"""
        if not sps_bool:
            optimizer = check['optimizer_type']
            optimizer.load_state_dict(check['optimizer_state_dict'])
        else:
            optimizer = sps.Sps(unet.parameters()) 
        
        scheduler.load_state_dict(check['scheduler'])     
        loss_function = check['loss_function']

        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Clean up checkpoint file """
        del check
        torch.cuda.empty_cache()

                

    #transforms = initialize_transforms_simple(p=0.5)

    """ Create datasets for dataloader """
    training_set = Dataset_tiffs_snake_seg(tracker.idx_train, examples, tracker.mean_arr, tracker.std_arr,
                                           sp_weight_bool=tracker.sp_weight_bool, transforms = tracker.transforms, resize_z=resize_z, skeletonize=skeletonize, all_trees=all_trees)
    val_set = Dataset_tiffs_snake_seg(tracker.idx_valid, examples_test, tracker.mean_arr, tracker.std_arr,
                                      sp_weight_bool=tracker.sp_weight_bool, transforms = 0, resize_z=resize_z, skeletonize=skeletonize, all_trees=all_trees)
    
    
    """ Create training and validation generators"""
    val_generator = data.DataLoader(val_set, batch_size=tracker.batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=True, drop_last = True)

    training_generator = data.DataLoader(training_set, batch_size=tracker.batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=True, drop_last=True)
         
    print('Total # training images per epoch: ' + str(len(training_set)))
    print('Total # validation images: ' + str(len(val_set)))
    

    """ Epoch info """
    train_steps_per_epoch = len(tracker.idx_train)/tracker.batch_size
    validation_size = len(tracker.idx_valid)
    epoch_size = len(tracker.idx_train)    
   
    
   
    
    """ Generate spatial weight matrix """
    def create_spatial_weight_mat(labels, edgeFalloff=10,background=0.01,approximate=True):    
           if approximate:   # does chebyshev
               dist1 = scipy.ndimage.distance_transform_cdt(labels)
               dist2 = scipy.ndimage.distance_transform_cdt(np.where(labels>0,0,1))    # sets everything in the middle of the OBJECT to be 0
                       
           else:   # does euclidean
               dist1 = scipy.ndimage.distance_transform_edt(labels, sampling=[1,1,1])
               dist2 = scipy.ndimage.distance_transform_edt(np.where(labels>0,0,1), sampling=[1,1,1])
               
           """ DO CLASS WEIGHTING instead of spatial weighting WITHIN the object """
           dist1[dist1 > 0] = 0.5
       
           dist = dist1+dist2
           attention = math.e**(1-dist/edgeFalloff) + background   # adds background so no loses go to zero
           
           
           """ TIGER REMOVED THIS SO NOT SO EXTREME - Jan. 9th, 2021 """
           #attention /= np.average(attention)
           
           return np.reshape(attention,labels.shape)
    
    center_im = np.zeros([48, 80, 80])
    center_im[int(center_im.shape[0]/2 - 1), int(center_im.shape[1]/2 - 1), int(center_im.shape[2]/2 - 1)] = 1
    
    spatial_center = create_spatial_weight_mat(center_im, edgeFalloff=20,background=0.01,approximate=False)
    torch_spatial_center = torch.tensor(spatial_center, dtype = torch.float, device=device, requires_grad=False)    
    

   
    """ Start training """
    for cur_epoch in range(len(tracker.train_loss_per_epoch), 10000): 
        
         """ check and plot params during training """             
         for param_group in optimizer.param_groups:
               #tracker.alpha = 0.5
               #param_group['lr'] = 1e-6   # manually sets learning rate
               if not sps_bool:
                    cur_lr = param_group['lr']
                    tracker.lr_plot.append(cur_lr)
               else:  tracker.lr_plot.append(0)
               tracker.print_essential()

         unet.train()  ### set PYTORCH to training mode

         start_time_epoch = time.perf_counter();
         loss_train = 0; jacc_train = []; ce_train = 0; dc_train = 0; hd_train = 0;
         iter_cur_epoch_train = 0; starter = 0;
         for batch_x, batch_y, spatial_weight in training_generator:
                 ### Test speed for debug
                 starter += 1
                 if starter == 2:  start = time.perf_counter()
                 if starter == 50: stop = time.perf_counter(); diff = stop - start; print(diff);  #break;
                     
                 """ Load data ==> shape is (batch_size, num_channels, depth, height, width)
                      (1) converts to Tensor
                      (2) normalizes + applies other transforms on GPU   ***INPUT LABELS MUST BE < 255??? or else get CudNN error
                 """
                
                 inputs, labels = transfer_to_GPU(batch_x, batch_y, device, tracker.mean_arr, tracker.std_arr)
                 inputs = inputs[:, 0, ...]

                 # PRINT OUT THE SHAPE OF THE INPUT
                 if iter_cur_epoch_train == 0:
                     print('input size is' + str(batch_x.shape))

                
                 """ initialize each iteration """
                 optimizer.zero_grad()    ### zero the parameter gradients
                 output_train = unet(inputs)  ### forward + backward + optimize
                                  
                 """ calculate loss: includes HD loss functions """
                 
                 if tracker.cID:

                     ### USE SOFTMAX, and NOT argmax because argmax creates some discontinuous skeleton
                     outputs_soft = F.softmax(output_train, dim=1)
                     loss = loss_function(labels.type(torch.float32).unsqueeze(1), outputs_soft[:, 1, :, :, :].unsqueeze(1))
                     # outputs_argm = torch.argmax(output_train, dim=1)
                     # loss = loss_function(labels.type(torch.float32).unsqueeze(1), outputs_argm.type(torch.float32).unsqueeze(1))
                     
                     #loss_function = cID_loss.soft_cldice(iter_=3, smooth = 0)
                     #outputs_argm = torch.argmax(output_train, dim=1)
                     #val = loss_function(labels[1].type(torch.float32).unsqueeze(0).unsqueeze(0), outputs_soft[1, 1, :, :, :].unsqueeze(0).unsqueeze(0))
                     #val = loss_function(labels[1].type(torch.float32).unsqueeze(0).unsqueeze(0), outputs_argm[1, :, :, :].type(torch.float32).unsqueeze(0).unsqueeze(0))

                     # y_true = labels[1].type(torch.float32).unsqueeze(0).unsqueeze(0)
                     # #y_pred =  outputs_soft[1, 1, :, :, :].unsqueeze(0).unsqueeze(0)
                     # y_pred = outputs_argm[1, :, :, :].type(torch.float32).unsqueeze(0).unsqueeze(0)
                     
                     # #plot_max(labels[0].detach().cpu().numpy())
                     # skel_pred = soft_skel(y_pred, 3)
                     # skel_true = soft_skel(y_true, 3)
                     # tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,0,:,:,:]))/(torch.sum(skel_pred[:,0,:,:,:]))    
                     # tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,0,:,:,:]))/(torch.sum(skel_true[:,0,:,:,:]))    
                     # cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)                 
                             

                       
                       
                 elif tracker.HD:
                     loss, tracker, ce_train, dc_train, hd_train = compute_HD_loss(output_train, labels, tracker.alpha, tracker, 
                                                                                    ce_train, dc_train, hd_train, val_bool=0,
                                                                                    spatial_weight=sW_centroid, weight_arr=torch_spatial_center)
                     
                     
                     """ can also check out the old HD function """
                     # loss, tracker, ce_train, dc_train, hd_train = compute_HD_loss_OLD(output_train, labels, tracker.alpha, tracker, 
                     #                                                               ce_train, dc_train, hd_train, val_bool=0)
                                          
                     
                 else:
                     if deep_sup:   ### IF DEEP SUPERVISION
                        # compute output
                        loss = 0
                        for output in output_train:
                             loss += loss_function(output, labels)
                        loss /= len(output_train)
                        output_train = output_train[-1]  # set this so can eval jaccard later
                   
                     else:   ### IF NORMAL LOSS CALCULATION
                        loss = loss_function(output_train, labels)
                        if torch.is_tensor(spatial_weight):   ### WITH SPATIAL WEIGHTING
                             spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                             weighted = loss * spatial_tensor
                             loss = torch.mean(weighted)
                             
                              
                        else:  ### NO WEIGHTING AT ALL
                             loss = torch.mean(loss)   
                                       
                 """ update and step trainer """
                 loss.backward()
                 if sps_bool:
                     optimizer.step(loss=loss)
                 else:
                     optimizer.step()
               
                 """ Training loss """
                 tracker.train_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                 loss_train += loss.cpu().data.numpy()
                
   
                 """ Calculate Jaccard on GPU """                 
                 #jacc = jacc_eval_GPU_torch(output_train, labels)
                 #jacc = jacc.cpu().data.numpy()
                 jacc = cID_metric_eval_CPU(output_train, labels=batch_y)
                               
                 jacc_train.append(jacc) # Training jacc
                 tracker.train_jacc_per_batch.append(jacc)
   
                 tracker.iterations = tracker.iterations + 1       
                 iter_cur_epoch_train += 1
                 if tracker.iterations % 100 == 0:
                     print('Trained: %d' %(tracker.iterations))


                 """ Plot for ground truth """
                 # output_train = output_train.cpu().data.numpy()            
                 # output_train = np.moveaxis(output_train, 1, -1)              
                 # seg_train = np.argmax(output_train[0], axis=-1)  
                  
                 # # convert back to CPU
                 # batch_x = batch_x.cpu().data.numpy() 
                 # batch_y = batch_y.cpu().data.numpy() 
 
                 # plot_trainer_3D_PYTORCH_snake_seg(seg_train, seg_train, batch_x[0], batch_x[0], batch_y[0], batch_y[0],
                 #                            s_path, iterations, plot_depth=8)



                 """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
                 #if tracker.iterations % 10000 == 0:
                     
                 #if tracker.iterations % train_steps_per_epoch == 0:

                     
                     
                     
                    
         tracker.train_loss_per_epoch.append(loss_train/iter_cur_epoch_train)
         tracker.train_jacc_per_epoch.append(np.nanmean(jacc_train))     
              
           
         loss_val = 0; jacc_val = []; val_idx = 0;
         iter_cur_epoch = 0;  ce_val = 0; dc_val = 0; hd_val = 0;  hd_value = 0;
         if cur_epoch % validate_every_num_epochs == 0:
               
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
                          if tracker.cID:
        
                              ### USE SOFTMAX, and NOT argmax because argmax creates some discontinuous skeleton
                              outputs_soft = F.softmax(output_val, dim=1)
                              loss = loss_function(labels_val.type(torch.float32).unsqueeze(1), outputs_soft[:, 1, :, :, :].unsqueeze(1))
                              # outputs_argm = torch.argmax(output_val, dim=1)
                              # loss = loss_function(labels_val.type(torch.float32).unsqueeze(1), outputs_argm.type(torch.float32).unsqueeze(1))
                            
                          elif tracker.HD:
                              loss, tracker, ce_val, dc_val, hd_val = compute_HD_loss(output_val, labels_val, tracker.alpha, tracker, 
                                                                                             ce_val, dc_val, hd_val, val_bool=1,
                                                                                             spatial_weight=sW_centroid, weight_arr=torch_spatial_center)
                      
                        
                        
                              
                              """ can also check out the old HD function """
                              # loss, tracker, ce_val, dc_val, hd_val = compute_HD_loss_OLD(output_val, labels_val, tracker.alpha, tracker, 
                              #                                                               ce_val, dc_val, hd_val, val_bool=1)
                                      
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
                                  # elif dist_loss:
                                  #        loss  # do not do anything if do not need to reduce
                                      
                                  else:
                                         loss = torch.mean(loss)  
  
                          """ Training loss """
                          tracker.val_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                          loss_val += loss.cpu().data.numpy()
                                           
                          """ Calculate jaccard on GPU """
                          #jacc = jacc_eval_GPU_torch(output_val, labels_val)
                          #jacc = jacc.cpu().data.numpy()
                          jacc = cID_metric_eval_CPU(output_val, labels=batch_y_val)
                          
                          jacc_val.append(jacc)
                          tracker.val_jacc_per_batch.append(jacc)   



                          """ HD_metric """
                          # outputs_argm = torch.argmax(output_train, dim=1)
                          # hd_metric = HD_metric.HausdorffDistance()
                          # hd_m = hd_metric.compute(outputs_argm.unsqueeze(1), labels.unsqueeze(1))
                          # hd_value += hd_m.cpu().data.numpy()
  
    
                          val_idx = val_idx + tracker.batch_size
                          print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                          iter_cur_epoch += 1
  
                          #if starter == 50: stop = time.perf_counter(); diff = stop - start; print(diff);  #break;
  
                             
                     tracker.val_loss_per_eval.append(loss_val/iter_cur_epoch)
                     tracker.val_jacc_per_eval.append(np.nanmean(jacc_val))       
                     
                     """ Add to scheduler to do LR decay
                             skipped if doing sps_bool!
                     """
                     if not sps_bool:
                         scheduler.step()
                         
         """ calculate new alpha for next epoch """   
         if tracker.HD:
               tracker.alpha = alpha_step(ce_train, dc_train, hd_train, iter_cur_epoch_train)
               
               #tracker.alpha = 0.5
  

         """ Plot metrics every epoch """      
         if cur_epoch % plot_every_num_epochs == 0:       
                
               
                """ Plot metrics in tracker """
                plot_tracker(tracker, s_path)
                
                """ custom plot """
                output_train = output_train.cpu().data.numpy()            
                output_train = np.moveaxis(output_train, 1, -1)              
                seg_train = np.argmax(output_train[0], axis=-1)  
                
                # convert back to CPU
                batch_x = batch_x.cpu().data.numpy() 
                batch_y = batch_y.cpu().data.numpy() 
                batch_x_val = batch_x_val.cpu().data.numpy()
                
                
                batch_y_val = batch_y_val.cpu().data.numpy() 
                output_val = output_val.cpu().data.numpy()            
                output_val = np.moveaxis(output_val, 1, -1)       
                seg_val = np.argmax(output_val[0], axis=-1)  
                
                plot_trainer_3D_PYTORCH_snake_seg(seg_train, seg_val, batch_x[0], batch_x_val[0], batch_y[0], batch_y_val[0],
                                         s_path, tracker.iterations, plot_depth=8)
                                               
                
         """ To save tracker and model (every x iterations) """
         if cur_epoch % save_every_num_epochs == 0:           
                 stop_time_epoch = time.perf_counter(); diff = stop_time_epoch - start_time_epoch; print(diff); 
                
                 save_name = s_path + 'check_' +  str(tracker.iterations)               
                 torch.save({
                  'tracker': tracker,
  
                  'model_type': unet,
                  'optimizer_type': optimizer,
                  'scheduler_type': scheduler,
                  
                  'model_state_dict': unet.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(),
                  'scheduler': scheduler.state_dict(),
                  'loss_function': loss_function,  
                  
                  }, save_name)



 
  
         """ check and plot params during training """             
         # for param_group in optimizer.param_groups:
         #       #tracker.alpha = 0.5
         #       #param_group['lr'] = 1e-6   # manually sets learning rate
         #       if not sps_bool:
         #           cur_lr = param_group['lr']
         #           tracker.lr_plot.append(cur_lr)
         #       tracker.print_essential()
  
         # unet.train()  ### set PYTORCH to training mode
  
         # start_time_epoch = time.perf_counter();
         # loss_train = 0; jacc_train = 0; ce_train = 0; dc_train = 0; hd_train = 0;
         # iter_cur_epoch = 0; starter = 0;                     
           
                

                
               
              
              
