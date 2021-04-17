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
import tifffile


import cIDice_metric as cID_metric
import cIDice_loss as cID_loss

import Hausdorff_metric as HD_metric


""" optional dataviewer if you want to load it """
# import napari
# with napari.gui_qt():
#     viewer = napari.view_image(seg_val)

torch.backends.cudnn.benchmark = True  ### set these options to improve speed
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    """" path to checkpoints """       
    # s_path = './(51) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Hd_loss_balance_repeat_MARCC/'; HD = 1; alpha = 1;
    # s_path = './(52) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Hd_loss_balance_NO_1st_im/'; dilation = 1; deep_supervision = False; tracker = 1;

    # s_path = './(53) Checkpoint_unet_medium_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/'; HD = 1; alpha = 1;
    
    s_path = './(85) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_HD_sps_only_cytosol_sW_kernel/';  HD = 1; alpha = 1;
    
    
    """ path to input data """
    input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads/'; dataset = 'new crop pads'
    #input_path = 'E:/7) Bergles lab data/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads/'; 

    #input_path = '/lustre04/scratch/yxu233/TRAINING FORWARD PROP ONLY SCALED crop pads/';  dataset = 'new crop pads'


    input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING SCALED crop pads seed 4 COLORED 48 z DENSE LABELS/Training_snake_seg/'; dataset = 'full historical type seed 4 z 48 dataset'


    im_type = 'c'

    """ Load filenames from tiff """
    # images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    # images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    # examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop')) for i in images]
    
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
            
            
            
    
    """ Also load in full images """
    full_input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/'
    images_full = glob.glob(os.path.join(full_input_path,'*_input.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images_full.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples_full = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop')) for i in images]
    
    
    input_im = tifffile.imread(images_full[0])
        
    
    
    
    # ### REMOVE IMAGE 1 from training data
    idx_skip = []
    for idx, im in enumerate(examples):
        filename = im['input']
        if '1to1pair_b_series_t1_input' in filename:
            print('skip')
            idx_skip.append(idx)
    
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
    
    tracker.idx_valid = idx_skip   ### IF ONLY WANT 
    
    
    tracker.idx_train = []

    #tracker.batch_size = 1
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

    print(onlyfiles_check)
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
       
        loss_val = 0; jacc_val = []; val_idx = 0;
        iter_cur_epoch = 0;  ce_val = 0; dc_val = 0; hd_val = 0; hd_value = []
        
        all_cID_3D = []; all_cID_2D = []
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
                               
                       
                       if hasattr(tracker, 'cID') and tracker.cID:
      
                            ### USE SOFTMAX, and NOT argmax because argmax creates some discontinuous skeleton
                            outputs_soft = F.softmax(output_val, dim=1)
                            loss = cID_loss_function(labels_val.type(torch.float32).unsqueeze(1), outputs_soft[:, 1, :, :, :].unsqueeze(1))
                               
                       elif tracker.HD:
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
                       #jacc = jacc_eval_GPU_torch(output_val, labels_val)
                       #jacc = jacc.cpu().data.numpy()
                       jacc = cID_metric_eval_CPU(output_val, labels=batch_y_val)
                       
                       jacc_val.append(jacc)
                       tracker.val_jacc_per_batch.append(jacc)   

             

                       """ HD_metric """
                       outputs_argm = torch.argmax(output_val, dim=1)
                       hd_metric = HD_metric.HausdorffDistance()
                       hd_m = hd_metric.compute(outputs_argm.unsqueeze(1), labels_val.unsqueeze(1))
                       hd_value.append(hd_m.cpu().data.numpy())
  
        


                       val_idx = val_idx + tracker.batch_size
                       print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                       

                       
                       iter_cur_epoch += 1
   
                       #if starter == 50: stop = time.perf_counter(); diff = stop - start; print(diff);  #break;
                       
                       
                       """ check each image and all metrics """
                       batch_x_val = batch_x_val.cpu().data.numpy()
                         
                         
                       batch_y_val = batch_y_val.cpu().data.numpy() 
                       output_val = output_val.cpu().data.numpy()            
                       output_val = np.moveaxis(output_val, 1, -1)       
                       seg_val = np.argmax(output_val[0], axis=-1)  
                       
                       
                       
                       # input_3D = batch_x_val[0][0]
                       # truth_3D = batch_y_val[0]
                       # seg_3D = seg_val
                       # #plot_max(truth_3D + seg_3D)
                       
                       
                       # ### DEBUG: plot
                       # #plt.close('all')
                       # in_im = plot_max(batch_x_val[0][0], plot=0)
                       # truth_im = plot_max(batch_y_val[0], plot=0)
                       # seg_im = plot_max(seg_val, plot=0)
                                              
                       # # plt.figure()
                       # # plt.subplot(1, 3, 1); plt.imshow(in_im)
                       # # if len(tracker.val_dc_pb) > 0:
                       # #     plt.title('DC: ' + str(np.round(tracker.val_dc_pb[-1], 4)))
                      
                       
                       # # plt.subplot(1, 3, 2); plt.imshow(truth_im); 
                       # # if len(tracker.val_jacc_per_batch) > 0:
                       # #     plt.title('Jacc: ' + str(np.round(tracker.val_jacc_per_batch[-1], 4)))
                           
                       
                       # cID_val_2D = cID_metric.clDice(truth_im, seg_im)
                       # cID_val_3D = cID_metric.clDice(truth_3D, seg_3D)
                       
                       # all_cID_3D.append(cID_val_3D); all_cID_2D.append(cID_val_2D); 
                       

                       
                       # # cID_loss_val = cID_loss.soft_dice_cldice(iter_=3, alpha=0.5, smooth = 1.)
                       
                       
                       # # cID_loss_val(y_true=truth_im, y_pred=seg_im)

                       # # plt.subplot(1, 3, 3); plt.imshow(seg_im);  
                       # # if len(tracker.val_hd_pb) > 0:
                       # #     plt.title('HD: ' + str(np.round(tracker.val_hd_pb[-1], 4))  + '\n' + 'cID_metric: ' + str(round(cID_val_3D, 4))
                                                                         
                       # #               )
 
                       # ### DEBUG:
                       #  # plt.figure(); plt.scatter(all_cID_3D, all_cID_2D, s=1); plt.xlabel('cID 3D'); plt.ylabel('cID 2D')
                       #  # plt.figure(); plt.scatter(all_cID_3D, np.vstack(tracker.val_dc_pb), s=1); plt.xlabel('cID 3D'); plt.ylabel('DICE')                       
                       #  # plt.figure(); plt.scatter(all_cID_3D, np.vstack(tracker.val_hd_pb), s=1); plt.xlabel('cID 3D'); plt.ylabel('HD')     
                       
                       #  # plt.figure(); plt.scatter(all_cID_3D, np.vstack(tracker.val_jacc_per_batch), s=1); plt.xlabel('cID 3D'); plt.ylabel('JACCARD')    
                        
                       #  # plt.figure(); plt.hist(all_cID_3D); plt.xlabel('cID 3D'); plt.xlim([0, 1])
                       #  # plt.figure(); plt.hist(np.vstack(tracker.val_dc_pb)); plt.xlabel('DICE'); plt.xlim([0, 1])                
                       #  # plt.figure(); plt.hist(np.vstack(tracker.val_hd_pb)); plt.xlabel('HD')     
                       
                       #  # plt.figure(); plt.hist(np.vstack(tracker.val_jacc_per_batch)); plt.xlabel('JACCARD')    
                                                

                        
                       print('num_tested')
                      
                       """ Early stop """
                       # if val_idx > 500:
                       #     break
                       #zzz
                       
                        
                  #print('mean cID 3D: ' + str(np.nanmean(all_cID_3D)))
                  print('mean DICE: ' + str(np.mean(np.vstack(tracker.val_dc_pb))))
                  print('mean HD: ' + str(np.mean(np.vstack(tracker.val_hd_pb))))
                  print('mean cID_3D: ' + str(np.nanmean(np.vstack(tracker.val_jacc_per_batch))))   
                          
                  tracker.val_loss_per_eval.append(loss_val/iter_cur_epoch)
                  tracker.val_jacc_per_eval.append(np.nanmean(jacc_val))       
                  tracker.val_ce_pb.append(np.nanmean(hd_value))
                                   
                  
                  """ Add to scheduler to do LR decay """
                  #scheduler.step()
                 
        """ Plot metrics every epoch """      
        if tracker.cur_epoch % plot_every_num_epochs == 0:       
            
            plot_metric_fun(tracker.train_jacc_per_epoch, tracker.val_jacc_per_eval, class_name='', metric_name='cID', plot_num=32)
            plt.figure(32); plt.savefig(s_path + '_RETRAIN_cID.png')
            
            plot_metric_fun(tracker.train_loss_per_epoch, tracker.val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
            plt.figure(33); plt.yscale('log'); plt.savefig(s_path + '_RETRAIN_loss_per_epoch.png')          


            plot_metric_fun(tracker.val_ce_pb, tracker.val_ce_pb, class_name='', metric_name='HD_metric', plot_num=32)
            plt.figure(32); plt.savefig(s_path + '_RETRAIN_HD_metric.png')
                        

            """ Separate losses """
            if tracker.HD:
                # plot_cost_fun(tracker.train_ce_pb, tracker.train_ce_pb)                   
                # plt.figure(25); plt.savefig(s_path + '_RETRAIN_global_loss_LOG_CE.png')
                # plt.close('all')
                  
                # plot_cost_fun(tracker.train_hd_pb, tracker.train_hd_pb)                   
                # plt.figure(25); plt.savefig(s_path + '_RETRAIN_global_loss_LOG_HD.png')
                # plt.close('all')
                  
                # plot_cost_fun(tracker.train_dc_pb, tracker.train_dc_pb)                   
                # plt.figure(25); plt.savefig(s_path + '_RETRAIN_global_loss_LOG_DC.png')
                # plt.close('all')                  
                
         
                ### for validation
                # plot_cost_fun(tracker.val_ce_pb, tracker.val_ce_pb)                   
                # plt.figure(25); plt.savefig(s_path + '_RETRAIN_VAL_global_loss_LOG_CE_per_epoch.png')
                # plt.close('all')
                  
                plot_cost_fun(tracker.val_hd_pb, tracker.val_hd_pb)                   
                plt.figure(25); plt.savefig(s_path + '_RETRAIN_VAL_global_loss_LOG_HD.png')
                plt.close('all')
                  
                plot_cost_fun(tracker.val_dc_pb, tracker.val_dc_pb)                   
                plt.figure(25); plt.savefig(s_path + '_RETRAIN_VAL_global_loss_LOG_DC.png')
                plt.close('all')        
            
         
            
            """ VALIDATION LOSS PER BATCH??? """
            plot_cost_fun(tracker.val_loss_per_batch, tracker.val_loss_per_batch)                   
            plt.figure(18); plt.savefig(s_path + '_RETRAIN_VAL_global_loss_VAL.png')
            plt.figure(19); plt.savefig(s_path + '_RETRAIN_VAL_detailed_loss_VAL.png')
            plt.figure(25); plt.savefig(s_path + '_RETRAIN_VAL_global_loss_LOG_VAL.png')
            plt.close('all')
              
        
            
            """ Plot metrics per batch """                
            # plot_metric_fun(tracker.train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
            # plt.figure(34); plt.savefig(s_path + '_RETRAIN_Jaccard_per_batch.png')
              
            # plot_cost_fun(tracker.train_loss_per_batch, tracker.train_loss_per_batch)                   
            # plt.figure(18); plt.savefig(s_path + '_RETRAIN_global_loss.png')
            # plt.figure(19); plt.savefig(s_path + '_RETRAIN_detailed_loss.png')
            # plt.figure(25); plt.savefig(s_path + '_RETRAIN_global_loss_LOG.png')
            # plt.close('all')



             
            """ custom plot """
            # output_train = output_train.cpu().data.numpy()            
            # output_train = np.moveaxis(output_train, 1, -1)              
            # seg_train = np.argmax(output_train[0], axis=-1)  
             
            # convert back to CPU
            # batch_x = batch_x.cpu().data.numpy() 
            # batch_y = batch_y.cpu().data.numpy() 

             
            plot_trainer_3D_PYTORCH_snake_seg(seg_val, seg_val, batch_x_val[0], batch_x_val[0], batch_y_val[0], batch_y_val[0],
                                      s_path, tracker.iterations, plot_depth=8)
                                            
             
        """ To save tracker and model (every x iterations) """
        # if tracker.cur_epoch % save_every_num_epochs == 0:           
        #       tracker.iterations += 1

        #       save_name = s_path + 'check_RERUN_' +  str(tracker.iterations)               
        #       torch.save({
        #        'tracker': tracker,

               
        #        }, save_name)
     
                

                
               
              
              
