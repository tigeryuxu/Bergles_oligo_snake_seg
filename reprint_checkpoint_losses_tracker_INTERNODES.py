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

import re

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
    s_path = './(1) Checkpoint_unet_MEDIUM_filt7x7_b8_HD_INTERNODE/'; HD = 1; alpha = 1;  sps_bool = 0
    
    s_path = '/media/user/storage/Data/(1) snake seg project/Backup checkpoints INTERNODE_linear_tracing/(2) Checkpoint_unet_MEDIUM_filt7x7_b8_HD_INTERNODE_SPS_optimizer/'; HD = 1; alpha = 1; sps_bool = 1;
    
    #s_path = './(3) Checkpoint_unet_MEDIUM_filt7x7_b8_HD_INTERNODE_sps_cID_loss/'; cID = 1; HD = 0; alpha = 0; sps_bool = 1; im_type = 0
    
    s_path = './(4) Checkpoint_unet_MEDIUM_filt7x7_b8_HD_INTERNODE_sps_NEW_HD_loss/'; cID = 0; HD = 1; alpha = 1; sps_bool = 1; im_type = 0; sW_centroid = 0
    
    """ path to input data """
    input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING linear myelin seg/TRAINING_linear_myelin_FULL_DATA/'
    

    #im_type = 'c'


    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), 
                     seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop'),  
                     orig_idx= int(re.search('_origId_(.*)_eId', i).group(1)),
                     x = int(re.search('_x_(.*)_y_', i).group(1)),
                     y = int(re.search('_y_(.*)_z_', i).group(1)),
                     z = int(re.search('_z_(.*)_NOCLAHE_', i).group(1)),                     
                     filename= i.split('/')[-1].split('_origId')[0].replace(',', ''))
                     for i in images]


    """ Also load in the all_tree_indices file """
    all_trees = [];


    # ### REMOVE IMAGE 1 from training data
    idx_skip = []
    for idx, im in enumerate(examples):
        filename = im['input']
        if '1to1pair_b_series_t1' in filename:
            print('skip')
            idx_skip.append(idx)



    ### USE THE EXCLUDED IMAGE AS VALIDATION/TESTING
    if len(idx_skip) > 0:
        examples_test = examples[idx_skip[0]:idx_skip[-1]]
    # else:
    #     examples_test = examples[0:20]

    examples = [i for j, i in enumerate(examples) if j not in idx_skip]
          
            
    counter = list(range(len(examples)))
    
    counter_val = list(range(len(examples_test)))  ### NEWLY ADDED!!!

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
        
    loss_function = check['loss_function']       
    tracker = check['tracker']
    scheduler = check['scheduler_type']

    unet = check['model_type']
    unet.load_state_dict(check['model_state_dict']) 

    """ OPTIMIZER HAS TO BE LOADED IN AFTER THE MODEL!!!"""
    if not sps_bool:
        optimizer = check['optimizer_type']
        optimizer.load_state_dict(check['optimizer_state_dict'])
    else:
        import sps
        optimizer = sps.Sps(unet.parameters()) 
    
    #scheduler.load_state_dict(check['scheduler'])     
    #loss_function = check['loss_function']


    tracker.idx_valid = counter
    
    tracker.idx_valid = idx_skip   ### IF ONLY WANT 
    
    
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

    print(onlyfiles_check)
    for check_file in onlyfiles_check:      
        last_file = check_file
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

   
                       """ Training loss """
                       # tracker.val_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                       # loss_val += loss.cpu().data.numpy()
                                        
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
  

                       tracker.val_hd_pb.append(hd_m.cpu().data.numpy())
  

                       """ Find DICE metric:
                                 - NOT using softmax!!! using argmax, so actual binary comparison
                                 """        
                       loss_DICE = dice_loss(outputs_argm, labels_val == 1)
                       tracker.val_dc_pb.append(loss_DICE.cpu().data.numpy())
                       
                       
                       
                       

                       val_idx = val_idx + tracker.batch_size
                       print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                       

                       
                       iter_cur_epoch += 1
   
                       #if starter == 50: stop = time.perf_counter(); diff = stop - start; print(diff);  #break;
                       
                       
                       """ check each image and all metrics """
                       DEBUG = 1
                       if DEBUG:
                            jacc = jacc_eval_GPU_torch(output_val, labels_val)
                            jacc = jacc.cpu().data.numpy()
                           
                
                               
                            batch_x_val = batch_x_val.cpu().data.numpy()
                          
                            batch_y_val = batch_y_val.cpu().data.numpy() 
                            output_val = output_val.cpu().data.numpy()            
                            output_val = np.moveaxis(output_val, 1, -1)       
                            seg_val = np.argmax(output_val[0], axis=-1)  
                              
                            input_3D = batch_x_val[0][0]
                            seed_3D = batch_x_val[0][1]
                            truth_3D = batch_y_val[0]
                            seg_3D = seg_val
                            intersect = truth_3D + seg_3D
                           
                            combined = np.zeros(np.shape(seg_3D))
                           
                            combined[truth_3D > 0] = 1
                            combined[seg_3D > 0] = 2
                            combined[intersect > 1] = 3
                           
                            # plt.figure();
                            # ma = np.amax(combined, axis=0)
                            # plt.imshow(ma, cmap='magma')
                           
                           
                           
                            """ Get sklearn metric """
                            from sklearn.metrics import jaccard_score
                            #jacc_new = jaccard_score(truth_3D.flatten(), seg_3D.flatten())
                           
                               
                            ### DEBUG: plot
                            #plt.close('all')
                            
                            input_3D[seed_3D > 0] = 255
                        
                            
                            in_im = plot_max(batch_x_val[0][0], plot=0)
                            truth_im = plot_max(batch_y_val[0], plot=0)
                            seg_im = plot_max(seg_val, plot=0)
                                                  
                            plt.figure()
                            plt.subplot(1, 3, 1); plt.imshow(in_im)
                            if len(tracker.val_dc_pb) > 0:
                                  plt.title('DC: ' + str(np.round(tracker.val_dc_pb[-1], 4)))
                              
                               
                            plt.subplot(1, 3, 2); plt.imshow(truth_im); 
                               
                            if len(tracker.val_jacc_per_batch) > 0:
                                  plt.title('Jacc: ' + str(np.round(jacc, 4)))
                               
                               
                            cID_val_2D = cID_metric.clDice(truth_im, seg_im)
                            cID_val_3D = cID_metric.clDice(truth_3D, seg_3D)
                               
                            all_cID_3D.append(cID_val_3D); all_cID_2D.append(cID_val_2D); 
                               
                            
                            plt.subplot(1, 3, 3); plt.imshow(seg_im);  
                            if len(tracker.val_hd_pb) > 0:
                                 plt.title('HD: ' + str(np.round(tracker.val_hd_pb[-1], 4))  + '\n' + 'cID_metric: ' + str(round(cID_val_3D, 4))                             
                                            )
                                                           
                            print('num_tested')
                              
                            # """ Early stop """
                            if val_idx > 300:
                                zzz
                                break
                               
                            
                  #print('mean cID 3D: ' + str(np.nanmean(all_cID_3D)))
                  print('mean DICE: ' + str(np.nanmean(np.vstack(tracker.val_dc_pb))))
                  print('mean HD: ' + str(np.nanmean(np.vstack(tracker.val_hd_pb))))
                  print('mean cID_3D: ' + str(np.nanmean(np.vstack(tracker.val_jacc_per_batch))))   
                          
                  tracker.val_loss_per_eval.append(loss_val/iter_cur_epoch)
                  tracker.val_jacc_per_eval.append(np.nanmean(jacc_val))       
                  tracker.val_ce_pb.append(np.nanmean(hd_value))
                                   
                  zzz
                                   
                  
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

             
            #plot_trainer_3D_PYTORCH_snake_seg(seg_val, seg_val, batch_x_val[0], batch_x_val[0], batch_y_val[0], batch_y_val[0],
            #                          s_path, tracker.iterations, plot_depth=8)
                                            
             

        
        
        
    """ To save tracker and model (every x iterations) """     
    #stop_time_epoch = time.perf_counter(); diff = stop_time_epoch - start_time_epoch; print(diff); 
   
    save_name = s_path + 'check_' +  num_check[0] + '_REPLOTTED'        
    torch.save({
     'tracker': tracker,
 
     'model_type': unet,
     'optimizer_type': optimizer,
     'scheduler_type': scheduler,
     
     'model_state_dict': unet.state_dict(),
     'optimizer_state_dict': optimizer.state_dict(),
     'scheduler': [],
     'loss_function': loss_function,  
     
     }, save_name)
   
               
              
              
