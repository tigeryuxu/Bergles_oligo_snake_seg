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
import bcolz
from sklearn.model_selection import train_test_split

from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order


from PYTORCH_dataloader import *
from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.train_tracker import *


from layers.UNet_pytorch_online import *
from layers.unet_nested import *
from layers.unet3_3D import *
from layers.switchable_BN import *

from losses_pytorch.HD_loss import *

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
    s_path = './(47) Checkpoint_nested_unet_SPATIALW_small_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/'; HD = 1; alpha = 1;
    #s_path = './(48) Checkpoint_nested_unet_SPATIALW_medium_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/';  HD = 1; alpha = 1;
    #s_path = './(49) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/'; HD = 1; alpha = 1;
    #s_path = './(50) Checkpoint_nested_unet_SPATIALW_medium_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance_FP16/';  HD = 1; alpha = 1; FP_16 = 1;
    
    s_path = './test_new/'; HD = 1; alpha = 1;
    
    """ path to input data """
    #input_path = '/media/user/storage/Data/(1) snake seg project/Train_SNAKE_SEG_scaled_cleaned/'; dataset = 'old' ### OLD DATA 80 x 80 x16
    #input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED/'; dataset = 'new'  ### NEW DATA 80 x 80 x 32
    input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads/'; dataset = 'new crop pads'
    

    input_path = 'E:/7) Bergles lab data/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads/'; 

    # """ load mean and std for normalization later """  
    mean_arr = np.load('./normalize/' + 'mean_VERIFIED.npy')
    std_arr = np.load('./normalize/' + 'std_VERIFIED.npy')   

    num_workers = 2;
 
    save_every_num_epochs = 1; plot_every_num_epochs = 1; validate_every_num_epochs = 1;      
    
    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
    onlyfiles_check.sort(key = natsort_key1)
    

    deep_supervision = False
    switch_norm = True
        
    if not onlyfiles_check:   
        """ Start network """   
        kernel_size = 5
        pad = int((kernel_size - 1)/2)
        #unet = UNet_online(in_channels=2, n_classes=2, depth=5, wf=3, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
        #                    batch_norm=True, batch_norm_switchable=False, up_mode='upsample')
        unet = NestedUNet(num_classes=2, input_channels=2, deep_supervision=deep_supervision, padding=pad, batch_norm_switchable=switch_norm)
        #unet = UNet_3Plus(num_classes=2, input_channels=2, kernel_size=kernel_size, padding=pad)

        unet.train()
        unet.to(device)
        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Select loss function """
        loss_function = torch.nn.CrossEntropyLoss(reduction='none')

        """ Select optimizer """
        lr = 1e-3; milestones = [5, 50, 100]  # with AdamW
        lr = 1e-5; milestones = [20, 100]  # with AdamW slow down

        optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

        """ Add scheduler """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        
        """ Prints out all variables in current graph """
        # Required to initialize all
        batch_size = 4;      
        test_size = 0.1  
        
        
        """ initialize training_tracker """
        train_tracker = train_tracker()
        
        """ Specify transforms """
        #transforms = initialize_transforms(p=0.5)
        transforms = 0        
        sp_weight_bool = 0
        resume = 0
        
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
        
        train_tracker = check['train_tracker']
        
        idx_train = check['idx_train']
        idx_valid = check['idx_valid']
        
        
        unet = check['model_type']
        optimizer = check['optimizer_type']
        scheduler = check['scheduler_type']
        
        
        unet.load_state_dict(check['model_state_dict'])
        unet.to(device)
        optimizer.load_state_dict(check['optimizer_state_dict'])
        scheduler.load_state_dict(check['scheduler'])
        
        if HD:
            alpha = check['alpha']
            
     
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
 
    
    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop')) for i in images]
    counter = list(range(len(examples)))
    
    
    all_branch_idx = np.load('./normalize/all_branch_idx.npy')
    
    if not resume:
        idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
        
            
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
   

    """ Start training """
    for cur_epoch in range(len(train_tracker.train_loss_per_epoch), 10000): 
         
         unet.train()         

         loss_train = 0; jacc_train = 0   
         ce_train = 0; dc_train = 0; hd_train = 0;
         
             
         for param_group in optimizer.param_groups:
              #param_group['lr'] = 1e-6   # manually sets learning rate
              cur_lr = param_group['lr']
              train_tracker.lr_plot.append(cur_lr)
              print('Current learning rate is: ' + str(cur_lr))
              print('Weight bool is: ' + str(sp_weight_bool))
              print('switch norm bool is: ' + str(switch_norm))
              print('batch_size is: ' + str(batch_size))
              print('dataset is: ' + dataset)              
              print('deep_supervision is: ' + str(deep_supervision))
              print('alpha is: ' + str(alpha))
              
         iter_cur_epoch = 0;   
         starter = 0;
         for batch_x, batch_y, spatial_weight in training_generator:
                ### Test speed for debug
                starter += 1
                if starter == 2:
                    start = time.perf_counter()
                if starter == 50:
                    stop = time.perf_counter(); diff = stop - start; print(diff)
                      
                    
                # PRINT OUT THE SHAPE OF THE INPUT
                if iter_cur_epoch == 0:
                    print('input size is' + str(batch_x.shape))
                
                """ Load data ==> shape is (batch_size, num_channels, depth, height, width)
                     (1) converts to Tensor
                     (2) normalizes + applies other transforms on GPU
                     ***INPUT LABELS MUST BE < 255??? or else get CudNN error
                """
                inputs, labels = transfer_to_GPU(batch_x, batch_y, device, mean_arr, std_arr)
                inputs = inputs[:, 0, ...]


                """ zero the parameter gradients"""
                optimizer.zero_grad()       
                
                """ forward + backward + optimize """
                output_train = unet(inputs)                    
               
                """ calculate loss """
                if deep_supervision:
                                   
                   # compute output
                   loss = 0
                   for output in output_train:
                        loss += loss_function(output, labels)
                   loss /= len(output_train)
                   
                   output_train = output_train[-1]  # set this so can eval jaccard later
               
                else:
               
                   loss = loss_function(output_train, labels)
               
               
                """ Include HD loss functions """
                if HD:
                   loss_ce = F.cross_entropy(output_train, labels)
                   outputs_soft = F.softmax(output_train, dim=1)
                   loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels == 1)
                   # compute distance maps and hd loss
                   with torch.no_grad():
                       # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
                       gt_dtm_npy = compute_dtm(labels.cpu().numpy(), outputs_soft.shape)
                       gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
                       seg_dtm_npy = compute_dtm(outputs_soft[:, 1, :, :, :].cpu().numpy()>0.5, outputs_soft.shape)
                       seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)
       
                   loss_hd = hd_loss(outputs_soft, labels, seg_dtm, gt_dtm)
                   
                   loss = alpha*(loss_ce+loss_seg_dice) + loss_hd
       
                   train_tracker.train_ce_pb.append(loss_ce.cpu().data.numpy())
                   train_tracker.train_dc_pb.append(loss_seg_dice.cpu().data.numpy())
                   train_tracker.train_hd_pb.append(loss_hd.cpu().data.numpy())
                   
                   ce_train += loss_ce.cpu().data.numpy()
                   dc_train += loss_seg_dice.cpu().data.numpy()
                   hd_train += loss_hd.cpu().data.numpy()                    
    
                else:
    
                   if torch.is_tensor(spatial_weight):
                        spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                        weighted = loss * spatial_tensor
                        loss = torch.mean(weighted)
                          
                   else:
                        loss = torch.mean(loss)   
                        #loss
                   
               
                loss.backward()
                optimizer.step()
               
  
                
                """ Training loss """
                train_tracker.train_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                loss_train += loss.cpu().data.numpy()
                
   
                """ Calculate Jaccard on GPU """                 
                jacc = jacc_eval_GPU_torch(output_train, labels)
                jacc = jacc.cpu().data.numpy()
                                            
                jacc_train += jacc # Training jacc
                train_jacc_per_batch.append(jacc)
   
                iterations = iterations + 1       
                iter_cur_epoch += 1
                if iterations % 100 == 0:
                    print('Trained: %d' %(iterations))


                """ Plot for ground truth """
                # output_train = output_train.cpu().data.numpy()            
                # output_train = np.moveaxis(output_train, 1, -1)              
                # seg_train = np.argmax(output_train[0], axis=-1)  
                  
                # # convert back to CPU
                # batch_x = batch_x.cpu().data.numpy() 
                # batch_y = batch_y.cpu().data.numpy() 
 
                # plot_trainer_3D_PYTORCH_snake_seg(seg_train, seg_train, batch_x[0], batch_x[0], batch_y[0], batch_y[0],
                #                            s_path, iterations, plot_depth=8)


         train_tracker.train_loss_per_epoch.append(loss_train/iter_cur_epoch)
         train_tracker.train_jacc_per_epoch.append(jacc_train/iter_cur_epoch)        
         
         
         """ calculate new alpha for next epoch """   
         if HD:
              mean_ce = ce_train/iter_cur_epoch
              mean_dc = dc_train/iter_cur_epoch
              mean_combined = (mean_ce + mean_dc)/2
             
              mean_hd = hd_train/iter_cur_epoch
             
              alpha = mean_hd/(mean_combined)
             
    
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
                


                        """ Include HD loss functions """
                        if HD:
                            loss_ce = F.cross_entropy(output_val, labels_val)
                            outputs_soft = F.softmax(output_val, dim=1)
                            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels_val == 1)
                            # compute distance maps and hd loss
                            with torch.no_grad():
                                # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
                                gt_dtm_npy = compute_dtm(labels_val.cpu().numpy(), outputs_soft.shape)
                                gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
                                seg_dtm_npy = compute_dtm(outputs_soft[:, 1, :, :, :].cpu().numpy()>0.5, outputs_soft.shape)
                                seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)
                
                            loss_hd = hd_loss(outputs_soft, labels_val, seg_dtm, gt_dtm)
                            loss = alpha*(loss_ce+loss_seg_dice) + (1 - alpha) * loss_hd
                

                            
                        else:
                            if torch.is_tensor(spatial_weight):
                                   spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                                   weighted = loss * spatial_tensor
                                   loss = torch.mean(weighted)
                            elif dist_loss:
                                   loss  # do not do anything if do not need to reduce
                                
                            else:
                                   loss = torch.mean(loss)  
                                   

                        """ Training loss """
                        train_tracker.val_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                        loss_val += loss.cpu().data.numpy()
                                         
                        """ Calculate jaccard on GPU """
                        jacc = jacc_eval_GPU_torch(output_val, labels_val)
                        jacc = jacc.cpu().data.numpy()
                        
                        jacc_val += jacc
                        train_tracker.val_jacc_per_batch.append(jacc)   

                        val_idx = val_idx + batch_size
                        print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                        iter_cur_epoch += 1
                           
                   train_tracker.val_loss_per_eval.append(loss_val/iter_cur_epoch)
                   train_tracker.val_jacc_per_eval.append(jacc_val/iter_cur_epoch)       
                   

                  
                   """ Add to scheduler to do LR decay """
                   scheduler.step()
                  
         if cur_epoch % plot_every_num_epochs == 0:       
              """ Plot sens + precision + jaccard + loss """           
              plot_metric_fun(train_jacc_per_epoch, val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32)
              plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
              
                 
              plot_metric_fun(train_loss_per_epoch, val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
              plt.figure(33); plt.yscale('log'); plt.savefig(s_path + 'loss_per_epoch.png')          


              plot_metric_fun(train_loss_per_epoch, val_loss_per_eval, class_name='', metric_name='loss', plot_num=40)
              plt.figure(40); plt.savefig(s_path + 'loss_per_epoch_NO_LOG.png')         
                   
              
                
              """ Separate losses """
              if HD:
                  plot_cost_fun(train_tracker.train_ce_pb, train_tracker.train_ce_pb)                   
                  plt.figure(18); plt.savefig(s_path + '_global_loss_CE.png')
                  #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
                  plt.figure(25); plt.savefig(s_path + '_global_loss_LOG_CE.png')
                  plt.close('all')
    
                  plot_cost_fun(train_tracker.train_hd_pb, train_tracker.train_hd_pb)                   
                  plt.figure(18); plt.savefig(s_path + '_global_loss_HD.png')
                  #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
                  plt.figure(25); plt.savefig(s_path + '_global_loss_LOG_HD.png')
                  plt.close('all')
    
                  plot_cost_fun(train_tracker.train_dc_pb, train_tracker.train_dc_pb)                   
                  plt.figure(18); plt.savefig(s_path + '_global_loss_DC.png')
                  #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
                  plt.figure(25); plt.savefig(s_path + '_global_loss_LOG_DC.png')
                  plt.close('all')                  
                  
                
              
              """ VALIDATION LOSS PER BATCH??? """
              plot_cost_fun(train_tracker.val_loss_per_batch, train_tracker.val_loss_per_batch)                   
              plt.figure(18); plt.savefig(s_path + '_VAL_global_loss_VAL.png')
              plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
              plt.figure(25); plt.savefig(s_path + '_VAL_global_loss_LOG_VAL.png')
              plt.close('all')
                
              
              plot_metric_fun(train_tracker.lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
              plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 

              """ Plot metrics per batch """                
              plot_metric_fun(train_tracker.train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
              plt.figure(34); plt.savefig(s_path + 'Jaccard_per_batch.png')
                                
              plot_cost_fun(train_tracker.train_loss_per_batch, train_tracker.train_loss_per_batch)                   
              plt.figure(18); plt.savefig(s_path + 'global_loss.png')
              plt.figure(19); plt.savefig(s_path + 'detailed_loss.png')
              plt.figure(25); plt.savefig(s_path + 'global_loss_LOG.png')
              plt.close('all')
                                 
              plot_depth = 8
              output_train = output_train.cpu().data.numpy()            
              output_train = np.moveaxis(output_train, 1, -1)              
              seg_train = np.argmax(output_train[0], axis=-1)  
              
              # convert back to CPU
              batch_x = batch_x.cpu().data.numpy() 
              batch_y = batch_y.cpu().data.numpy() 
              batch_x_val = batch_x_val.cpu().data.numpy()
              
              plot_trainer_3D_PYTORCH_snake_seg(seg_train, seg_val, batch_x[0], batch_x_val[0], batch_y[0], batch_y_val[0],
                                       s_path, iterations, plot_depth=plot_depth)
                                             
              
         """ To save (every x iterations) """
         if cur_epoch % save_every_num_epochs == 0:           
              
               train_tracker.cur_epoch = cur_epoch
               train_tracker.iterations = iterations
               
               save_name = s_path + 'check_' +  str(iterations)               
               torch.save({
                'train_tracker': train_tracker,
                
                'idx_train': idx_train,
                'idx_valid': idx_valid,
                
                'model_type': unet,
                'optimizer_type': optimizer,
                'scheduler_type': scheduler,
                
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                
                'alpha': alpha,

                'lr_plot': lr_plot,
                
                 # newly added
                'mean_arr': mean_arr,
                'std_arr': std_arr,
                
                'batch_size': batch_size,  
                'sp_weight_bool': sp_weight_bool,
                'loss_function': loss_function,  
                'transforms': transforms  

                
                
                }, save_name)
     
                

                
               
              
              