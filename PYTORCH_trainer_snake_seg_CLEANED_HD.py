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
    s_path = './(51) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Hd_loss_balance_repeat_MARCC/'; HD = 1; alpha = 1;
    
    """ path to input data """
    input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads/'; dataset = 'new crop pads'
    #input_path = 'E:/7) Bergles lab data/Traces files/TRAINING FORWARD PROP ONLY SCALED crop pads/'; 

    #input_path = '/lustre04/scratch/yxu233/TRAINING FORWARD PROP ONLY SCALED crop pads/';  dataset = 'new crop pads'

    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop')) for i in images]
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

    if not onlyfiles_check:   ### if no old checkpoints found, start new network and tracker
 
        """ Hyper-parameters """
        deep_sup = False
        switch_norm = True
        sp_weight_bool = 0
        transforms = 0; #transforms = initialize_transforms(p=0.5)
        batch_size = 4;      
        test_size = 0.1  

        """ Initialize network """  
        kernel_size = 5
        pad = int((kernel_size - 1)/2)
        #unet = UNet_online(in_channels=2, n_classes=2, depth=5, wf=3, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
        #                    batch_norm=True, batch_norm_switchable=False, up_mode='upsample')
        unet = NestedUNet(num_classes=2, input_channels=2, deep_sup=deep_sup, padding=pad, batch_norm_switchable=switch_norm)
        #unet = UNet_3Plus(num_classes=2, input_channels=2, kernel_size=kernel_size, padding=pad)

        unet.train()
        unet.to(device)
        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Select loss function *** unimportant if using HD loss """
        if not HD:    loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        else:         loss_function = 'Haussdorf'
            

        """ Select optimizer """
        lr = 1e-5; milestones = [20, 100]  # with AdamW slow down
        optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)

        """ Add scheduler """
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
            
        """ initialize index of training set and validation set, split using size of test_size """
        idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
        
        """ initialize training_tracker """
        tracker = tracker(batch_size, test_size, mean_arr, std_arr, idx_train, idx_valid, deep_sup=deep_sup, switch_norm=switch_norm, alpha=alpha, HD=HD,
                                          sp_weight_bool=sp_weight_bool, transforms=transforms, dataset=input_path)

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
        optimizer = check['optimizer_type']
        scheduler = check['scheduler_type']
        unet.load_state_dict(check['model_state_dict'])
        unet.to(device)
        optimizer.load_state_dict(check['optimizer_state_dict'])
        scheduler.load_state_dict(check['scheduler'])     
        loss_function = check['loss_function']

        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        
        """ Clean up checkpoint file """
        del check
        torch.cuda.empty_cache()

                
    """ Create datasets for dataloader """
    training_set = Dataset_tiffs_snake_seg(tracker.idx_train, examples, tracker.mean_arr, tracker.std_arr, sp_weight_bool=tracker.sp_weight_bool, transforms = tracker.transforms)
    val_set = Dataset_tiffs_snake_seg(tracker.idx_valid, examples, tracker.mean_arr, tracker.std_arr, sp_weight_bool=tracker.sp_weight_bool, transforms = 0)
    
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
   
    """ Start training """
    for cur_epoch in range(len(tracker.train_loss_per_epoch), 10000): 
     
         """ check and plot params during training """             
         for param_group in optimizer.param_groups:
              #param_group['lr'] = 1e-6   # manually sets learning rate
              cur_lr = param_group['lr']
              tracker.lr_plot.append(cur_lr)
              tracker.print_essential()

         unet.train()  ### set PYTORCH to training mode

         start_time_epoch = time.perf_counter();
         loss_train = 0; jacc_train = 0; ce_train = 0; dc_train = 0; hd_train = 0;
         iter_cur_epoch = 0; starter = 0;
         for batch_x, batch_y, spatial_weight in training_generator:
                ### Test speed for debug
                starter += 1
                if starter == 2:  start = time.perf_counter()
                if starter == 50: stop = time.perf_counter(); diff = stop - start; print(diff);  #break;
                     
                """ Load data ==> shape is (batch_size, num_channels, depth, height, width)
                     (1) converts to Tensor
                     (2) normalizes + applies other transforms on GPU   ***INPUT LABELS MUST BE < 255??? or else get CudNN error
                """
                inputs, labels = transfer_to_GPU(batch_x, batch_y, device, mean_arr, std_arr)
                inputs = inputs[:, 0, ...]

                # PRINT OUT THE SHAPE OF THE INPUT
                if iter_cur_epoch == 0:
                    print('input size is' + str(batch_x.shape))

                
                """ initialize each iteration """
                optimizer.zero_grad()    ### zero the parameter gradients
                output_train = unet(inputs)  ### forward + backward + optimize
                                  
                """ calculate loss: includes HD loss functions """
                if HD:
                    loss, tracker, ce_train, dc_train, hd_train = compute_HD_loss(output_train, labels, alpha, tracker, 
                                                                                  ce_train, dc_train, hd_train, val_bool=0)
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
                optimizer.step()
               
                """ Training loss """
                tracker.train_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
                loss_train += loss.cpu().data.numpy()
                
   
                """ Calculate Jaccard on GPU """                 
                jacc = jacc_eval_GPU_torch(output_train, labels)
                jacc = jacc.cpu().data.numpy()
                                            
                jacc_train += jacc # Training jacc
                tracker.train_jacc_per_batch.append(jacc)
   
                tracker.iterations = tracker.iterations + 1       
                iter_cur_epoch += 1
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


         tracker.train_loss_per_epoch.append(loss_train/iter_cur_epoch)
         tracker.train_jacc_per_epoch.append(jacc_train/iter_cur_epoch)        
         
         
         """ calculate new alpha for next epoch """   
         if HD:
             alpha = alpha_step(ce_train, dc_train, hd_train, iter_cur_epoch)

    
         """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
         loss_val = 0; jacc_val = 0; val_idx = 0;
         iter_cur_epoch = 0;  ce_val = 0; dc_val = 0; hd_val = 0;
         if cur_epoch % validate_every_num_epochs == 0:
             
              with torch.set_grad_enabled(False):  # saves GPU RAM
                   unet.eval()
                   for batch_x_val, batch_y_val, spatial_weight in val_generator:
                      
                        """ Transfer to GPU to normalize ect... """
                        inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, mean_arr, std_arr)
                        inputs_val = inputs_val[:, 0, ...]   
             
                        # forward pass to check validation
                        output_val = unet(inputs_val)

                        """ calculate loss 
                                include HD loss functions """
                        if HD:
                            loss, tracker, ce_val, dc_val, hd_val = compute_HD_loss(output_val, labels_val, alpha, tracker, 
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

                        val_idx = val_idx + batch_size
                        print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                        iter_cur_epoch += 1

                        #if starter == 50: stop = time.perf_counter(); diff = stop - start; print(diff);  #break;

                           
                   tracker.val_loss_per_eval.append(loss_val/iter_cur_epoch)
                   tracker.val_jacc_per_eval.append(jacc_val/iter_cur_epoch)       
                   
                   """ Add to scheduler to do LR decay """
                   scheduler.step()
                  
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
     
                

                
               
              
              
