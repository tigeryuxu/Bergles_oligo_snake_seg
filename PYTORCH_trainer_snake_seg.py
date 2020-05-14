# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger





TO DO snake seg:
    
    - *** DOUBLE CHECK if need to do CHECK_RESIZE when loading data...
        and if having troubles ==> FIX IN MATLAB

    - add spatial weighting of loss 


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

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    """" Input paths """    
    #s_path = './(9) Checkpoints_TITAN_5x5_256x64_NO_transforms_COMPLEX_LR_sched/'    
    #s_path = './(10) Checkpoints_TITAN_5x5_256x64_TRANSFORMS_COMPLEX_LR_sched/'
    s_path = './(1) Checkpoint_PYTORCH/'
    s_path = './(2) Checkpoint_PYTORCH_spatial_weight/'
    
    
    #input_path = './Train_matched_quads_PYTORCH_256_64_MATCH_ILASTIK/'        
    input_path = '/media/user/storage/Data/(1) snake seg project/Train_SNAKE_SEG_scaled_cleaned/'
        
    #input_path = '/media/user/storage/Train/'
    
    input_path = './Train_SNAKE_SEG_scaled_cleaned/'
    
    
    """ Start network """   
    #kernel_size=5
    #pad = int((kernel_size - 1)/2)
    #unet = UNet(in_channel=1,out_channel=2, kernel_size=kernel_size, pad=pad)
        
    unet = UNet_online(in_channels=2, n_classes=2, depth=5, wf=3, padding= int((5 - 1)/2), batch_norm=False, up_mode='upconv')
    
    unet.to(device)
    print('parameters:', sum(param.numel() for param in unet.parameters()))

    loss_function = torch.nn.CrossEntropyLoss(reduction='none')
    
    
    
    #optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)   
    lr = 1e-3; milestones = [5, 10, 100]
    #lr = 1e-4;  milestones = [6, 14, 20]
    
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    """ Add scheduler """
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
    resume = 0
    
    
    """ TO LOAD OLD CHECKPOINT """
    # Read in file names
    onlyfiles_check = glob.glob(os.path.join(s_path,'check_*'))
    onlyfiles_check.sort(key = natsort_key1)
    
    if not onlyfiles_check:   
        """ Get metrics per batch """
        train_loss_per_batch = []; train_jacc_per_batch = []
        val_loss_per_batch = []; val_jacc_per_batch = []
        """ Get metrics per epoch"""
        train_loss_per_epoch = []; train_jacc_per_epoch = []
        val_loss_per_eval = []; val_jacc_per_eval = []
        plot_sens = []; plot_sens_val = [];
        plot_prec = []; plot_prec_val = [];
        lr_plot = [];
        iterations = 0;
        

    else:             
        """ Find last checkpoint """       
        last_file = onlyfiles_check[-1]
        split = last_file.split('check_')[-1]
        num_check = split.split('.')
        checkpoint = num_check[0]
        checkpoint = 'check_' + checkpoint

        print('restoring weights')
        check = torch.load(s_path + checkpoint)
        cur_epoch = check['cur_epoch']
        iterations = check['iterations']
        idx_train = check['idx_train']
        idx_valid = check['idx_valid']
        
        unet.load_state_dict(check['model_state_dict'])
        optimizer.load_state_dict(check['optimizer_state_dict'])
        scheduler.load_state_dict(check['scheduler'])
        
        """ Restore per batch """
        train_loss_per_batch = check['train_loss_per_batch']
        train_jacc_per_batch = check['train_jacc_per_batch']
        val_loss_per_batch = check['val_loss_per_batch']
        val_jacc_per_batch = check['val_jacc_per_batch']
     
        """ Restore per epoch """
        train_loss_per_epoch = check['train_loss_per_epoch']
        train_jacc_per_epoch = check['train_jacc_per_epoch']
        val_loss_per_eval = check['val_loss_per_eval']
        val_jacc_per_eval = check['val_jacc_per_eval']
     
        plot_sens = check['plot_sens']
        plot_sens_val = check['plot_sens_val']
        plot_prec = check['plot_prec']
        plot_prec_val = check['plot_prec_val']
     
        lr_plot = check['lr_plot']
        
        resume = 1
        
       
    """ Prints out all variables in current graph """
    # Required to initialize all
    batch_size = 8; save_every_num_epochs = 1;
    plot_every_num_epochs = 1;
    validate_every_num_epochs = 1;        
    test_size = 0.1
    """ Load training data """
    print('loading data')   
    num_workers = 2;
    """ Specify transforms """
    #transforms = initialize_transforms(p=0.5)
    transforms = initialize_transforms_simple(p = 0.5)
    #transforms = 0
    
    sp_weight_bool = 0
    

    # """ load mean and std """  
    mean_arr = np.load('./normalize/' + 'mean_VERIFIED.npy')
    std_arr = np.load('./normalize/' + 'std_VERIFIED.npy')    

    """ Load filenames from tiff """
    images = glob.glob(os.path.join(input_path,'*_NOCLAHE_input_crop.tif'))    # can switch this to "*truth.tif" if there is no name for "input"
    images.sort(key=natsort_keygen(alg=ns.REAL))  # natural sorting
    examples = [dict(input=i,truth=i.replace('_NOCLAHE_input_crop.tif','_DILATE_truth_class_1_crop.tif'), seed_crop=i.replace('_NOCLAHE_input_crop','_DILATE_seed_crop')) for i in images]
    counter = list(range(len(examples)))
    
    if not resume:
        idx_train, idx_valid, empty, empty = train_test_split(counter, counter, test_size=test_size, random_state=2018)
        
    training_set = Dataset_tiffs_snake_seg(idx_train, examples, mean_arr, std_arr, sp_weight_bool=sp_weight_bool, transforms = transforms)
    val_set = Dataset_tiffs_snake_seg(idx_valid, examples, mean_arr, std_arr, transforms = 0)
    
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
    for cur_epoch in range(len(train_loss_per_epoch), 10000):           
         loss_train = 0
         jacc_train = 0   
                  
         for param_group in optimizer.param_groups:
              cur_lr = param_group['lr']
              lr_plot.append(cur_lr)
              print('Current learning rate is: ' + str(cur_lr))

         iter_cur_epoch = 0;          
         for batch_x, batch_y, spatial_weight in training_generator:
             
                """ Speed testing
                        
                    batch 8, no workers, 50, == 19.74
                    ''' 4 workers ==> 9.2
                
                """
                if iterations == 1:
                    start = time.perf_counter()
                if iterations == 50:
                    stop = time.perf_counter(); diff = stop - start; print(diff)
                    

                """ Plot for debug """ 
                # if iterations % 1 == 0 and iterations != 0:
                #     np_inputs = np.asarray(batch_x.numpy()[0], dtype=np.uint8)
                #     np_labels = np.asarray(batch_y.numpy()[0], dtype=np.uint8)
                #     np_labels[np_labels > 0] = 255
                    
                #     # imsave(s_path + str(iterations) + '_input.tif', np_inputs)
                #     # imsave(s_path + str(iterations) + '_label.tif', np_labels)
                    
                #     in_max = plot_max(np_inputs[0], plot=0)
                #     seed_max = plot_max(np_inputs[1], plot=0)
                #     lb_max = plot_max(np_labels, plot=0)
                    
                #     imsave(s_path + str(iterations) + '_max_input.tif', in_max)
                #     imsave(s_path + str(iterations) + '_max_crop_seed.tif', seed_max)
                #     imsave(s_path + str(iterations) + '_max_label.tif', lb_max)                
                
                """ Load data ==> shape is (batch_size, num_channels, depth, height, width)
                     (1) converts to Tensor
                     (2) normalizes + appl other transforms on GPU
                     (3) ***add non-blocking???
                     ***INPUT LABELS MUST BE < 255??? or else get CudNN error
                     
                """
                inputs, labels = transfer_to_GPU(batch_x, batch_y, device, mean_arr, std_arr)
                inputs = inputs[:, 0, ...]


                """ zero the parameter gradients"""
                optimizer.zero_grad()       
                
                """ forward + backward + optimize """
                output_train = unet(inputs)
                
                loss = loss_function(output_train, labels)
                if len(spatial_weight.size()):
                     spatial_tensor = torch.tensor(spatial_weight, dtype = torch.float, device=device, requires_grad=False)          
                     weighted = loss * spatial_tensor
                     loss = torch.mean(weighted)
                else:
                     loss = torch.mean(loss)                
                
                
                loss.backward()
                optimizer.step()
               
                """ Training loss """
                """ ********************* figure out how to do spatial weighting??? """
                train_loss_per_batch.append(loss.cpu().data.numpy());  # Training loss
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
                
                
    
         train_loss_per_epoch.append(loss_train/iter_cur_epoch)
         train_jacc_per_epoch.append(jacc_train/iter_cur_epoch)              
    
         """ Should I keep track of loss on every single sample? and iteration? Just not plot it??? """   
         loss_val = 0; jacc_val = 0
         precision_val = 0; sensitivity_val = 0; val_idx = 0;
         iter_cur_epoch = 0; 
         if cur_epoch % validate_every_num_epochs == 0:
             
              with torch.set_grad_enabled(False):  # saves GPU RAM
                   for batch_x_val, batch_y_val in val_generator:
                        
                        """ Transfer to GPU to normalize ect... """
                        inputs_val, labels_val = transfer_to_GPU(batch_x_val, batch_y_val, device, mean_arr, std_arr)
             
                        inputs_val = inputs_val[:, 0, ...]   
             
                        # forward pass to check validation
                        output_val = unet(inputs_val)
                        loss = loss_function(output_val, labels_val)
          
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
                        # TP, FN, FP = find_TP_FP_FN_from_im(seg_val, batch_y_val[0])
                                       
                        # if TP + FN == 0: TP;
                        # else: sensitivity = TP/(TP + FN); sensitivity_val += sensitivity;    # PPV
                                       
                        # if TP + FP == 0: TP;
                        # else: precision = TP/(TP + FP);  precision_val += precision    # precision             
              
                        val_idx = val_idx + batch_size
                        print('Validation: ' + str(val_idx) + ' of total: ' + str(validation_size))
                        
                        iter_cur_epoch += 1
                
              
                           
                   val_loss_per_eval.append(loss_val/iter_cur_epoch)
                   val_jacc_per_eval.append(jacc_val/iter_cur_epoch)       
                   
                   #plot_prec.append(precision_val/(validation_size/batch_size))
                   #plot_sens.append(sensitivity_val/(validation_size/batch_size))
                   
                  
                   """ Add to scheduler to do LR decay """
                   scheduler.step()
                  
         if cur_epoch % plot_every_num_epochs == 0:       
              """ Plot sens + precision + jaccard + loss """
              #plot_metric_fun(plot_sens, plot_sens_val, class_name='', metric_name='sensitivity', plot_num=30)
              #plt.figure(30); plt.savefig(s_path + 'Sensitivity.png')
                    
              #plot_metric_fun(plot_prec, plot_prec_val, class_name='', metric_name='precision', plot_num=31)
              #plt.figure(31); plt.savefig(s_path + 'Precision.png')
           
              plot_metric_fun(train_jacc_per_epoch, val_jacc_per_eval, class_name='', metric_name='jaccard', plot_num=32)
              plt.figure(32); plt.savefig(s_path + 'Jaccard.png')
                   
                 
              plot_metric_fun(train_loss_per_epoch, val_loss_per_eval, class_name='', metric_name='loss', plot_num=33)
              plt.figure(33); plt.yscale('log'); plt.savefig(s_path + 'loss_per_epoch.png')          
                   
              
              plot_metric_fun(lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
              plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 
     

              """ Plot metrics per batch """                
              plot_metric_fun(train_jacc_per_batch, [], class_name='', metric_name='jaccard', plot_num=34)
              plt.figure(34); plt.savefig(s_path + 'Jaccard_per_batch.png')
                                
              plot_cost_fun(train_loss_per_batch, train_loss_per_batch)                   
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
               save_name = s_path + 'check_' +  str(iterations)               
               torch.save({
                'cur_epoch': cur_epoch,
                'iterations': iterations,
                'idx_train': idx_train,
                'idx_valid': idx_valid,
                
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                
                'train_loss_per_batch': train_loss_per_batch,
                'train_jacc_per_batch': train_jacc_per_batch,
                'val_loss_per_batch': val_loss_per_batch,
                'val_jacc_per_batch': val_jacc_per_batch,
                
                'train_loss_per_epoch': train_loss_per_epoch,
                'train_jacc_per_epoch': train_jacc_per_epoch,
                'val_loss_per_eval': val_loss_per_eval,
                'val_jacc_per_eval': val_jacc_per_eval,
                
                'plot_sens': plot_sens,
                'plot_sens_val': plot_sens_val,
                'plot_prec': plot_prec,
                'plot_prec_val': plot_prec_val,
                
                'lr_plot': lr_plot
                }, save_name)
     
                

                
               
              
              