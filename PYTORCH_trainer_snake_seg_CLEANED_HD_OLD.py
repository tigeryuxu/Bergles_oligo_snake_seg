# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger

TO DO snake seg:
    
    - *** DOUBLE CHECK if need to do CHECK_RESIZE when loading data...
        and if having troubles ==> FIX IN MATLAB

    - add spatial weighting of loss 

    TO TRY:
        - with deep supervision
        - with conv1 after upsampling
        - with more filters per layer
        - with upsample vs. conv
        - setup on MARCC
        - fix the broken data?
        - more transforms??? ==> are they messing with the seed channel???
        
        - no spatial weight with nested?
                
        - ***slower training speed
                
        ***TRANSFORMS WONT WORK WITH SPATIAL WEIGHT??? BECAUSE THE WEIGHT MAP NEEDS TO BE REMADE???
        ***CLEAN GARBAGE WITHIN CELL BODIES IN TRAINING DATA!!!
"""

""" ALLOWS print out of results on compute canada """
import matplotlib
matplotlib.rc('xtick', labelsize=8)
matplotlib.rc('ytick', labelsize=8) 
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

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


# with napari.gui_qt():
#     viewer = napari.view_image(seg_val)

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 

if __name__ == '__main__':
        
    """ Define GPU to use """
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    print(device)
    
    
    """" Input paths """    
    s_path = './(35) Checkpoint_nested_unet_SPATIALW_complex_SWITCH_NORM/'
    #s_path = './(36) Checkpoint_nested_unet_SPATIALW_complex_SWITCH_NORM_medium/'
    #s_path = './(37) Checkpoint_nested_unet_SPATIALW_complex/'
    
    #s_path = './(38) Checkpoint_nested_unet_SPATIALW_complex_batch_4_NEW_DATA/'
    
    #s_path = './(39) Checkpoint_nested_unet_SPATIALW_simple_b4_NEW_DATA_SWITCH_NORM/'
    
    
    s_path = './(40) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad/'

    #s_path = './(41) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_crop_pad/'
    
    #s_path = './(42) Checkpoint_nested_unet_SPATIALW_medium_b4_SWITCH_NORM_crop_pad/'
    
    
    #s_path = './(43) Checkpoint_nested_unet_SPATIALW_medium_b4_SWITCH_NORM_crop_pad_deep_sup/'
    
    
    #s_path = './(44) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_deep_sup/'
    
    
    cont_HD = 0
    
    #s_path = './(45) Checkpoint_nested_unet_SPATIALW_medium_b4_NEW_DATA_SWITCH_NORM_crop_pad_CONTINUE_Haussdorf/'; cont_HD = 1
    
    
    #s_path = './(46) Checkpoint_nested_unet_SPATIALW_small_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf/';  HD = 1; alpha = 0.001;
    
    
    #s_path = './(47) Checkpoint_nested_unet_SPATIALW_small_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/';  HD = 1; alpha = 1;
    
    
    #s_path = './(48) Checkpoint_nested_unet_SPATIALW_medium_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/';  HD = 1; alpha = 1;
    
    
    s_path = './(49) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/'; HD = 1; alpha = 1;
    

    FP_16 = 0

    #s_path = './(50) Checkpoint_nested_unet_SPATIALW_medium_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance_FP16/';  HD = 1; alpha = 1; FP_16 = 1;
    
    
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
    
    
    deep_supervision = False
    switch_norm = True
    if both:
        loss_function_2 = torch.nn.CrossEntropyLoss()
        #loss_function = HDDTBinaryLoss(); dist_loss = 1
        
    if not onlyfiles_check:   
        """ Get metrics per batch """
        train_loss_per_batch = []; train_jacc_per_batch = []
        val_loss_per_batch = []; val_jacc_per_batch = []
        
        
        train_ce_pb = []; train_hd_pb = []; train_dc_pb = [];
        
        
        
        """ Get metrics per epoch"""
        train_loss_per_epoch = []; train_jacc_per_epoch = []
        val_loss_per_eval = []; val_jacc_per_eval = []
        plot_sens = []; plot_sens_val = [];
        plot_prec = []; plot_prec_val = [];
        lr_plot = [];
        iterations = 0;
        
        """ Start network """   
        kernel_size = 5
        pad = int((kernel_size - 1)/2)
        #unet = UNet(in_channel=1,out_channel=2, kernel_size=kernel_size, pad=pad)
        
        #unet = UNet_online(in_channels=2, n_classes=2, depth=5, wf=3, kernel_size = kernel_size, padding= int((kernel_size - 1)/2), 
        #                    batch_norm=True, batch_norm_switchable=False, up_mode='upsample')


        unet = NestedUNet(num_classes=2, input_channels=2, deep_supervision=deep_supervision, padding=pad, batch_norm_switchable=switch_norm)
        #unet = UNet_upsample(num_classes=2, input_channels=2, padding=pad)

        #unet = UNet_3Plus(num_classes=2, input_channels=2, kernel_size=kernel_size, padding=pad)

        unet.train()
        unet.to(device)
        print('parameters:', sum(param.numel() for param in unet.parameters()))  
        

        """ Select loss function """
        loss_function = torch.nn.CrossEntropyLoss(reduction='none')
        #kwargs = {"alpha": 0.5, "gamma": 2.0, "reduction": 'none'}
        #loss_function = kornia.losses.FocalLoss(**kwargs)

        
        """ ****** DISTANCE LOSS FUNCTIONS *** CHECK IF NEED TO BE (X,Y,Z) format??? """
        # DC_and_HDBinary_loss, BDLoss, HDDTBinaryLoss
        #loss_function = DC_and_HDBinary_loss(); dist_loss = 1
        #loss_function = HDDTBinaryLoss(); dist_loss = 1
        #if both:
        #    loss_function_2 = torch.nn.CrossEntropyLoss()
            
            
        #loss_function = FocalLoss(apply_nonlin=None, alpha=None, gamma=2, balance_index=0, smooth=1e-5, size_average=True)
        #loss_function = DC_and_CE_loss()
        
        
        #loss_function = L.lovasz_hinge()
        

        """ Select optimizer """
        #lr = 1e-3; milestones = [20, 50, 100]  # with AdamW *** EXPLODED ***
        lr = 1e-3; milestones = [5, 50, 100]  # with AdamW
        lr = 1e-5; milestones = [20, 100]  # with AdamW slow down
        #lr = 1e-6; milestones = [1000]  # with AdamW slow down

        #optimizer = torch.optim.SGD(unet.parameters(), lr = lr, momentum=0.90)
        #optimizer = torch.optim.Adam(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
        optimizer = torch.optim.AdamW(unet.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01, amsgrad=False)



        """ Add scheduler """

        # *** IF WITH ADAM CYCLING ==> set cycle_momentum == False
        #scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=1e-7, max_lr=1e-3, step_size_up=2000, step_size_down=None, 
        #                                              mode='triangular', gamma=1.0, scale_fn=None, scale_mode='cycle', 
        #                                              cycle_momentum=False, base_momentum=0.8, max_momentum=0.9, last_epoch=-1)
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.1, last_epoch=-1)
        resume = 0
        
        """ Prints out all variables in current graph """
        # Required to initialize all
        batch_size = 4;      
        test_size = 0.1
        """ Load training data """
        print('loading data')   
        
        """ Specify transforms """
        #transforms = initialize_transforms(p=0.5)
        #transforms = initialize_transforms_simple(p = 0.5)
        transforms = 0
        
        sp_weight_bool = 0
        
        
        
        if FP_16:
            scaler = GradScaler()
        
 
    

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
  
        """ Restore per batch """
        train_loss_per_batch = check['train_loss_per_batch']
        train_jacc_per_batch = check['train_jacc_per_batch']
        val_loss_per_batch = check['val_loss_per_batch']
        val_jacc_per_batch = check['val_jacc_per_batch']
     
        
        if HD:
            train_ce_pb  = check['train_ce_pb']
            train_hd_pb  = check['train_hd_pb']
            train_dc_pb  = check['train_dc_pb']
            alpha = check['alpha']
            
        
     
        
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
 
    
 
    """ switch to using HD loss """

    if cont_HD:
        
           loss_function = HDDTBinaryLoss(); dist_loss = 1     



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
            
            
        ### FOR VALIDATION DATA
         
        # if branch_bool:
        #     match = set(idx_valid) & set(all_branch_idx)
        #     idx_valid = idx_valid + list(match) + list(match)
            
    
    """ For plotting ground truth """
    #idx_train = np.sort(idx_train)
    ### also need to set shuffle == False below



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
    for cur_epoch in range(len(train_loss_per_epoch), 10000): 
         
         unet.train()         

         loss_train = 0
         jacc_train = 0   
         
         
         ce_train = 0; dc_train = 0; hd_train = 0;
         
             
         for param_group in optimizer.param_groups:
              #param_group['lr'] = 1e-6   # manually sets learning rate
              cur_lr = param_group['lr']
              lr_plot.append(cur_lr)
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
                
                """ Speed testing
                        
                    batch 8, no workers, 50, == 19.74
                    ''' 4 workers ==> 9.2
                
                """
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
                     (2) normalizes + appl other transforms on GPU
                     (3) ***add non-blocking???
                     ***INPUT LABELS MUST BE < 255??? or else get CudNN error
                     
                """
                inputs, labels = transfer_to_GPU(batch_x, batch_y, device, mean_arr, std_arr)
                inputs = inputs[:, 0, ...]


                """ zero the parameter gradients"""
                optimizer.zero_grad()       
                
                """ forward + backward + optimize """
                if not FP_16:
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
            
                        train_ce_pb.append(loss_ce.cpu().data.numpy())
                        train_dc_pb.append(loss_seg_dice.cpu().data.numpy())
                        train_hd_pb.append(loss_hd.cpu().data.numpy())
                        
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
               
                else:
                    with autocast():
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
                
                            train_ce_pb.append(loss_ce.cpu().data.numpy())
                            train_dc_pb.append(loss_seg_dice.cpu().data.numpy())
                            train_hd_pb.append(loss_hd.cpu().data.numpy())
                            
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
                            
                        
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()


                    #loss.backward()
                    #optimizer.step()

                
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


                """ Plot for ground truth """
                # output_train = output_train.cpu().data.numpy()            
                # output_train = np.moveaxis(output_train, 1, -1)              
                # seg_train = np.argmax(output_train[0], axis=-1)  
                  
                # # convert back to CPU
                # batch_x = batch_x.cpu().data.numpy() 
                # batch_y = batch_y.cpu().data.numpy() 
 
                # plot_trainer_3D_PYTORCH_snake_seg(seg_train, seg_train, batch_x[0], batch_x[0], batch_y[0], batch_y[0],
                #                            s_path, iterations, plot_depth=8)

                
                
    
         train_loss_per_epoch.append(loss_train/iter_cur_epoch)
         train_jacc_per_epoch.append(jacc_train/iter_cur_epoch)        
         
         
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

                        """ Plot for ground truth """
                        # if loss.cpu().data.numpy() > 0.1:
                        #     batch_x_val = batch_x_val.cpu().data.numpy() 

                        #     print(loss.cpu().data.numpy())
                        #     for b_x, b_y, out in zip(batch_x_val, batch_y_val, output_val):
                        #         seg_val = np.argmax(out, axis=-1)  
                        #         plot_trainer_3D_PYTORCH_snake_seg(seg_val, seg_val, b_x, b_x, b_y, b_y,
                        #                                     s_path, val_idx, plot_depth=8)
                                
                        #         val_idx -= 1
                                
                        #     #zzz
    
    
                        #     val_idx += batch_size
                            



                        # if not unet.training:
                        #         for module in unet.modules():
                        #             if isinstance(module, SwitchNorm3d):
                        #             #if isinstance(module, torch.nn.modules.BatchNorm3d):
                        #                 print('setting track running stats to TRUE')
                        #                 #module.track_running_stats = True
                        #                 print(module.running_mean)
                        #                 print(module.running_var)
                        #                 break
                        #                 #               break
                                
                        """ Plot for ground truth """
                        # output_val = output_val.cpu().data.numpy()            
                        # output_val = np.moveaxis(output_val, 1, -1)              
                        # seg_val = np.argmax(output_val[0], axis=-1)  
                          
                        # # convert back to CPU
                        # batch_x_val = batch_x_val.cpu().data.numpy() 
                        # batch_y_val = batch_y_val.cpu().data.numpy() 
         
                        # plot_trainer_3D_PYTORCH_snake_seg(seg_val, seg_val, batch_x_val[0], batch_x_val[0], batch_y_val[0], batch_y_val[0],
                        #                             s_path, iterations, plot_depth=8)                
                      
                           
                   val_loss_per_eval.append(loss_val/iter_cur_epoch)
                   val_jacc_per_eval.append(jacc_val/iter_cur_epoch)       
                   

                  
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


              plot_metric_fun(train_loss_per_epoch, val_loss_per_eval, class_name='', metric_name='loss', plot_num=40)
              plt.figure(40); plt.savefig(s_path + 'loss_per_epoch_NO_LOG.png')         
                   
              
                
              """ Separate losses """
              if HD:
                  plot_cost_fun(train_ce_pb, train_ce_pb)                   
                  plt.figure(18); plt.savefig(s_path + '_global_loss_CE.png')
                  #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
                  plt.figure(25); plt.savefig(s_path + '_global_loss_LOG_CE.png')
                  plt.close('all')
    
                  plot_cost_fun(train_hd_pb, train_hd_pb)                   
                  plt.figure(18); plt.savefig(s_path + '_global_loss_HD.png')
                  #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
                  plt.figure(25); plt.savefig(s_path + '_global_loss_LOG_HD.png')
                  plt.close('all')
    
                  plot_cost_fun(train_dc_pb, train_dc_pb)                   
                  plt.figure(18); plt.savefig(s_path + '_global_loss_DC.png')
                  #plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
                  plt.figure(25); plt.savefig(s_path + '_global_loss_LOG_DC.png')
                  plt.close('all')                  
                  
                
              
              """ VALIDATION LOSS PER BATCH??? """
              plot_cost_fun(val_loss_per_batch, val_loss_per_batch)                   
              plt.figure(18); plt.savefig(s_path + '_VAL_global_loss_VAL.png')
              plt.figure(19); plt.savefig(s_path + '_VAL_detailed_loss_VAL.png')
              plt.figure(25); plt.savefig(s_path + '_VAL_global_loss_LOG_VAL.png')
              plt.close('all')
                
              
              plot_metric_fun(lr_plot, [], class_name='', metric_name='learning rate', plot_num=35)
              plt.figure(35); plt.savefig(s_path + 'lr_per_epoch.png') 

              """ Plot negative loss """
              if loss < 0:
                  plot_metric_fun(train_loss_per_epoch[cur_epoch - 5: -1], val_loss_per_eval[cur_epoch - 5: -1], class_name='', metric_name='loss', plot_num=36)
                  plt.figure(36); plt.savefig(s_path + 'loss_per_epoch_NEGATIVE.png')     
                  
                  # plot_metric_fun(train_loss_per_batch[iterations - (iter_cur_epoch * 5 * batch_size): -1], val_loss_per_eval[iterations - (iter_cur_epoch * 5): -1], class_name='', metric_name='loss', plot_num=37)
                  # plt.figure(37); plt.savefig(s_path + 'loss_NEGATIVE_ZOOM.png')                   

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
                
                
                'model_type': unet,
                'optimizer_type': optimizer,
                'scheduler_type': scheduler,
                
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                
                'train_loss_per_batch': train_loss_per_batch,
                'train_jacc_per_batch': train_jacc_per_batch,
                'val_loss_per_batch': val_loss_per_batch,
                'val_jacc_per_batch': val_jacc_per_batch,
                
                
                'train_ce_pb': train_ce_pb,
                'train_hd_pb': train_hd_pb,
                'train_dc_pb': train_dc_pb,
                'alpha': alpha,
                
                
                'train_loss_per_epoch': train_loss_per_epoch,
                'train_jacc_per_epoch': train_jacc_per_epoch,
                'val_loss_per_eval': val_loss_per_eval,
                'val_jacc_per_eval': val_jacc_per_eval,
                
                'plot_sens': plot_sens,
                'plot_sens_val': plot_sens_val,
                'plot_prec': plot_prec,
                'plot_prec_val': plot_prec_val,
                
                'lr_plot': lr_plot,
                
                 # newly added
                'mean_arr': mean_arr,
                'std_arr': std_arr,
                
                'batch_size': batch_size,  
                'sp_weight_bool': sp_weight_bool,
                'loss_function': loss_function,  
                'transforms': transforms  

                
                
                }, save_name)
     
                

                
               
              
              