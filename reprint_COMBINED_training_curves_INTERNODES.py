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
    # s_path = './(51) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Hd_loss_balance_repeat_MARCC/'; HD = 1; alpha = 1;
    #s_path = './'    

    s_path = '/media/user/storage/Data/(1) snake seg project/Backup checkpoints INTERNODE_linear_tracing/'; HD = 1; alpha = 1; sps_bool = 1;
    
    
    #s_path = './(4) Checkpoint_unet_MEDIUM_filt7x7_b8_HD_INTERNODE_sps_NEW_HD_loss/'; HD = 1; alpha = 1; sps_bool = 1;
    
    
    deep_sup = 0; dist_loss = 0
    
    directories = glob.glob(s_path + "*/")
    directories.sort(key = natsort_key1)
    
    print(directories)


    all_val_metrics = {'folder':[], 'cID_metric':[], 'HD_metric':[], 'iter_nums':[]}
    for id_fold, folder in enumerate(directories):    
        """ TO LOAD OLD CHECKPOINT """
        # Read in file names
        onlyfiles_check = glob.glob(os.path.join(folder,'*_REPLOTTED'))
        onlyfiles_check.sort(key = natsort_key1)
        
        ### only if a "replotted" file exists in the folder
        if len(onlyfiles_check) > 0:

            ### also get ALL checkpoint files, so can plot correct iterations to match
            all_check_files = glob.glob(os.path.join(folder,'check_*'))
            all_check_files.sort(key = natsort_key1)
            list_check_iter = []
            for file in all_check_files[:-1]:
                split = file.split('check_')[-1]
                num_check = split.split('_REPLOTTED')[0]
                list_check_iter.append(int(num_check))
                   
            

            """ Find last checkpoint """       
            last_file = onlyfiles_check[-1]
            split = last_file.split('check_')[-1]
            num_check = split.split('.')
            checkpoint = num_check[0]
            checkpoint = 'check_' + checkpoint
        
            print('restoring weights from: ' + checkpoint)
            check = torch.load(folder + checkpoint, map_location=device)
        
            tracker = check['tracker']
            
            #if len(all_check_files) > 15:
            #        zzz
            
            all_val_metrics['folder'].append(folder)
            all_val_metrics['cID_metric'].append(tracker.val_jacc_per_eval)
            all_val_metrics['HD_metric'].append(tracker.val_ce_pb)
            
            all_val_metrics['iter_nums'].append(list_check_iter)
 
  

    
s_path = './Internode_plots/'
""" Plot output combined graphs """

all_val_metrics['folder']
#folders_to_plot = [0, 1, 2, 3, 4, 5, 6]

folders_to_plot = [0, 1]
leg_to_plot = ['(1) HD no sps', '(2) HD sps']

folders_to_plot = [1]
leg_to_plot = ['(2) HD sps']

""" (1) interesting graphs (HD vs. not HD, vs. cID) """
fontsize = 14;
rot = 0
plt.figure(figsize=(5, 4)); 
legend = []
for id_n in folders_to_plot:

    cID_curve = all_val_metrics['cID_metric'][id_n]
    iter_n = all_val_metrics['iter_nums'][id_n]
    
    iter_n = np.asarray(iter_n)/10000
    
    plt.plot(iter_n, cID_curve, 'k')
    #plt.plot(cID_curve)
    
    

    name = all_val_metrics['folder'][id_n][2:6]  +   all_val_metrics['folder'][id_n][60:]
    
    legend.append(name)
    
plt.legend(leg_to_plot, loc='lower right', frameon=False, fontsize=fontsize-2)
plt.ylabel('cIDice', fontsize=fontsize);  plt.yticks(fontsize=fontsize - 2); plt.ylim([0.5, 1.0])
plt.xlabel('Iterations ($10^4$)', fontsize=fontsize); plt.xticks(fontsize=fontsize - 2, rotation=rot); #plt.xlim([0, 180000])

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(s_path + 'compare_SPS_cID_metric_INTERNODES.png', dpi=300)
    
    
plt.figure(figsize=(5, 4)); 
legend = []
for id_n in folders_to_plot:

    cID_curve = all_val_metrics['HD_metric'][id_n]
    iter_n = all_val_metrics['iter_nums'][id_n]
    
    iter_n = np.asarray(iter_n)/10000
    
    plt.plot(iter_n, cID_curve, 'k')
    #plt.plot(cID_curve)

    name = all_val_metrics['folder'][id_n][2:6]  +   all_val_metrics['folder'][id_n][60:]
    
    legend.append(name)
    
plt.legend(leg_to_plot, frameon=False, fontsize=fontsize-2)
plt.ylabel('Haussdorf dist (px)', fontsize=fontsize); plt.yticks(fontsize=fontsize - 2);  plt.ylim([0, 10])
plt.xlabel('Iterations ($10^4$)', fontsize=fontsize);  plt.xticks(fontsize=fontsize - 2, rotation=rot);  #plt.xlim([0, 180000])

ax = plt.gca()
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(s_path + 'compare_SPS_HD_metric_INTERNODES.png', dpi=300)


