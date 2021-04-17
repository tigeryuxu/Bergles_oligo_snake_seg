#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 23:56:54 2021

@author: user
"""
import matplotlib.pyplot as plt
import numpy as np
import glob, os
import datetime
import time
from sklearn.model_selection import train_test_split


import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
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
 
import re
import sps
import matplotlib
import tifffile as tiff


""" Set globally """
matplotlib.use('Qt5Agg')
plt.rc('xtick',labelsize=12)
plt.rc('ytick',labelsize=12)
#plt.rcParams['figure.dpi'] = 300
ax_title_size = 18
leg_size = 16

check_path = './(2) Checkpoint_unet_MEDIUM_filt7x7_b8_HD_INTERNODE_SPS_optimizer/'; dilation = 1; deep_supervision = False; tracker = 1; HISTORICAL = 0;

directories = glob.glob(check_path + "*/")
directories.sort(key = natsort_key1)

print(directories)

all_val_metrics = []
for id_fold, folder in enumerate(directories):
    
    s_path = folder;
    """ Save dataframe as pickle """
    import pickle
    # with open(s_path + 'all_trees.pkl', 'wb') as f:
    #     pickle.dump(all_trees, f)
    
    # Load back pickle
    with open(s_path + 'all_trees.pkl', 'rb') as f:
          all_trees = pickle.load(f)
            
    
    """ Also have to load in image to see size """
    input_path = '/media/user/storage/Data/(1z) paranode_identification/Training_COMBINED_DATASET/test_image/';
    images = glob.glob(os.path.join(input_path,'*input.tif*'))
    images.sort(key = natsort_key1)
    examples = [dict(input=i, paranodes=i.replace('_input.tif','_paranodes_from_MAT.tif'), 
                 val_mat=i.replace('_input.tif','.mat'), val_im=i.replace('_input.tif','_seeds.tif')) for i in images]

    input_im = tiff.imread(examples[id_fold]['input'])
    input_im =  np.moveaxis(input_im, 0, 2)
    
    plot_max(input_im, ax=-1)
    
    
    ### get name for saving later
    filename = examples[id_fold]['input'].split('/')[-1].split('.')[0:-1]
    filename = '.'.join(filename)        
    
    
    
    """ Also have to load in original paranodes """
    paranodes = tiff.imread(examples[id_fold]['paranodes'])
    paranodes =  np.moveaxis(paranodes, 0, 2)    
    
    
    
    
    """ Linearize and plot raw image of internodes (colored) """
    internodes_list = []
    
    im_internodes = np.zeros(np.shape(input_im));
    for tree_id, tree in enumerate(all_trees):
        
        ### skip if only 3 nodes in tree
        if len(tree) < 4:
            continue;
        
        ### loop through all the nodes in the current tree starting from root            
        for r_id, row in enumerate(tree.coords):                
            if r_id == 0:  internode = row
            else:
                internode = np.concatenate((internode, row))
                
        internodes_list.append(internode)   # add complete internode to list
        
        im_internodes[internode[:, 0], internode[:, 1], internode[:, 2]] = tree_id + 1

        # if tree_id > 27:
        #     plot_max(im_internodes, ax=-1)

    color_im = np.copy(im_internodes)

    print("Saving after first iteration")
    color_im = convert_matrix_to_multipage_tiff(color_im)
    imsave(s_path + filename + '_overall_output_1st_iteration_COLOR_INTERNODES_RAW.tif', np.asarray(color_im * 255, dtype=np.uint8))
       


    
    """ WHAT TO ADD: """
    
    # DO CORRECT RESOLUTION
    # SUBTRACT OUT PARANODES TO CREATE SEPARATE OBJECTS
    
    
    import more_itertools as mit  
    """ First remove all paranodes that coloc with > 1 internode (b/c means will create weird artifact on subtraction) """

    cleaned_paranodes = np.zeros(np.shape(input_im))
    bw_deg = paranodes > 0
    labelled = measure.label(bw_deg)
    cc_paranodes = measure.regionprops(labelled)
    
    
    for paranode in cc_paranodes:
        
        coords = paranode['coords']
        
        vals = im_internodes[coords[:, 0], coords[:, 1], coords[:, 2]]   ### find all values colocalized with current paranode
        
        if len(np.unique(vals)) > 2:   ### skip if colocalizes with > 1 internode
            continue;
        else:
            cleaned_paranodes[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
            #print('yo')
        
        
        
    """ Linearize the tree so that each row is just a single pixel list from root to end
            also eliminate trees that are only a single pixel
    
    """
    internodes_list = []
    
    im_internodes = np.zeros(np.shape(input_im)); id_internode = 0;
    for tree_id, tree in enumerate(all_trees):
        
        ### skip if only 3 nodes in tree
        if len(tree) < 4:
            continue;
        
        
        ### loop through all the nodes in the current tree starting from root            
        for r_id, row in enumerate(tree.coords):                
            if r_id == 0:  internode = row
            else:
                internode = np.concatenate((internode, row))
                

        
        
        """ Also check if the paranode will cut this linear segment into pieces! Do this by:
            
                (1) Find coordinates of overlap between internodes and paranode (using image of paranodes)
                (2) Find WHERE in the list of coordinates of internode those overlapped coords are located (i.e. near start/end? or middle)
                
                (3) if near middle, then subtract out +/- 3 pixels to cleave into multiple fragments!
            
            """
            
        loc_paranodes = np.where(cleaned_paranodes[internode[:, 0], internode[:, 1], internode[:, 2]])[0]
        ### find consecutive groups of numbers (indicating regions of paranodes)                      
        list_groups = [list(group) for group in mit.consecutive_groups(loc_paranodes)]
        
        ### see if any of the matched groups do NOT include coords that are at the start/end of the internodes
        id_break_points = []
        for group in list_groups:
           
            if np.min(group) < 20 or np.max(group) > len(internode) - 20 or len(group) < 4:   ### must also be at least > 4 pixels overlap
                continue;
                
            else:   ### otherwise, means might actually be in the middle matched! 
                    ### in which case, delete a few pixels and split the internodes list at that location!
                    
                mid_to_elim = group[round(len(group)/2)]
                id_break_points.append(mid_to_elim)
        
        ### now start splitting up internodes:
        if len(id_break_points) > 0:
            
            start = 0
            for break_p in id_break_points:
                
                ### NOTE: if the gap between the current break point and the last break point is super small, then is the NODE OF RANVIER SPACE!!! LEAVE IT EMPTY!!!
                if start + 20 > break_p:
                    #zzz
                    #inter_segment = internode[start - 4:break_p - 2]
                    start = break_p + 2
                    continue;
                
                else:   ### otherwise, continue with splitting by breakpoint
                    inter_segment = internode[start:break_p - 2]     
                    start = break_p + 2  ### start at breakpoint + 2, so total space in between (gap) is 4 pixels
                    
                    internodes_list.append(inter_segment)   # add complete internode to list
                    
                    im_internodes[inter_segment[:, 0], inter_segment[:, 1], inter_segment[:, 2]] = id_internode + 1  ### plot for debug  
                    id_internode += 1
                    
            ### add final segment:
            inter_segment = internode[start: - 1]     
            internodes_list.append(inter_segment)   # add complete internode to list
            
            im_internodes[inter_segment[:, 0], inter_segment[:, 1], inter_segment[:, 2]] = id_internode + 1  ### plot for debug  
            id_internode += 1
                
        else:
            internodes_list.append(internode)   # add complete internode to list
        
            im_internodes[internode[:, 0], internode[:, 1], internode[:, 2]] = id_internode + 1            
            id_internode += 1
        


    ### check if paranode subtraction was successful
    #paranodes[paranodes > 0] = 1   
    #im_internodes[im_internodes > 0] = 1   
    #plot_max(paranodes + im_internodes, ax=-1) 
    
    
    
    
    
    plot_max(im_internodes, ax=-1)
    print("Saving cleaned internodes")
    im = convert_matrix_to_multipage_tiff(im_internodes)
    imsave(s_path + filename + '_cleaned_internodes.tif', np.asarray(im * 255, dtype=np.uint8))            
        
    
    
    """ More cleaning needed:
        
            - if segments overlap near same coordinates ==> keep only the longest ones!!!
            
            - also, in above, if subtraction of paranode creates 2 very small segments (or even 1 really small segment)... maybe reconsider...
                OR fixed above partially, by just removing all paranodes that coloc with > 1 internode (b/c means will create weird artifact on subtraction)
        
        
        """
    
    
    
 
    """ Figure out which internodes have 0, 1, or >= 2 paranodes attached so can combine in next step
                ==> and make annotated dataframe based on confidence!!! 
                
                
                Note: do NOT use "clean_paranodes" for this section!
                    ***means should have nothing that is "0" paranodes
    """       
    single_paranode = [];
    double_paranode = [];
    no_paranodes = [];
    for internode in internodes_list:
        
        loc_paranodes = np.where(paranodes[internode[:, 0], internode[:, 1], internode[:, 2]])[0]
        ### find consecutive groups of numbers (indicating regions of paranodes)                      
        list_groups = [list(group) for group in mit.consecutive_groups(loc_paranodes)]
        
        ### see if any of the matched groups do NOT include coords that are at the start/end of the internodes
        id_break_points = []
        
        num_paranodes = len(list_groups)
        
        
        #num_paranodes = 0;
        # for group in list_groups:
           
        #     if np.min(group) < 50 or len(group) < 4:   ### must also be at least > 4 pixels overlap
        #         num_paranodes += 1
        
        #     if np.max(group) > len(internode) - 50 or len(group) < 2:
        #         num_paranodes += 1
                
        ### If:
            # 0 paranodes == delete
            # 1 paranodes == put into single paranode
            # 2 paranodes == put into double
            # 3 paranodes... should not be possible? B/c we're only looking at head/butt
   
        if num_paranodes == 0: no_paranodes.append(internode)
        elif num_paranodes == 1:  single_paranode.append(internode)
        elif num_paranodes >= 2: double_paranode.append(internode)

    
    # im_cleaned = np.zeros(np.shape(input_im))
    # for idx_i, internode in enumerate(no_paranodes): im_cleaned[internode[:, 0], internode[:, 1], internode[:, 2]] = idx_i + 1
    # im = convert_matrix_to_multipage_tiff(im_cleaned)
    # imsave(s_path + filename + '_0_paranode_internodes.tif', np.asarray(im * 255, dtype=np.uint8))           
    
    im_1_paranode = np.zeros(np.shape(input_im))
    for idx_i, internode in enumerate(single_paranode): im_1_paranode[internode[:, 0], internode[:, 1], internode[:, 2]] = idx_i + 1
    im = convert_matrix_to_multipage_tiff(im_1_paranode)
    imsave(s_path + filename + '_1_paranode_internodes.tif', np.asarray(im * 255, dtype=np.uint8))  

    im_2_paranode = np.zeros(np.shape(input_im))
    for idx_i, internode in enumerate(double_paranode): im_2_paranode[internode[:, 0], internode[:, 1], internode[:, 2]] = idx_i + 1
    im = convert_matrix_to_multipage_tiff(im_2_paranode)
    imsave(s_path + filename + '_2_paranode_internodes.tif', np.asarray(im * 255, dtype=np.uint8))          
    
    
    
    """ For internodes in the "0" paranode list, see if they match to any 1_paranode segments """
    copy_single = np.copy(single_paranode)
    combined_singles = []
    
    id_ignore = []
    for id_0, inter_0 in enumerate(single_paranode):
        
        if id_0 in id_ignore: continue
        
        ### expand neighborhood so easier to match indices
        dil_inter_0 = np.unique(np.vstack(expand_coord_to_neighborhood(inter_0, lower=1, upper=2)), axis=0)
        
        
        matched_len = []; matched_id = []
        for id_m, inter_1 in enumerate(copy_single):
            if id_m == id_0: continue   ### skip self-comparison
            
            nrows, ncols = inter_1.shape
            dtype={'names':['f{}'.format(i) for i in range(ncols)],
                   'formats':ncols * [inter_1.dtype]}

            C = np.intersect1d(inter_1.view(dtype), dil_inter_0.view(dtype))
            C = C.view(inter_1.dtype).reshape(-1, ncols)
            
            
            if len(C) > 20:  ### ONLY COMBINED IF > 4 pixels matched    
            
            
                """ HACK: this still might cause errors... sometimes maybe 1 segment will match with several others??? three-way tie??? """
            
                #print(len(C))
                matched_len.append(len(C))
                matched_id.append(id_m)


        # if len(matched_len) > 1:
        #     print('yo')

        ### combine with the LARGEST matched segment!
        if len(matched_len) > 0:
            longest_match = np.argmax(matched_len)
            
            inter_1_match = copy_single[matched_id[longest_match]]
            
            combined = np.concatenate((single_paranode[id_0], inter_1_match))
            
            remove_dup = np.unique(combined, axis=0)
            
            ### add to list of combined_singles
            combined_singles.append(remove_dup)
            
            
            ### delete from list so can't re-concatenate multiple times
            id_ignore.append(id_0)
            id_ignore.append(matched_id[longest_match])
        else:
            combined_singles.append(inter_0)
            


    all_internodes = np.concatenate((combined_singles, double_paranode))
    
    full_internodes = np.zeros(np.shape(input_im))
    for idx_i, internode in enumerate(all_internodes): full_internodes[internode[:, 0], internode[:, 1], internode[:, 2]] = idx_i + 1
    im = convert_matrix_to_multipage_tiff(full_internodes)
    imsave(s_path + filename + '_FULL_CLEANED_internodes.tif', np.asarray(im * 255, dtype=np.uint8))          
    
    
    
    
    
    
    """ Final(?) cleaning step ==> if an internode exists almost fully WITHIN another internode, then delete it from the list """   
    to_drop = []
    iter_num = 0
    for id_0, internode in enumerate(all_internodes):
        dil_inter_0 = np.unique(np.vstack(expand_coord_to_neighborhood(internode, lower=1, upper=2)), axis=0)
                    
        for id_m, inter_1 in enumerate(all_internodes):
            tmp = np.zeros(np.shape(full_internodes))
            if id_m == id_0: continue   ### skip self-comparison
            
            nrows, ncols = inter_1.shape
            dtype={'names':['f{}'.format(i) for i in range(ncols)],
                   'formats':ncols * [inter_1.dtype]}

            C = np.intersect1d(inter_1.view(dtype), dil_inter_0.view(dtype))
            C = C.view(inter_1.dtype).reshape(-1, ncols)
            
            
            """ Drop around ratio 0.2 - 0.25 is bad ones!!!"""
            
            if len(C) > 4:  ### ONLY COMBINED IF > 4 pixels matched
                # print(len(C)) 
                # print(len(inter_1))
                
                tmp[inter_1[:, 0], inter_1[:, 1], inter_1[:, 2]] = 1
                tmp[internode[:, 0], internode[:, 1], internode[:, 2]] = 2
                tmp[C[:, 0], C[:, 1], C[:, 2]] = 3
                
                
                #plot_max(tmp, ax=-1)
                
                
                
                
                if len(C)/len(inter_1) > 0.2:
                        """ If this ratio is larger, means that C is getting close to length of inter_1, so must drop inter_1!!! """
                        to_drop.append(id_m)
                        #print(len(C)/len(inter_1))
                        #print(len(inter_1))
                #     plot_max(tmp, ax=-1)
                
                # if iter_num == 18:
                #     zzz
                
                #     zzz
                # iter_num += 1
                

                
    ### drop as needed
    all_internodes = np.delete(all_internodes, to_drop)

    
        
    
    
    
    
    
                

    
    
    """ Load in validation data
    
    
                ***USE DENSE IMAGE FROM NEW DATASET FOR TRUE VALIDATION!!!
                    - will need to scale it first!!!
                    
    
    """
    
    
    
    
    #val_path = '/media/user/storage/Data/(1) snake seg project/Traces files/paranodes for RL send MALIKA_1/'
    
    #val_path = '/media/user/storage/Data/(1z) paranode_identification/Training_COMBINED_DATASET/test_image/';
    
    
    val_path = input_path + '/';
    
    
    
    
    ### (1) First load .mat file
    mat_name = examples[id_fold]['val_mat']
    mat_name = mat_name.split('/')[-1].split('.')[0:-1]
    mat_name = '.'.join(mat_name)
    
    
    import scipy.io as sio
    mat_contents = sio.loadmat(val_path + mat_name)
    
    list_sheaths = mat_contents['s_coords_RESIZED'][0]
    
    list_sheaths = list_sheaths['sheath_coords']
    overlay_output = np.zeros(np.shape(input_im))
    
    val_metrics = {'num_match': [], 'len_diff': [], 'prop_diff': [], 'overlap_diff': [], 'total_len_diff': [], 'total_prop_diff': [], }
    for id_s, sheath in enumerate(list_sheaths):
        val_im = np.zeros(np.shape(input_im))
        sheath = np.vstack(sheath)
        
        sheath = sheath - 1;   ### subtract one to start indexing from 0!!!
        
        
        val_im[sheath[:, 0], sheath[:, 1], sheath[:, 2]] = 1
        
        
        overlay_output[sheath[:, 0], sheath[:, 1], sheath[:, 2]] = 1
        
        ### then search through all the ouput segmentations to find ones that match
        num_match = 0; len_diffs = []; prop_diffs = []
        total_lens = 0;  matched = 0
        for seg in all_internodes:
            
            # expand so search is larger
            expand = np.unique(np.vstack(expand_coord_to_neighborhood(seg, lower=1, upper=2)), axis=0)
            
            """ MIGHT GET TOO LARGE b/c of building up previous end points, so need to ensure crop """
            expand[np.where(expand[:, 0] >= np.shape(input_im)[0]), 0] = np.shape(input_im)[0] - 1
            expand[np.where(expand[:, 1] >= np.shape(input_im)[1]), 1] = np.shape(input_im)[1] - 1
            expand[np.where(expand[:, 2] >= np.shape(input_im)[2]), 2] = np.shape(input_im)[2] - 1    
            
            
            
            
            if len(np.where(val_im[expand[:, 0], expand[:, 1], expand[:, 2]])[0])  > 20:   # match at least 5 pixels???
                #print(len(np.where(val_im[expand[:, 0], expand[:, 1], expand[:, 2]])[0]))
                
                matched = 1
            
            
                #val_im[expand[:, 0], expand[:, 1], expand[:, 2]] = 1

                overlay_output[seg[:, 0], seg[:, 1], seg[:, 2]] = 2
                
                
                ### calculate % of overlap and NUMBER of overlapped
                num_match += 1
                len_diff = len(sheath) - len(seg)   # if negative, then segmentation longer than ground truth
                len_diffs.append(len_diff)
                
                prop_diff = len_diff/len(sheath)
                prop_diffs.append(prop_diff)
                
                total_lens += len(seg); 
                
                #plot_max(overlay_output, ax=-1)
                #print(prop_diffs)
                
                
                
        val_metrics['num_match'].append(num_match)
        val_metrics['len_diff'].append(len_diffs)     
        val_metrics['prop_diff'].append(prop_diffs)     

        if matched:
            total_len_diff = len(sheath) - total_lens
            total_prop_diff = total_len_diff/len(sheath)

            val_metrics['total_len_diff'].append(total_len_diff)     
            val_metrics['total_prop_diff'].append(total_prop_diff)                      
                    
            
    all_val_metrics.append(val_metrics)
            
    plt.close('all')
    
    arr_props = np.asarray(val_metrics['total_prop_diff'])
    plt.figure()
    plt.hist(arr_props)      ### negative values mean that segmentation longer than ground truth
    
    perc = len(np.where((arr_props > -0.4) & (arr_props < 0.6))[0])   # within +/- 50% proportion
    total_perc = perc/len(arr_props)
    print("total_prop_diff:" + str(total_perc))
    

    arr_props = np.asarray(val_metrics['total_len_diff'])
    plt.figure()
    plt.hist(arr_props)      ### negative values mean that segmentation longer than ground truth
    
    perc = len(np.where((arr_props > -50) & (arr_props < 50))[0])   # within +/- 50% proportion
    total_perc = perc/len(arr_props)
    print("total_len_diff:" + str(total_perc))



    # arr_props = np.asarray(val_metrics['prop_diff'])
    # plt.figure()
    # plt.hist(arr_props)      ### negative values mean that segmentation longer than ground truth
    
    # perc = len(np.where((arr_props > -50) & (arr_props < 50))[0])   # within +/- 50% proportion
    # total_perc = perc/len(arr_props)
    # print(total_perc)


    """ Also plot number of segments that are missed entierly???"""

    all_props = np.asarray(val_metrics['prop_diff'])
    
    num_missed = 0
    for prop in all_props:
        
        
        if len(prop) == 0:
            num_missed += 1
            print('empty')
    

    prop_missed = num_missed/len(all_props)
    print(prop_missed)



    """ Also save the ground truth validation image"""
    ground_truth = np.zeros(np.shape(input_im))
    for id_s, sheath in enumerate(list_sheaths):
        val_im = np.zeros(np.shape(input_im))
        sheath = np.vstack(sheath)
        sheath = sheath - 1;   ### subtract one to start indexing from 0!!!
        ground_truth[sheath[:, 0], sheath[:, 1], sheath[:, 2]] = id_s
                


    im = convert_matrix_to_multipage_tiff(ground_truth)
    imsave(s_path + filename + '_FULL_ground_truth.tif', np.asarray(im * 255, dtype=np.uint8))          
    

    #zzz

    """ Also remove internodes that go off the image screen??? """
    
    
    """ Also remove internodes that wind up back at the cell soma!!! """
        
        
        



""" Plot cumulative validation metrics """

# combine into one plot

combined_prop = []
for val_metrics in all_val_metrics:
    #combined_prop.append(val_metrics['total_prop_diff'])

    combined_prop = np.concatenate((combined_prop, val_metrics['total_prop_diff']))


plt.figure()
plt.hist(combined_prop)      ### negative values mean that segmentation longer than ground truth


perc = len(np.where((combined_prop > -0.4) & (combined_prop < 0.6))[0])   # within +/- 50% proportion
total_perc = perc/len(combined_prop)
print("total_prop_diff:" + str(total_perc))






