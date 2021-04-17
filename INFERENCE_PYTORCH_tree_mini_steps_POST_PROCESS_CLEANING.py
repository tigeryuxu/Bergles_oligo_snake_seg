# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger
### TO DO:
    # (4) paranodes      
      
    Different trainings:
        ### ***larger model on COMPUTE CANADA  
        ### ***exclude 1st image   
        ### ***seed every 5
    
    
    
    
(1) edges are being cropped weird... ==> maybe skeletonize first and THEN remove pixels??? to prevent less weird cuts???
(2) what happens if runs over itself???
(3) some crop seeds still being cut short...
    


(4) ***maybe batch process the mini-steps???
    
    
    
        
    
    ### ERRORS:
        - unlucky bend might bend back from edge of image frame... somehow need to prevent that...
                ==> example: crop 6, 29
            
        - myelin is cutting off wayyy too much. Need to allow propagation to new non-myelin segments beyond!!!
                
    
    
    ### To add:
        - better paranode detection/association at the end
        - ***identify all un-associated myelin segments at the end and associate based on proximity of nearest?
            also can try re-running snake seg from tips??? to search for thin cytosolic segments???
    
    
    
    
    
"""
import matplotlib
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 
#matplotlib.use('Qt5Agg')


""" Libraries to load """
import numpy as np
from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.IO_func import *
import glob, os
import datetime
import time
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from layers.UNet_pytorch import *
from layers.UNet_pytorch_online import *
from sklearn.model_selection import train_test_split

from matlab_crop_function import *
from off_shoot_functions import *
from tree_functions import *
from skimage.morphology import skeletonize_3d, skeletonize
from skimage.transform import rescale, resize, downscale_local_mean

import tifffile as tifffile

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 



""" Define GPU to use """
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

""" Decide if use pregenerated seeds or not """
pregenerated = 1
        
"""  Network Begins: """
tracker = 0
#check_path = './(48) Checkpoint_nested_unet_SPATIALW_medium_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/'
#check_path = './(49) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/'; dilation = 1; deep_supervision = False;
#check_path = './(49) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/train with 1e6 after here/'; dilation = 1; deep_supervision = False;


#check_path = './(51) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Hd_loss_balance_repeat_MARCC/'; dilation = 1; deep_supervision = False; tracker = 1;


#check_path = './(52) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Hd_loss_balance_NO_1st_im/'; dilation = 1; deep_supervision = False; tracker = 1;


#check_path = './(53) Checkpoint_unet_medium_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/'; dilation = 1; deep_supervision = False; tracker = 1;


#check_path = './(54) Checkpoint_nested_unet_medium_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/'; dilation = 1; deep_supervision = False; tracker = 1;


#check_path = './(55) Checkpoint_unet_LARGE_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/'; dilation = 1; deep_supervision = False; tracker = 1;

#check_path = './(56) Checkpoint_unet_nested_LARGE_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/';  dilation = 1; deep_supervision = False; tracker = 1;


#check_path = './(59) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/';  dilation = 1; deep_supervision = False; tracker = 1;

check_path = '/media/user/storage/Data/(1) snake seg project/Backup checkpoints/(59) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/'; dilation = 1; deep_supervision = False; tracker = 1;

#check_path = './(60) Checkpoint_unet_COMPLEX_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/'; dilation = 1; deep_supervision = False; tracker = 1;


#check_path = './(62) Checkpoint_unet_COMPLEX_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/';  dilation = 1; deep_supervision = False; tracker = 1;


check_path = './(82) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_HD_sps_only_cytosol/';  dilation = 1; deep_supervision = False; tracker = 1; HISTORICAL = 0;  combine_tree = 0
load_myelin = 1;
#z_size = 32


check_path = './(80) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_HD_only_cytosol/';dilation = 1; deep_supervision = False; tracker = 1; HISTORICAL = 0;  combine_tree = 0
#check_path = './(81) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_no_HD_only_cytosol/';dilation = 1; deep_supervision = False; tracker = 1; HISTORICAL = 0;  combine_tree = 0
#check_path = './(83) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_no_HD_sps_only_cytosol/'; dilation = 1; deep_supervision = False; tracker = 1; HISTORICAL = 0;  combine_tree = 0

#check_path = './(84) Checkpoint_unet_MEDIUM_filt_7x7_b4_type_dataset_NO_1st_im_no_HD_sps_CYTOSOL_and_MYELIN/'; dilation = 1; deep_supervision = False; tracker = 1; HISTORICAL = 0;  combine_tree = 0




""" Historical with 2 step training data """
z_size = 48
#check_path = '/media/user/storage/Data/(1) snake seg project/Backup checkpoints/(65) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_2_step_REAL_HISTORICAL/';  dilation = 1; deep_supervision = False; tracker = 1; HISTORICAL = 1;

""" No historical, 2 step matched """
# HISTORICAL = 0;
# storage_path = '/media/user/storage/Data/(1) snake seg project/Backup checkpoints/'
# check_path = storage_path + '(66) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_2_step/'; dilation = 1; deep_supervision = False; tracker = 1;




""" For neuron """
# z_size = 32
# HISTORICAL = 0;
# storage_path = '/media/user/storage/Data/(1) snake seg project/Backup checkpoints/'
# check_path = storage_path + '(67) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_4_step_NEURON/'; dilation = 1; deep_supervision = False; tracker = 1;
# combine_tree = 1;


""" For neuron """   ### used for presentation???
z_size = 32
HISTORICAL = 0;
storage_path = '/media/user/storage/Data/(1) snake seg project/Backup checkpoints/'
check_path = storage_path + '(68) Checkpoint_unet_MEDIUM_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_4_step_NEURON_DILATE_2/'; dilation = 2; deep_supervision = False; tracker = 1;
combine_tree = 1;
load_myelin = 0;



""" For neuron large network """
# z_size = 32
# HISTORICAL = 0;
# storage_path = '/media/user/storage/Data/(1) snake seg project/Backup checkpoints/'
# check_path = storage_path + '(68)_2 Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_4_step_NEURON_DILATE_2/'; dilation = 2; deep_supervision = False; tracker = 1;
# combine_tree = 1;

#s_path = check_path + 'TEST_inference_185437_last_first_CLEANED_correct_scale/'


s_path = check_path + 'TEST_inference_185437_last_first_CLEANED_correct_scale_100000_short_first_no_edge_ANIMATION/'

#s_path = check_path + 'TEST_inference_185437_CLEANED_correct_scale_100000_short_first_EDGE_REMOVAL_different_subtractor/'




s_path = check_path + 'TEST_inference_last_first_MYELIN_edge_remove_last_first/'
#s_path = check_path + 'TEST_inference_185437_last_first_CLEANED_HISTORICAL_NEURON/'
#s_path = check_path + 'TEST_inference_185437_shortest_first_REAL_troubleshoot/'

#s_path = check_path + 'TEST_inference_158946_shortest_first_CLEANED_3_FULL_AUTO/'

#s_path = check_path + 'TEST_inference_344383_shortest_first_CLEANED_3_NEURON/'
#s_path = check_path + 'TEST_inference_344383_shortest_first_CLEANED_3/'

#s_path = check_path + 'FULL_AUTO_TEST_inference_158946_last_first_NEURON/'

#s_path = check_path + 'FULL_AUTO_TEST_inference_158946_last_first_REAL_2_CARE_RESTORED_FULL_AUTO/'

#s_path = check_path + 'TEST_inference_185437_last_first_NEURON_INTENSITY_troubleshoot/'

#s_path = check_path + 'TEST_inference_185437_shortest_first_NEURON_TWO/'
#s_path = check_path + 'TEST_inference_185437_last_first_NEURON/'

try:
    # Create target Directory
    os.mkdir(s_path)
    print("Directory " , s_path ,  " Created ") 
except FileExistsError:
    print("Directory " , s_path ,  " already exists")

#input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation large_25px_NEW/';  seed_crop_size=100; seed_z_size=80
""" CORRECTLY SCALED!!! """
#input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation large REDO multiply scaling/';  seed_crop_size=100; seed_z_size=80

#input_path = 'E:/7) Bergles lab data/Traces files/seed generation large_25px/'

#input_path = '/media/user/storage/Data/(1) snake seg project/CARE_flipped_reconstruction/to segment/'


#input_path = '/media/user/storage/Data/(1) snake seg project/BigNeuron data/gold166/Training data neurons/test/';  seed_crop_size=150; seed_z_size=80

#input_path = '/media/user/storage/Data/(1) snake seg project/BigNeuron data/gold166/Training data neurons/test/';  seed_crop_size=80; seed_z_size=50


""" Su-Jeong neuron """
input_path = '/media/user/Seagate Portable Drive/Bergles lab data 2021/Su_Jeong_neurons/Training data SOLANGE/seed generation large SWC/';   seed_crop_size=150; seed_z_size=50



""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*input.tif*'))
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)
examples = [dict(input=i,truth=i.replace('input.tif','truth.tif'), cell_mask=i.replace('input.tif','input_cellMASK.tif'),
                 seeds = i.replace('input.tif', 'seeds.tif'),
                 myelin = i.replace('input.tif', 'input_overall_output_1st_iteration_COLOR_INTERNODES.tif')) for i in images]

counter = list(range(len(examples)))  # create a counter, so can randomize it

""" TO LOAD OLD CHECKPOINT """
onlyfiles_check = glob.glob(os.path.join(check_path,'check_*'))
onlyfiles_check.sort(key = natsort_key1)
    
""" Find last checkpoint """       
last_file = onlyfiles_check[-1]
split = last_file.split('check_')[-1]
num_check = split.split('.')
checkpoint = num_check[0]
checkpoint = 'check_' + checkpoint

print('restoring weights of checkpoint: ' + str(num_check[0]))
check = torch.load(check_path + checkpoint, map_location=device)
unet = check['model_type']
unet.load_state_dict(check['model_state_dict']) 


if not tracker:
    mean_arr = check['mean_arr'];  std_arr = check['std_arr']
else:
    ### IF LOAD WITH TRACKER
    tracker = check['tracker']
    mean_arr = tracker.mean_arr; std_arr = tracker.std_arr


""" Set to eval mode for batch norm """
unet.eval();   unet.to(device)

input_size = 80

crop_size = int(input_size/2)



""" Change to scaling per crop??? """
original_scaling = 0.2076;
target_scale = 0.20;
scale_factor = original_scaling/target_scale;
scaled_crop_size = round(input_size/scale_factor);
scaled_crop_size = math.ceil(scaled_crop_size / 2.) * 2  ### round up to even num

scale_for_animation = 0; animation_order = [];

for i in range(len(examples)):            
    
    
        """ NEW: load in myelin segments """
        myelin_df = pd.DataFrame()
        if load_myelin:
            myelin_im = tifffile.imread(examples[i]['myelin'])
            myelin_im =  np.moveaxis(myelin_im, 0, 2)
            
            cc_myelin = measure.regionprops(myelin_im, intensity_image=myelin_im)
            
            ### sort and eliminate all small segments (which are paranodes)
            for cc in cc_myelin:
                if len(cc['coords']) > 10:
                        print(len(cc['coords']))
                        myelin_df = myelin_df.append({'myelin_cc': cc, 'myelin_val': cc['max_intensity'], 'cytosol_matched': []}, ignore_index=True)
                    

            ### dilate the myelin im a bit
            myelin_im_dil = dilate_by_ball_to_binary(myelin_im, radius=1)

    
    

        """ (1) Loads data as sorted list of seeds """
        
        sorted_list, input_im, width_tmp, height_tmp, depth_tmp, overall_coord, all_seeds, all_seeds_no_50 = load_input_as_seeds(examples, im_num=i,
                                                                                                                                 pregenerated=pregenerated, s_path=s_path,
                                                                                                                                 seed_crop_size=seed_crop_size, seed_z_size=seed_z_size)   
        input_name = examples[i]['input']
        filename = input_name.split('/')[-1].split('.')[0:-1]
        filename = '.'.join(filename)
        
      
        """ scale input im for animations """
        if scale_for_animation:
            
             input_im_rescaled = convert_matrix_to_multipage_tiff(input_im)   
             input_im_rescaled = resize(input_im_rescaled, (input_im_rescaled.shape[0] * scale_for_animation, input_im_rescaled.shape[1] * scale_for_animation, input_im_rescaled.shape[2]  * scale_for_animation))
             
            
        """ add seeds to form roots of tree """
        """ (1) First loop through and turn each seed into segments at branch points 
            (2) Then add to list with parent/child indices
        """
    
        """ only 50 """
        #all_seeds[all_seeds !=  50] = 0;
        bw = np.copy(all_seeds); bw[bw > 0] = 1
        labelled = measure.label(bw)
        cc = measure.regionprops(labelled, intensity_image=all_seeds)

        all_trees = []
        for seg in cc:
            coords = seg['coords']
            
            if pregenerated:
                idx_start = np.where(all_seeds[coords[:, 0], coords[:, 1], coords[:, 2]] == 50)[0]
            else:
                idx_start = np.where(all_seeds[coords[:, 0], coords[:, 1], coords[:, 2]] == 2)[0]
  
            print(idx_start)
    
            if len(idx_start) == 0:
                print('SKIPPED, couldnt find initial starting point')
                continue
            elif len(idx_start) > 1:
                
                print('more than 1 start point found')
                
                """ If more than 1 start point, use the one closest to either beginning or end of list of coords """
                if np.min(len(coords) - idx_start) < 3:
                    idx_start = idx_start[np.argmin(len(coords) - idx_start)]
                elif np.min(idx_start) < 3:
                    idx_start = idx_start[np.argmin(idx_start)]
                else:
                    idx_start = idx_start[0]
                        
                
                
            start_coords = coords[idx_start]
            if len(coords) < 4:
                continue;
            ordered, tree_order, discrete_segs = order_coords_from_start(coords, start_coords)
            
            """ else add to tree """
            tree_df = pd.DataFrame()
            tree_df = treeify_nx(tree_df, discrete_segs, tree_idx=0, disc_idx=0, parent=-1, start_tree=1)            
    
            all_trees.append(tree_df)
            
            
        track_trees = np.zeros(np.shape(input_im))    
        num_tree = 0
        
        
        """ Concatenate into one big tree """
        if combine_tree:
            combined_trees = all_trees[0]
            for small_t in all_trees[1:]:
                
                small_t.cur_idx = small_t.cur_idx + np.max(combined_trees.cur_idx) + 1
                for p_id, parent in enumerate(small_t.parent):
                    if parent != -1:
                        parent = parent + np.max(combined_trees.cur_idx) + 1   ### add values of the maximum of full list of trees
                        
                    ### also add this value to the children
                    children = small_t.iloc[p_id].child
                    if len(children) > 0:
                        children = list(np.asarray(children) + np.max(combined_trees.cur_idx) + 1)
                        small_t.child[p_id] = children
               
                        
                    small_t.parent[p_id] = parent;
                
                combined_trees = pd.concat([combined_trees, small_t], ignore_index=True)
            
            all_trees = []
            all_trees.append(combined_trees)
        
        """ Create box in middle to only get seed colocalized with middle after splitting branchpoints """
        center_cube = create_cube_in_im(width=10, input_size=input_size, z_size=z_size)
        #small_center_cube = create_cube_in_im(width=8, input_size=input_size, z_size=z_size)
        center_cube_pm = create_cube_in_im(width=8, input_size=input_size * 2, z_size=z_size * 2)
        small_cube = create_cube_in_im(width=5, input_size=input_size * 2, z_size=z_size * 2) 
        
        resize_crop = 0;
        for it, tree in enumerate(all_trees):
             matplotlib.use('Agg')
                        
             
             #tree = all_trees[6]
             """ Keep looping until everything has been visited """  
             iterator = 0;
             while np.asarray(tree.visited.isnull()).any():   
                 
                ### convert center cube back to original size
                if resize_crop == 1:
                   center_cube_pm = create_cube_in_im(width=8, input_size=input_size * 2, z_size=z_size * 2)
                   small_cube = create_cube_in_im(width=5, input_size=input_size * 2, z_size=z_size * 2) 
                   resize_crop = 0

                """ Get coords at node
                        ***go to node that is SHORTEST PATH LENGTH AWAY FIRST!!!
                """                  
                unvisited_indices = np.where(tree.visited.isnull() == True)[0]
                
                """ Go to index of SHORTEST PATH FIRST """
                # all_lengths = []
                # for ind in unvisited_indices:                 
                #     parent_coords = get_parent_nodes(tree, ind, num_parents=100, parent_coords = [])
                
                #     if len(parent_coords) > 0:
                #         parent_coords = np.vstack(parent_coords)
                
                #     all_lengths.append(len(parent_coords))                 
                # node_idx = unvisited_indices[np.argmin(all_lengths)]
                
                # print(all_lengths)
                
                
                    
                """ Or, just go to very last position """
                node_idx = unvisited_indices[-1]
                
                
                """ Save order so can generate animation later """
                animation_order.append(node_idx)
                
                ### SKIP IF NO END_BE_COORD FROM 
                if np.isnan(tree.end_be_coord[node_idx]).any():
                     tree.visited[node_idx] = 1; iterator += 1; continue;
                    
                
                
                cur_coords, cur_be_start, cur_be_end, centroid_start, centroid_end, parent_coords = get_next_coords(tree, node_idx, num_parents=20)
                
                
                
                    
                
                """ NEW addition: Feb. 23, 2021
                
                        instead of deleting out prev_seg components OR myelin, we just find out here whether these
                        endpoints are falling into that territory! If so, we give these endpoints a value:
                                visited = 0 ==> means colocalized with previous segment
                                visited = -3 ==> means colocalized with myelin segment
                                
                        then, at the end of the analysis, we can prune these segments accordingly
                """

                ### insert entire current segment and get a crop of it to compare with later
                x_n = int(cur_be_end[0]); y_n = int(cur_be_end[1]); z_n = int(cur_be_end[2])
                cur_seg = np.zeros(np.shape(input_im))
                cur_seg[cur_coords[:, 0], cur_coords[:, 1], cur_coords[: , 2]] = 1
                crop_seed, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(cur_seg, y_n, x_n, z_n, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                
               
                crop_input, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(input_im, y_n, x_n, z_n, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                
                             
                if load_myelin:
                    print('checking myelin association')
                    
                    ### if this matches, then end point is in a myelin segment!
                    crop_myelin_dil, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(myelin_im_dil, y_n, x_n, z_n, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                
                    bool_myelin = myelin_im_dil[cur_be_end[0], cur_be_end[1], cur_be_end[2]]
                    
                    
                    ### matched section must also be AT LEAST 10 pixels long
                    bool_myelin_len = myelin_im_dil[cur_coords[:, 0], cur_coords[:, 1], cur_coords[: , 2]]
                    bool_myelin_len = len(np.where(bool_myelin_len)[0])
                    
                    ### if bool_myelin is true, then print out the image for debugging
                    ### also set visited == -3
                    if bool_myelin and bool_myelin_len >= 10:
                        
                        plot_save_max_project(fig_num=20, im=crop_input, max_proj_axis=-1, title='input_im', 
                                              name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '(1)_input_im.png', pause_time=0.001)                         
                        plot_save_max_project(fig_num=20, im=crop_seed, max_proj_axis=-1, title='myelin_block', 
                                              name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '(2)_seed.png', pause_time=0.001) 
                        plot_save_max_project(fig_num=20, im=crop_myelin_dil, max_proj_axis=-1, title='myelin_block', 
                                              name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '(3)_myelin_BLOCK.png', pause_time=0.001) 
                        
                        tree.visited[node_idx] = -3; iterator += 1; continue;
                        
                    
                ### also check if end point is located within a previously segmented location

                all_coords = show_tree_FAST_drop_index(tree, drop_id=node_idx)  ### eliminate current coords from that previous coords
                im_prev = np.zeros(np.shape(input_im))
                im_prev[all_coords[:, 0], all_coords[:, 1], all_coords[:, 2]] = 1
                                
                prev_crop, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(im_prev, y_n, x_n, z_n, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                
                
                prev_crop = dilate_by_ball_to_binary(prev_crop, radius=2)
                bool_prev = prev_crop[crop_size - 1, crop_size - 1, int(z_size/2 - 1)]
                
                if bool_prev:
                        plot_save_max_project(fig_num=20, im=crop_input, max_proj_axis=-1, title='input_im', 
                                              name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '(1)_input_im.png', pause_time=0.001)                         
                        plot_save_max_project(fig_num=20, im=crop_seed, max_proj_axis=-1, title='myelin_block', 
                                              name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '(2)_seed.png', pause_time=0.001) 
                        plot_save_max_project(fig_num=20, im=prev_crop, max_proj_axis=-1, title='myelin_block', 
                                              name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '(3)_prev_BLOCK.png', pause_time=0.001) 
                        
                        tree.visited[node_idx] = 0; iterator += 1; continue;
                                            
                
                
                
                # if load_myelin:
                #     
                    
                    
                #     # if num_tree == 1:
                #     #     zzz
                        
                #     myelin_crop, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(myelin_im, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
            
            
                #     ### might need to dilate out only_coloc a bit??? to make sure matches myelin???
                    
                #     coloc_myelin = dilate_by_ball_to_binary(output_PYTORCH, radius=2)
                #     val_myelin = myelin_crop[coloc_myelin > 0]
                #     val_myelin = val_myelin[val_myelin > 0]                    
                #     if len(val_myelin) > 0:

                    
                #         ### find all unique values, and only keep the one that is MOST colocalized
                        
                #         from collections import Counter
                #         unique = np.unique(val_myelin)
                #         counts = Counter(val_myelin)
                #         most_match = counts.most_common(1)   ### gets the most common
                        
                #         myelin_val = most_match[0][0]   ### get value of myelin sheath matched
                #         num_matched = most_match[0][1]  ### get length of pixels matched
                        
                #         # skip if too short of a match
                #         # if num_matched < 5:
                #         #     continue
                        
                        
                #         # otherwise, subtract out the myelin segment
                #         #myelin_dil = dilate_by_ball_to_binary(myelin_crop, radius=2)
                #         #only_coloc[myelin_dil > 0] = 0;
                        
                        
                #         ### Actually, just add the myelin to the crop_seed_full, which will be subtracted from the image later in a smarter way!!!
                #         myelin_crop[myelin_crop > 0] = 1
                #         myelin_dil = dilate_by_ball_to_binary(myelin_crop, radius=2)
                #         crop_seed_full = crop_seed_full + myelin_dil
                #         crop_seed_full[crop_seed_full > 0] = 1
                        
                        
                #         ### add values to myelin dataframe
                    
                    

                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                
                

                
                """ Skip if on edges of actual image """
                st = centroid_start
                if st[0] > input_im.shape[0] - 5 or st[0] < 0 + 5 or st[1] > input_im.shape[1] - 5 or st[1] < 0 + 5 or st[2] > input_im.shape[2] - 2 or st[2] < 0 + 2:
                    tree.visited[node_idx] = 1; iterator += 1; continue;

                """ Order coords """
                ### SKIP IF TOO SHORT for mini-steps
                if len(cur_coords) == 1:
                     tree.visited[node_idx] = 1; iterator += 1; continue;

                """ Split into mini-steps """
                ### Step size:
                step_size = 5; 
                step_size_first = step_size          
                if len(cur_coords) <= step_size:  ### KEEP GOING IF ONLY SMALL SEGMENT
                      step_size_first = len(cur_coords) - 1
            
                output_tracker = np.zeros(np.shape(input_im))
                    
                for step in range(step_size_first, len(cur_coords), step_size):
                    
                      """ Setup to get historical crops """
                      past_im = []
                      plt.close('all')
                      plt.figure(100);
                      hist_step_size = 20   # pixels
                      if HISTORICAL and len(parent_coords) > 0:
                          if len(parent_coords) > 0:
                                im_hist = np.zeros(input_im.shape, dtype=np.int32)
                          """ Include current coords in it """
                          hist_coords = np.concatenate((parent_coords, cur_coords[0:step]))
                          
                          
                          """ OR, start from parents only!!! which is what's done in training??? """
                          #hist_coords = parent_coords
                          
                          
                          """ Training actually just uses the ENTIRE cur_coords!!! """
                          full_hist_coords = np.concatenate((parent_coords, cur_coords))
                          
                          im_hist[full_hist_coords[:, 0], full_hist_coords[:, 1], full_hist_coords[:, 2]] = 1 
        
                            
                          for p_idx, hist_id in enumerate(range(len(hist_coords) - (3), 0, -hist_step_size)):
                              
                              x_pa = int(hist_coords[hist_id, 0]); y_pa =  int(hist_coords[hist_id, 1]); z_pa = int(hist_coords[hist_id, 2])
                              """ use centroid of object to make seed crop """
                              crop_hist, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(input_im, y_pa, x_pa, z_pa, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                
                              #cur_seg_im[x,y,z] = 2   ### IF WANT TO SEE WHAT THE CROP IS CENTERING ON
                              crop_seed_hist, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(im_hist, y_pa, x_pa, z_pa, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                                                      
    
                              """ Dilate the seed by sphere 1 to mimic training data """
                              crop_seed_hist = skeletonize_3d(crop_seed_hist)
                              """ Make sure no gaps in crop_seed """
                              crop_seed_hist, output_non_bin = bridge_end_points(crop_seed_hist, bridge_radius=2)
                              crop_seed_hist = dilate_by_ball_to_binary(crop_seed_hist, radius=dilation)

                              """ Check nothing hanging off edges in seed  """
                              crop_seed_hist = check_resized(crop_seed_hist, z_size, width_max=input_size, height_max=input_size)
        

                              crop_seed_hist[crop_seed_hist > 0] = 255        
                              
                              plt.subplot(2, 10, p_idx + 1)
                              ma = plot_max(crop_hist, ax=-1, plot=0)
                              plt.imshow(ma); plt.axis('off')
                              
                              
                              plt.subplot(2, 10, (p_idx + 11))
                              ma = plot_max(crop_seed_hist, ax=-1, plot=0)
                              plt.imshow(ma); plt.axis('off')
                              
                    
                              past_im.append(crop_hist)
                              past_im.append(crop_seed_hist)
            
                              if len(past_im) == 20:
                                  break
                              
                          plt.savefig(s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) + '_HISTORICAL.png')
                          
                          
                      if HISTORICAL:
                          ### append blanks otherwise
                          while len(past_im) < 20:
                              past_im.append(np.zeros([crop_size * 2, crop_size * 2, z_size]))
                      
                          past_im = np.asarray(past_im)

                    
                      """ Regardless of historical """
                      x = int(cur_coords[step, 0]); y = int(cur_coords[step, 1]); z = int(cur_coords[step, 2])
                                            
                      cur_seg_im = np.zeros(np.shape(input_im))   # maybe speed up here by not creating the image every time???
                      cur_seg_im[cur_coords[0:step, 0], cur_coords[0:step, 1], cur_coords[0:step, 2]] = 1
                      cur_seg_im[x, y, z] = 1    # add the centroid as well
                      
                      # add the parent
                      if len(parent_coords) > 0:
                          cur_seg_im[parent_coords[:, 0], parent_coords[:, 1], parent_coords[:, 2]] = 1
                          
                          
                      """ use centroid of object to make seed crop """
                      crop, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(input_im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                
                      #cur_seg_im[x,y,z] = 2   ### IF WANT TO SEE WHAT THE CROP IS CENTERING ON
                      crop_seed, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(cur_seg_im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                      
                      
                      
                      
                      """ Dilate the seed by sphere 1 to mimic training data """
                      crop_seed = skeletonize_3d(crop_seed)
                      
                      
                      
                      """ Make sure no gaps in crop_seed """
                      crop_seed, output_non_bin = bridge_end_points(crop_seed, bridge_radius=2)
                      
                      crop_seed = dilate_by_ball_to_binary(crop_seed, radius=dilation)
        
                      """ Check nothing hanging off edges in seed  """
                      crop_seed = check_resized(crop_seed, z_size, width_max=input_size, height_max=input_size)


                      """ Send to segmentor for INFERENCE """
                      crop_seed[crop_seed > 0] = 255  
                      output_PYTORCH = UNet_inference_PYTORCH(unet,np.asarray(crop, np.float32), crop_seed, mean_arr, std_arr, device=device, deep_supervision=deep_supervision, past_im=past_im)
            
                      """ Since it's centered around crop, ensure doesn't go overboard """
                      output_PYTORCH[boundaries_crop == 0] = 0


                      """ REMOVE EDGE """
                      # dist_xy = 5; dist_z = 2
                      # edge = np.zeros(np.shape(crop_seed)).astype(np.uint8)
                      # #edge[dist_xy:pm_crop_size * 2-dist_xy, dist_xy:pm_crop_size * 2-dist_xy, dist_z:pm_z_size-dist_z] = 1
                      # edge[dist_xy:crop_size * 2-dist_xy, dist_xy:crop_size * 2-dist_xy, dist_z:z_size-dist_z] = 1
                      # edge = np.where((edge==0)|(edge==1), edge^1, edge)
                      # output_PYTORCH[edge == 1] = 0


                      # """ SAVE max projections"""
                      # plot_save_max_project(fig_num=5, im=crop_seed, max_proj_axis=-1, title='crop seed dilated', 
                      #                       name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(2)_seed.png', pause_time=0.001)
                      # plot_save_max_project(fig_num=2, im=output_PYTORCH, max_proj_axis=-1, title='segmentation', 
                      #                       name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(3)_segmentation.png', pause_time=0.001)
                      # plot_save_max_project(fig_num=3, im=crop, max_proj_axis=-1, title='input', 
                      #                       name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(1)_input_im.png', pause_time=0.001)


                      ### HACKY - figure out how to get sizes to be better
                      prev_seg = output_tracker[box_xyz[0]:box_xyz[1], box_xyz[2]:box_xyz[3], box_xyz[4]:box_xyz[5]] 
                      output_tracker[box_xyz[0]:box_xyz[1], box_xyz[2]:box_xyz[3], box_xyz[4]:box_xyz[5]] = prev_seg + output_PYTORCH[box_over[0]:prev_seg.shape[0] + box_over[0], box_over[2]:prev_seg.shape[1] + box_over[2], box_over[4]:prev_seg.shape[2] + box_over[4]]
            
                    
                """ SAVE max projections"""
                plot_save_max_project(fig_num=5, im=crop_seed, max_proj_axis=-1, title='crop seed dilated', 
                                      name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(2)_seed.png', pause_time=0.001)
                plot_save_max_project(fig_num=2, im=output_PYTORCH, max_proj_axis=-1, title='segmentation', 
                                      name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(3)_segmentation.png', pause_time=0.001)
                plot_save_max_project(fig_num=3, im=crop, max_proj_axis=-1, title='input', 
                                      name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(1)_input_im.png', pause_time=0.001)
       


                
                """ ALSO HAVE TO RESET CUR_BE_END to align with current location of mini-step and NOT actual end of segment!!!
                """
                #cur_be_end = np.vstack(expand_coord_to_neighborhood([cur_coords[step]], lower=1, upper=2))
                cur_coords, cur_be_start, cur_be_end, centroid_start, centroid_end, parent_coords = get_next_coords(tree, node_idx, num_parents=50)

                """ Link any missing coords due to be subtractions """
                # if len(cur_coords) > 3:
                #     cur_coords = connect_nearby_px(cur_coords)
                
                x = int(centroid_end[0]); y = int(centroid_end[1]); z = int(centroid_end[2])
                cur_seg_im = np.zeros(np.shape(input_im))   # maybe speed up here by not creating the image every time???
                cur_seg_im[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = 1
                
                # add the parent
                # if len(parent_coords) > 0:
                #     cur_seg_im[parent_coords[:, 0], parent_coords[:, 1], parent_coords[:, 2]] = 1                
 
        
                ### Define size of larger crop:
                pm_crop_size = crop_size * 2
                pm_z_size = z_size * 2
                output_tracker[output_tracker > 0] = 1
                output_PYTORCH, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(output_tracker, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
            
                ### if not big enough, recrop with larger
                mult = 3
                while np.any(cur_be_start[0] - box_xyz[0] < 0) or np.any(cur_be_start[1] - box_xyz[2] < 0) or np.any(cur_be_start[2] - box_xyz[4] < 0) or  np.any(box_xyz[1] - cur_be_start[0] < 0) or np.any(box_xyz[3] - cur_be_start[1] < 0) or np.any(box_xyz[5] - cur_be_start[2] < 0):
                    pm_crop_size = crop_size * mult
                    pm_z_size = z_size * 3
                
                    output_tracker[output_tracker > 0] = 1
                    output_PYTORCH,  box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(output_tracker, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
                    resize_crop = 1
                    
                    center_cube_pm = create_cube_in_im(width=8, input_size=input_size * mult, z_size=z_size * 3)
                    
                    small_cube = create_cube_in_im(width=5, input_size=input_size * mult, z_size=z_size * 3)
                    
                    mult += 1
                    
                    ### Don't let it get TOO crazy big
                    if mult >= 8:
                        break
                                              
                ### Don't let it get TOO crazy big
                if mult >= 8:
                   tree.visited[node_idx] = 1; print('Finished'); iterator += 1;               
                   continue                      



                


                crop_seed, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(cur_seg_im, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
                crop_seed = skeletonize_3d(crop_seed)     
                
                
                
                """ Make sure no gaps in crop_seed """
                crop_seed, output_non_bin = bridge_end_points(crop_seed, bridge_radius=2)
                #crop_seed = dilate_by_ball_to_binary(crop_seed, radius=dilation)
                #crop_seed[crop_seed > 0] = 255

                """ Get separate full crop size """
                 #parent_coords = np.vstack(parent_coords)
                if len(parent_coords)> 0:
                    #parent_coords = connect_nearby_px(parent_coords)
                    
                    cur_seg_im_full = np.zeros(np.shape(input_im))   # maybe speed up here by not creating the image every time???
                    cur_seg_im_full[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = 1
                    cur_seg_im_full[parent_coords[:, 0], parent_coords[:, 1], parent_coords[:, 2]] = 1
                 
                    
                     
                    crop_seed_full, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(cur_seg_im_full, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
                    crop_seed_full = skeletonize_3d(crop_seed_full)     
                    
                    """ Make sure no gaps in crop_seed """
                    crop_seed_full, output_non_bin = bridge_end_points(crop_seed_full, bridge_radius=2)
                    #crop_seed_full = dilate_by_ball_to_binary(crop_seed_full, radius=dilation)
                    
                else:
                    crop_seed_full = crop_seed
                    
                """ Things to fix still:
                         ***circle instead of cube subtraction??? ==> b/c creating bad cut-offs right now
                    """
  

  
                """ ***FIND anything that has previously been identified
                    ***EXCLUDING CURRENT CROP_SEED
                """
                all_segs = show_tree_FAST(tree)
                                  
                ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
                for cur_tree in all_trees:
                    all_segs = np.concatenate((all_segs, show_tree_FAST(cur_tree)))
                im_prev = np.zeros(np.shape(input_im))
                im_prev[all_segs[:, 0], all_segs[:, 1], all_segs[:, 2]] = 1
                
                
                im_prev[im_prev > 0] = 1
                crop_prev, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(im_prev, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                                                      
                crop_prev = skeletonize_3d(crop_prev)
                crop_prev[crop_prev > 0] = 1
                
                
                ### EXCLUDE current crop seed
                im_sub = subtract_im_no_sub_zero(crop_prev, crop_seed)
                
                # with dilation
                #im_sub = im_sub + crop_seed
                
                im_dil = dilate_by_ball_to_binary(im_sub, radius=2)
                
                ### but add back in current crop seed (so now without dilation)
                im_dil = im_dil + crop_seed
                im_dil[im_dil > 0] = 1
                
  
                """ delete all small objects """
                # labelled = measure.label(output_PYTORCH)
                # cc = measure.regionprops(labelled); 
                # cleaned = np.zeros(np.shape(output_PYTORCH))
                # for seg in cc:
                #        coord = seg['coords']; 
                #        if len(coord) > 10:
                #            cleaned[coord[:, 0], coord[:, 1], coord[:, 2]] = 1
  
                # output_PYTORCH = cleaned
  
                
                """ add in crop seed and subtract later??? """
                output_PYTORCH = output_PYTORCH + crop_seed
                output_PYTORCH[output_PYTORCH > 0] = 1

                
                """ Keep only what is colocalized with the center """   ### Feb. 25th, 2021 ==> Tiger moved earlier!
                # (1) use old start_coords to find only nearby segments           
                # ***or just use center cube
                coloc_with_center = output_PYTORCH + center_cube_pm
                output_PYTORCH = find_overlap_by_max_intensity(bw=output_PYTORCH, intensity_map=coloc_with_center) 
                



                """ LINK EVERY END POINT TOGETHER USING line_nd """      
                output_PYTORCH = skeletonize_3d(output_PYTORCH)                    
                output_PYTORCH, output_non_bin = bridge_end_points(output_PYTORCH, bridge_radius=2)  ### Tiger: added this back in on Feb. 25th, because dilation ==> skel results in some missing pixels... that make fragments...
                
                
                only_coloc = output_PYTORCH


                plot_save_max_project(fig_num=3, im=output_non_bin, max_proj_axis=-1, title='output_be', 
                                      name=s_path + filename + '_Crop_' + str(num_tree) + '_' + str(iterator) +  '_step_' + str(step) +'_(4)_output_be.png', pause_time=0.001)                                              
                    


                """ Subtract out previously identified myelin segments
                        and also add this current cytsol value to corresponding myelin segment so 
                        can find out what is attached to what at the end
                        
                        
                        
                        
                    *** NEED TO ADD IN A SMART-STOP:
                            
                        
                        ### (1) remove all objects that are NOT connected to main center point object
                        
                        ### (2) output_PYTORCH - (prev_seg + myelin  ==> specifically myelin that has been matched in ouput_pytorch
                        
                        ### (3) then 
                        
                        ### (2) identify what is the
                                            
                
                """

                # if load_myelin:
                #     print('checking myelin association')
                    
                    
                #     # if num_tree == 1:
                #     #     zzz
                        
                #     myelin_crop, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(myelin_im, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
            
            
                #     ### might need to dilate out only_coloc a bit??? to make sure matches myelin???
                    
                #     coloc_myelin = dilate_by_ball_to_binary(output_PYTORCH, radius=2)
                #     val_myelin = myelin_crop[coloc_myelin > 0]
                #     val_myelin = val_myelin[val_myelin > 0]                    
                #     if len(val_myelin) > 0:

                    
                #         ### find all unique values, and only keep the one that is MOST colocalized
                        
                #         from collections import Counter
                #         unique = np.unique(val_myelin)
                #         counts = Counter(val_myelin)
                #         most_match = counts.most_common(1)   ### gets the most common
                        
                #         myelin_val = most_match[0][0]   ### get value of myelin sheath matched
                #         num_matched = most_match[0][1]  ### get length of pixels matched
                        
                #         # skip if too short of a match
                #         # if num_matched < 5:
                #         #     continue
                        
                        
                #         # otherwise, subtract out the myelin segment
                #         #myelin_dil = dilate_by_ball_to_binary(myelin_crop, radius=2)
                #         #only_coloc[myelin_dil > 0] = 0;
                        
                        
                #         ### Actually, just add the myelin to the crop_seed_full, which will be subtracted from the image later in a smarter way!!!
                #         myelin_crop[myelin_crop > 0] = 1
                #         myelin_dil = dilate_by_ball_to_binary(myelin_crop, radius=2)
                #         crop_seed_full = crop_seed_full + myelin_dil
                #         crop_seed_full[crop_seed_full > 0] = 1
                        
                        
                #         ### add values to myelin dataframe
                    
                    
                #         plot_save_max_project(fig_num=20, im=myelin_crop, max_proj_axis=-1, title='_final_added', 
                #                name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) +  '_step_' + str(step) + '_(5_5)_myelin.png', pause_time=0.001) 
    
                    
                    
                    
                    

                """ Subtract out previous segmentations, but do so smartly:
                    
                        (1) subtract out old seed and see what's left
                        (2) make into connected components
                        (3) then, add in prev_seg dilated ==> see what areas still (new areas) == 2 (i.e. overlapped with previous output still)
                        (4) if area has == 2, then keep it
                            ***this allows propagation of segmentation to OVERLAP IF there is new segments identified WITHIN the current segmentation
                    
                    """
                # sub_seed = subtract_im_no_sub_zero(output_PYTORCH, crop_seed_full)
                # #crop_prev[sub_seed == 0] = 0
                # added = np.copy(sub_seed)
                # added[im_dil > 0] = 2
                # added[added == 2] = 0
                
                # added = sub_seed + added
                # bw_added = np.copy(added)
                # bw_added[bw_added > 0] = 1
                
                # labelled = measure.label(bw_added)
                # cc = measure.regionprops(labelled, intensity_image=added); 
                # cleaned = np.zeros(np.shape(output_PYTORCH))
                # for seg in cc:
                #        coord = seg['coords']
                #        max_intensity = seg['max_intensity']
                #        if max_intensity == 2:
                #            cleaned[coord[:, 0], coord[:, 1], coord[:, 2]] = 1
  
                # output_PYTORCH = cleaned                
                # output_PYTORCH = output_PYTORCH + crop_seed ### add it back in
                    
                    
                

                
                """ moved here: subtract out past identified regions LAST to not prevent propagation
                
                    this is just a way to see how much of the image is NEW segmentation (i.e. is it worth it to even keep going here)
                
                """
                only_coloc[only_coloc > 0] = 1
                sub_seed = subtract_im_no_sub_zero(only_coloc, crop_seed)
                sub_seed = subtract_im_no_sub_zero(sub_seed, im_dil)
                
                
                
                
                
                """ skip if everything was subtracted out last time: """
                if np.count_nonzero(sub_seed) < 8:
                        tree.visited[node_idx] = 1; print('Finished')                     
                        plot_save_max_project(fig_num=10, im=np.zeros(np.shape(only_coloc)), max_proj_axis=-1, title='_final_added', 
                                name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) +  '_step_' + str(step) + '_(6)_zfinal_added.png', pause_time=0.001) 
                        iterator += 1;
                        continue                      
  
                else:
                    
                    """ Dilate to clean up a bit """
                    only_coloc = dilate_by_ball_to_binary(only_coloc, radius=2)                    


                    """ Make sure to skeletonize again """
                 
                    
                    only_coloc = skeletonize_3d(only_coloc)    
                    only_coloc[only_coloc > 0] = 1



                    
                

                    """ REMOVE EDGE end points b/c crop is bigger than possible to ever reach edges """
                    # dist_xy = 2; dist_z = 1
                    # edge = np.zeros(np.shape(output_PYTORCH)).astype(np.int64)
                    # edge[dist_xy:pm_crop_size * 2-dist_xy, dist_xy:pm_crop_size * 2-dist_xy, dist_z:pm_z_size-dist_z] = 1
                    # edge = np.where((edge==0)|(edge==1), edge^1, edge)
                    
                    # edge[degrees == 2] = 0
                    # degrees[edge > 0] = 0  
                    
                    
                    
                    # NEW: skeletonize and extract ordered graph
                    # then convert into tree
                    

                    pixel_graph, degrees, coordinates = bw_skel_and_analyze(only_coloc)

                    """ Also need to add in the starting point """
                    tmp_degrees = np.copy(degrees)
                    cur_start = np.copy(cur_be_start)
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    """TIGER:HACK -- IS THIS SCALE LINE BELOW CORRECT???"""
                    

                    
                    cur_start = scale_coord_to_full(cur_start, -1 * np.asarray(box_xyz), -1 * np.asarray(box_over))
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    ### check limits to ensure doesnt go out of frame
                    #cur_start = check_limits([cur_start], pm_crop_size * 2, pm_crop_size * 2, pm_z_size)[0]
                    
                    
                    ### HACK: fix how so end points cant leave frame
                    """ MIGHT GET TOO LARGE b/c of building up previous end points, so need to ensure crop """
                    # cur_start[np.where(cur_start[:, 0] >= pm_crop_size * 2), 0] = pm_crop_size * 2 - 1
                    # cur_start[np.where(cur_start[:, 1] >= pm_crop_size * 2), 1] = pm_crop_size * 2 - 1
                    # cur_start[np.where(cur_start[:, 2] >= pm_z_size), 2] = pm_z_size - 1
                    if cur_start[0] >= pm_crop_size * 2: cur_start[0] = pm_crop_size * 2 - 1
                    if cur_start[1] >= pm_crop_size * 2: cur_start[1] = pm_crop_size * 2 - 1
                    if cur_start[2] >= pm_z_size * 2: cur_start[2] = pm_z_size * 2 - 1
                    
                    
                    ### Then set degrees
                    tmp_degrees[cur_start[0], cur_start[1], cur_start[2]] = 20
                    
                    tmp_degrees[degrees == 0] = 0

                    loc_start = np.transpose(np.where(tmp_degrees == 20))
                    
                    
                    """ If not working (not matching a cur_start), then force the match by subtracting out the center
                            and then linking all end points together to the start point
                    """
                    if len(loc_start) == 0:
                          mid = cur_start
                        
                          mid_hood = expand_coord_to_neighborhood([mid], lower=2, upper=2 + 1)
                          mid_hood = np.vstack(mid_hood)

                          """ MIGHT GET TOO LARGE b/c of building up previous end points, so need to ensure crop """
                          mid_hood[np.where(mid_hood[:, 0] >= pm_crop_size * 2), 0] = pm_crop_size * 2 - 1
                          mid_hood[np.where(mid_hood[:, 1] >= pm_crop_size * 2), 1] = pm_crop_size * 2 - 1
                          mid_hood[np.where(mid_hood[:, 2] >= pm_z_size), 2] = pm_z_size - 1                        


                          tmp_degrees[mid_hood[:, 0], mid_hood[:, 1], mid_hood[:, 2]] = 0
                          degrees[mid_hood[:, 0], mid_hood[:, 1], mid_hood[:, 2]] = 0
                    
                          pixel_graph, tmp_degrees, coordinates = bw_skel_and_analyze(tmp_degrees)
                          coord_end = np.transpose(np.vstack(np.where(tmp_degrees == 1)))
                        
                        
                          match = 0
                          center = [pm_crop_size - 1, pm_crop_size - 1, int(pm_z_size/2 - 1)]
                          for coord in coord_end:
                            
                            print(np.linalg.norm(center - coord))
                            if np.linalg.norm(mid - coord) <= 8:
                                line_coords = line_nd(mid, coord, endpoint=False)
                                line_coords = np.transpose(line_coords)      
                                
                                degrees[line_coords[:, 0], line_coords[:, 1], line_coords[:, 2]] = 2
                                #degrees[center[0], center[1], center[2]] = 2               
                                
                                print('loop')
                                match = 1
                         
                            
                          """ If no end points matched, then just use the entire coord body itself and find closest point """
                          if not match:
                              all_dist = []; all_lines = [];
                              for coord in coordinates:
                                dist = np.linalg.norm(mid - coord)
                                if dist <= 8:
                                    line_coords = line_nd(mid, coord, endpoint=False)
                                    line_coords = np.transpose(line_coords)      
                                    
                                    
                                    #degrees[center[0], center[1], center[2]] = 2               
                                    
                                    all_dist.append(dist)
                                    all_lines.append(line_coords)
                                    
                                    
                                    
                              """ If still not matched after this, then skip, probably because too much was cut out from crop_prev """
                              if len(all_dist) == 0:
                                tree.visited[node_idx] = 1;
                                print('SKIPPED CROP_PREV'); iterator += 1; continue                             
                                     
                              close_idx = np.argmin(all_dist)
                              degrees[all_lines[close_idx][:, 0], all_lines[close_idx][:, 1], all_lines[close_idx][:, 2]] = 2
                             
                             

                            
                          degrees[mid[0], mid[1], mid[2]] = 20
                    else:
                    
                        # ### also set everything in neighborhood to be NOT end point
                        # for point in cur_start:
                        #     if degrees[point[0], point[1], point[2]] == 1 or degrees[point[0], point[1], point[2]] == 3:
                        #         degrees[point[0], point[1], point[2]] = 2
                        #         print('replace')
                                
                    
                        degrees[loc_start[0][0], loc_start[0][1], loc_start[0][2]] = 20



                    # if node_idx == 30:
                    #     zzz                    


                    """ To fix: 
                        
                        
                        DEC. 11th ==> start coord is being added at weird spot, so making small tiny segs
   
                        """

                    start = np.transpose(np.where(degrees == 20))[0]
                 
                    
                    """ Order the coordinates in degrees into discrete segments that can then be treeify-ed"""
                    ordered, discrete_segs, be_coords = order_skel_graph(degrees, start=start, end=[])
                    ordered_non_scaled = np.copy(ordered)
                    ordered = scale_coords_of_crop_to_full(ordered, box_xyz, box_over)

                    all_lengths = []
                    for idx_s, seg in enumerate(discrete_segs):
                        seg = scale_coords_of_crop_to_full(np.vstack(seg), box_xyz, box_over)
                        discrete_segs[idx_s] = seg
                        all_lengths.append(len(seg))
                    
                    """ If discrete segs is only single segment, then also need to break it up by the current crop point so that it's not
                            just one looong segment that will get set as visited == 1    
                            
                        OR only one discrete segment is longer than 4 pixels
                    """
                    if len(discrete_segs) == 1 or len(np.where(np.asarray(all_lengths) > 4)[0]) == 1:
                        idx_longest = np.argmax(all_lengths)
                        all_dist = [];
                        for coord in discrete_segs[idx_longest]:
    
                            all_dist.append(np.linalg.norm([x, y, z] - coord))
                            
                        min_id = np.argmin(all_dist)
                        divider = discrete_segs[idx_longest][min_id]                                  
                            
                        first_seg = discrete_segs[idx_longest][0:min_id + 1]
                        sec_seg = discrete_segs[idx_longest][min_id:]
                        
                        ### add these newly divided segments into discrete_segs
                        discrete_segs[idx_longest] = first_seg
                        discrete_segs.append(sec_seg)
                                
                    for idx_s, seg in enumerate(be_coords):
                        
                        
                        
                        """ TIGER-HACK: -- is this scaling statemement correct??? """
                        
                        
                        seg = scale_coords_of_crop_to_full(np.vstack(seg), box_xyz, box_over)
                        be_coords[idx_s] = seg


                    check_debug = np.zeros(degrees.shape, dtype=np.int32)
                    for idx_hist, row in enumerate(ordered_non_scaled):
                        check_debug[row[0], row[1], row[2]] = idx_hist + 1

                    plot_save_max_project(fig_num=10, im=check_debug, max_proj_axis=-1, title='_final_added', 
                                name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator)  + '_step_' + str(step) + '_(6)_zfinal_added.png', pause_time=0.001) 



                                      
                    
                    """ Detect t-shape and only keep one direction"""
                    
                    """ (1) First combine all branchpoint neighborhoods """
                    branchpoints = np.vstack(be_coords[1:])
                    combined_b = []
                    while len(branchpoints) > 0:
                        branchpoint = branchpoints[-1]
                        expanded = expand_coord_to_neighborhood([branchpoint], lower=3, upper=4)
                        
                        append_b = [branchpoint]
                        branchpoints = np.delete(branchpoints, -1, axis=0) # delete current branchpoint
                        
                        bp_copy = np.copy(branchpoints); to_del = []
                        for id_b, branch in enumerate(bp_copy):
                            if branch.tolist() in expanded:
                                append_b = np.concatenate((append_b, [branch]), axis=0)
                                to_del.append(id_b)
                        branchpoints = np.delete(branchpoints, to_del, axis=0) # delete current branchpoint that matched once already
                        
                        combined_b.append(append_b)
         
                    
                    """ (2) then go through each neighborhood and find which discrete segs belong to which neighborhood
                                - also must have minimum length > 5 pixels 
                                - ALSO, must have an endpoint!!!
                                
                                """
                    
                    no_loops_segs = elim_loops(discrete_segs, tree_idx=0, disc_idx=0, parent=0, be_coords=be_coords, cleaned_segs=[])
                    
                    for neighbor_num, neighborhood in enumerate(combined_b):
                        
                        ### find matching discrete segs
                        matched = []
                       
                        for seg in no_loops_segs:
                            
                            if len(neighborhood) == 1 and (seg[:, None] == neighborhood).all(-1).any() and len(seg) > 6 :
                                start_id = np.where((seg[:, None] == neighborhood).all(-1))[0][0]
                                seg_ord, empty, empty = order_coords_from_start(seg, start=seg[start_id])   ### ORDER COORDINATES SO START FROM START
                                
                                matched.append(seg_ord)
                                
                            elif len(neighborhood) > 1 and (neighborhood[:, None] == seg).all(-1).any() and len(seg) > 6 :
                                
                                start_id = np.transpose(np.where((neighborhood[:, None] == seg).all(-1)))
                                start_id = start_id[0, 1]
                                
                                seg_ord, empty, empty = order_coords_from_start(seg, start=seg[start_id])   ### ORDER COORDINATES SO START FROM START
                                
                                matched.append(seg_ord)
                                
                        """ (3) If > 4 discrete segs at a neighborhood, then go through each discrete seg and find out directional vector
                                between 2 lines in 2D (don't need 3D???), find max theta to any other segment
                                        if theta == 180 for 2 pairs of discrete segs, choose discrete seg pair that is earliest in detection (ordered coords)
                        """
                        if len(matched) >= 4:
                            id_s = 0
                            pairs = []; pairs_2 = [];
                            while len(matched) > 0:
                            #for id_s, seg in enumerate(matched):
                                seg = matched[0]
                                vector_1 = seg[0] - seg[5]
                                
                                all_angles = []
                                for id_c, check_seg in enumerate(matched):
                                    #if id_s == id_c: continue
                                    
                                    vector_2 = check_seg[0] - check_seg[5]
                                    
                                    uv_1 = vector_1 / np.linalg.norm(vector_1)
                                    uv_2 = vector_2 / np.linalg.norm(vector_2)
                                    angle = np.arccos(np.dot(uv_1, uv_2))
                                    deg = angle * (180/math.pi)
                                    print(deg)
                                    all_angles.append(deg)
                                                                   
                                ### if found pair that is > 160 degrees ==> means is a pair!!!
                                if np.nanmax(all_angles) > 150:
                                    idx_max = np.nanargmax(all_angles)
                                    pairs.append(seg); pairs_2.append(matched[idx_max])
                                    matched = np.delete(matched, [id_s, idx_max])  ### remove the matched AND current one from consideration   
                            
                                else:   ### even if didn't find any, just remove it
                                    matched = np.delete(matched, [id_s])  ### remove current one from consideration   

                        
                            """ Next see which pair comes first! If there is more than 1 pair """
                            if len(pairs) > 1:
                                ### find index that the pairs correspond to
                                id_pairs = []
                                for p_seg in pairs:
                                    for id_seg, seg in enumerate(discrete_segs):
                                        if p_seg in seg:
                                            id_pairs.append(id_seg)
                                
                                id_pairs_2 = []
                                for p_seg in pairs_2:
                                    for id_seg, seg in enumerate(discrete_segs):
                                        if p_seg in seg:
                                            id_pairs_2.append(id_seg)                                
                                
                                stack_ind = np.transpose(np.vstack([id_pairs, id_pairs_2]))
                                row, col = np.unravel_index(stack_ind.argmin(), stack_ind.shape)
                                
                                ind_to_delete = np.delete(stack_ind, row, axis=0)  ### exclude the good row, so all that's left is what we want to delete
                                ind_to_delete = np.asarray(ind_to_delete).flatten()
                                
                                """ (4) delete from discrete segs!!! """
                                discrete_segs = np.delete(discrete_segs, ind_to_delete)
                                
                                print('DELETED a t-intersection!!! ')

                                """ Plot for debug: """
                                check_debug = np.zeros(degrees.shape, dtype=np.int32)
                                for id_seg, seg in enumerate(discrete_segs):
                                    scaled = scale_coords_of_crop_to_full(np.vstack(seg), -1 * np.asarray(box_xyz), -1 * np.asarray(box_over))
                                    
                                    check_debug[scaled[:, 0], scaled[:, 1], scaled[:, 2]] = id_seg + 1
                
                                plot_save_max_project(fig_num=10, im=check_debug, max_proj_axis=-1, title='_MINUS_T_INTER', 
                                            name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator)  + '_step_' + str(step) + '_neighbor_num_' + str(neighbor_num) + '_(7)_MINUS_T_INTER.png', pause_time=0.001) 



                    """ IF is empty (no following part) """
                    if len(ordered) == 0:
                        tree.visited[node_idx] = 1;
                        print('Finished'); iterator += 1; continue
                    
                    else:
                        
                        
                        """ First drop the previous node and reorganize, cur_idx should auto update to max idx
                                only need to update parent idx to be parent of node_idx BEFORE deleting AND depth
                        """

                    
                        tmp = tree.copy()
                        
                        #tree = tmp.copy()
                        #root_neighborhood = cur_be_start
                        idx_to_del = np.where(tree.cur_idx == node_idx)[0][0]
                        parent = tree.parent[node_idx]
                        depth_tree = tree.depth[node_idx]  
                        
                        cur_idx = tree.cur_idx[node_idx]
                       
                        tree = tree.drop(index = idx_to_del)
                        
                        
                        
                        
                        """ else add to tree """
                        tree = treeify_nx(tree, discrete_segs, tree_idx=node_idx, disc_idx=0, parent=parent, be_coords=be_coords)
    
    
                        """ DEBUG: ensure it is re-inserted """
                        print(tree.cur_idx[idx_to_del])
                        
                        
                        ### set "visited" to correct value
                        for idx, node in tree.iterrows():
                              if node.visited == -1:
                                  continue
                              elif len(node.child) > 0:  ### if it has children, don't visit again!
                                  node.visited = 1
                              elif not node.visited:
                                  node.visited = np.nan    
                        
                        
                        """ set parent is visited to true """
                        tree.visited[node_idx] = 1;
                        
                        iterator += 1;
                        
                        print('Finished one iteration'); plt.close('all')
                       
                    """ Save image for animation """
                    if scale_for_animation:
                         # ### Just current tree???
                         # im = show_tree_FAST(tree, track_trees)
                                             
                         # ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
                         # for cur_tree in all_trees:
                         #       im += show_tree_FAST(cur_tree, track_trees)        

                         all_segs = show_tree_FAST(tree)
                                          
                         ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
                         for cur_tree in all_trees:
                            all_segs = np.concatenate((all_segs, show_tree_FAST(cur_tree)))
                         # im = np.zeros(np.shape(input_im))
                         # im[all_segs[:, 0], all_segs[:, 1], all_segs[:, 2]] = 1
                         
                         
                         """ scale each axis """
                         all_segs[:, 0] = all_segs[:, 0] * scale_for_animation
                         all_segs[:, 1] = all_segs[:, 1] * scale_for_animation
                         all_segs[:, 2] = all_segs[:, 2] * scale_for_animation

                         im = np.zeros(np.shape(input_im_rescaled))
                         #im = convert_matrix_to_multipage_tiff(im)   
                         im[all_segs[:, 2], all_segs[:, 0], all_segs[:, 1]] = 1
                         
                               
                         print("Saving animation")
                                           
                         
   
                         im[im > 0] = 1
                         
                         """ RESCALE IF WANT LOWER QUALITY DATA to save space!!! *** but must scale both ouptut AND input!!! """
                         # image_rescaled = resize(im, (im.shape[0] * scale_for_animation, im.shape[1] * scale_for_animation, im.shape[2] * scale_for_animation))
                         # image_rescaled[image_rescaled > 0.01] = 1   # binarize again
                         
                         imsave(s_path + filename + '_ANIMATION_crop_' + str(num_tree) + '_' + str(iterator) + '.tif', np.asarray(im * 255, dtype=np.uint8)) 
                         imsave(s_path +  filename + '_ANIMATION_input_im_' + str(num_tree) + '_' + str(iterator) + '.tif', np.asarray(input_im_rescaled, dtype=np.uint8))
                                        
    
               
             """ Add expanded tree back into all_trees """
             all_trees[num_tree] = tree
               
             num_tree += 1 
             print('Tree #: ' + str(num_tree) + " of possible: " + str(len(all_trees)))


             """ Set globally """
             matplotlib.use('Qt5Agg')
             plt.rc('xtick',labelsize=0)
             plt.rc('ytick',labelsize=0)
             #plt.rcParams['figure.dpi'] = 300
             ax_title_size = 18
             leg_size = 16
                     
                 
             """ Save max projections and pickle file """
             im = show_tree(tree, track_trees)
             plot_save_max_project(fig_num=6, im=im, max_proj_axis=-1, title='overall seg', 
                                        name=s_path + filename + '_overall_segmentation_' + str(num_tree) + '_.png', pause_time=0.001)

             """ Save max projections and pickle file """
             im[im > 0] = 1
             plot_save_max_project(fig_num=7, im=im, max_proj_axis=-1, title='overall seg', 
                                        name=s_path + filename + '_overall_segmentation_BW' + str(num_tree) + '_.png', pause_time=0.001)        

        
             
        
        
        
        
        
        
        
        
        
        print('save entire tree')
        ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
        
        empty = np.zeros(np.shape(input_im))
        im = np.zeros(np.shape(input_im))
        for cur_tree in all_trees:
            im += show_tree(cur_tree, empty)
            
            
            plot_max(show_tree(cur_tree, empty), ax=-1)
            
            
            
        color_im = np.copy(im)
        im[im > 0] = 1
        plot_save_max_project(fig_num=7, im=im, max_proj_axis=-1, title='overall seg', 
                                   name=s_path + filename + '_overall_segmentation_BW' + str(num_tree) + '_.png', pause_time=0.001)        

        print("Saving after first iteration")
        im = convert_matrix_to_multipage_tiff(im)
        imsave(s_path + filename + '_overall_output_1st_iteration.tif', np.asarray(im * 255, dtype=np.uint8))

        print("Saving after first iteration")
        color_im = convert_matrix_to_multipage_tiff(color_im)
        imsave(s_path + filename + '_overall_output_1st_iteration_COLOR.tif', np.asarray(color_im * 255, dtype=np.uint8))
           
        all_trees_copy = all_trees.copy()           
        all_starting_indices = [];
        idx = 0;
        for tree in all_trees:
        
            """ first clean up parent/child associations """
            for index, vertex in tree.iterrows():
                 cur_idx = vertex.cur_idx
                 children = np.where(tree.parent == cur_idx)
                 
                 vertex.child = children[0]
                                  
            if idx == 0:
                all_trees_appended = all_trees[0]
                all_starting_indices.append(0)
                idx += 1
                continue
                 
            
            for r_id, row in enumerate(tree.child):                
                tree.child[r_id] = np.add(tree.child[r_id], len(all_trees_appended) ).tolist() 
            tree.parent = tree.parent + len(all_trees_appended) 
            tree.cur_idx = tree.cur_idx + len(all_trees_appended) 
            
            all_trees_appended = all_trees_appended.append(tree, ignore_index=True)
            
            all_starting_indices.append(len(all_trees_appended))
            
            idx += 1       

        """ Saves tree as swc file """ 
        save_tree_to_swc(all_trees_appended, s_path, filename = filename + 'output.swc', scale_xy=0.20756792660398113, scale_z=1)

        ### also save each individual tree???
        for counter, tree in enumerate(all_trees):
            save_tree_to_swc(tree, s_path, filename = filename + '_' + str(counter) + '_output.swc', scale_xy=0.20756792660398113, scale_z=1)
        
        """ Save tree as obj file """
        save_tree_to_obj(all_trees, s_path, filename = filename + 'output.obj')

        ### also save individual trees
        for counter, tree in enumerate(all_trees):
            save_tree_to_obj([tree], s_path, filename = filename + '_' + str(counter) + '_output.obj')


        """ *** PRIOR to this final step, also make sure to combine all non-branched together to make less vertices!!!         
                /media/user/storage/Data/(1) snake seg project/Traces files/swc files                
                ***also combine all .swc into one single file?
        
        """

        
        
        
        
        
        
        
    