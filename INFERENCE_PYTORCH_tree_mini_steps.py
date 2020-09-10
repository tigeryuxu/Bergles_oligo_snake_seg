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


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 



""" Define GPU to use """
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

""" Decide if use pregenerated seeds or not """
pregenerated = 0
        
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


check_path = './(59) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/';  dilation = 1; deep_supervision = False; tracker = 1;
#check_path = './(60) Checkpoint_unet_COMPLEX_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/'; dilation = 1; deep_supervision = False; tracker = 1;


#check_path = './(62) Checkpoint_unet_COMPLEX_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_NO_1st_im_5_step/';  dilation = 1; deep_supervision = False; tracker = 1;


#storage_path = '/media/user/storage/Data/(1) snake seg project/Backup checkpoints/'
#check_path = storage_path + '(65) Checkpoint_unet_LARGE_filt7x7_b4_NEW_DATA_B_NORM_crop_pad_Hd_loss_balance_FULL_DATA_REPEAT_SW_ACCID/'; dilation = 1; deep_supervision = False; tracker = 1;


#s_path = check_path + 'TEST_inference_132455_last_first_REAL/'

#s_path = check_path + 'TEST_inference_158946_shortest_first/'

#s_path = check_path + 'FULL_AUTO_TEST_inference_158946_last_first_NEURON/'

#s_path = check_path + 'FULL_AUTO_TEST_inference_158946_last_first_REAL_2_CARE_RESTORED_FULL_AUTO/'

s_path = check_path + 'TEST_inference_185437_shortest_first_NEURON/'
#s_path = check_path + 'TEST_inference_185437_last_first_NEURON/'

try:
    # Create target Directory
    os.mkdir(s_path)
    print("Directory " , s_path ,  " Created ") 
except FileExistsError:
    print("Directory " , s_path ,  " already exists")

input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation large_25px_NEW/'
#input_path = 'E:/7) Bergles lab data/Traces files/seed generation large_25px/'

#input_path = '/media/user/storage/Data/(1) snake seg project/CARE_flipped_reconstruction/to segment/'


input_path = '/media/user/storage/Data/(1) snake seg project/BigNeuron data/gold166/Training data neurons/test/'

""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*input.tif*'))
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)
examples = [dict(input=i,truth=i.replace('input.tif','truth.tif'), cell_mask=i.replace('input.tif','input_cellMASK.tif'),
                 seeds = i.replace('input.tif', 'seeds.tif')) for i in images]

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
z_size = 32


""" Change to scaling per crop??? """
original_scaling = 0.2076;
target_scale = 0.20;
scale_factor = original_scaling/target_scale;
scaled_crop_size = round(input_size/scale_factor);
scaled_crop_size = math.ceil(scaled_crop_size / 2.) * 2  ### round up to even num

scale_for_animation = 0

for i in range(len(examples)):              

        """ (1) Loads data as sorted list of seeds """
        
        sorted_list, input_im, width_tmp, height_tmp, depth_tmp, overall_coord, all_seeds, all_seeds_no_50 = load_input_as_seeds(examples, im_num=i,
                                                                                                                                 pregenerated=pregenerated, s_path=s_path,
                                                                                                                                 seed_crop_size=150, seed_z_size=80)   

        input_name = examples[i]['input']
        filename = input_name.split('/')[-1].split('.')[0:-1]
        filename = '.'.join(filename)
        
       
        """ scale input im for animations """
        if scale_for_animation:
             input_im_rescaled = convert_matrix_to_multipage_tiff(input_im)   
             
            
        """ add seeds to form roots of tree """
        """ (1) First loop through and turn each seed into segments at branch points 
            (2) Then add to list with parent/child indices
        """
    
        """ only 50 """
        if pregenerated:
            all_seeds[all_seeds !=  50] = 0;
            labelled = measure.label(all_seeds)
            cc = measure.regionprops(labelled)
            all_coords_root = []

        else:
            all_seeds[all_seeds !=  2] = 0;
            labelled = measure.label(all_seeds)
            cc = measure.regionprops(labelled)
            all_coords_root = []

        for point in cc:
            coord_point = point['coords']
            all_coords_root.append(coord_point)
        all_trees = []
        for i_r, root in enumerate(sorted_list):
            tree_df, children = get_tree_from_im_list(root, input_im, width_tmp, height_tmp, depth_tmp, all_coords_root)                                          
            
            ### HACK: continue if empty
            if len(tree_df) == 0:
                continue;
            tmp_tree_im = np.zeros(np.shape(input_im))
            #im = show_tree(tree_df, tmp_tree_im)
            #plot_max(im, ax=-1)                
                        
            ### set "visited" to correct value
            for idx, node in tree_df.iterrows():
                if not isListEmpty(node.child):
                     node.visited = 1
                else:
                    node.visited = np.nan            
            # append to all trees
            all_trees.append(tree_df)
            
            ### skip rest for debugging
            #if i_r == 0:
            #       break
            

 


        track_trees = np.zeros(np.shape(input_im))    
        num_tree = 0

        """ Create box in middle to only get seed colocalized with middle after splitting branchpoints """
        center_cube = create_cube_in_im(width=10, input_size=input_size, z_size=z_size)
        #small_center_cube = create_cube_in_im(width=8, input_size=input_size, z_size=z_size)
        
        
        center_cube_pm = create_cube_in_im(width=8, input_size=input_size * 2, z_size=z_size * 2)
    
        small_cube = create_cube_in_im(width=5, input_size=160, z_size=64) 
        
        
        matplotlib.use('Agg')
        
        
        for iterator, tree in enumerate(all_trees):
             #if iterator != 2:
             #     continue
             
            
             """ Keep looping until everything has been visited """  
             resize = 0
             while np.asarray(tree.visited.isnull()).any():   

                 
                ### convert center cube back to original size
                if resize == 1:
                   center_cube_pm = create_cube_in_im(width=8, input_size=input_size * 2, z_size=z_size * 2)
                   small_cube = create_cube_in_im(width=5, input_size=input_size * 2, z_size=z_size * 2) 
                   resize = 0

                """ Get coords at node
                        ***go to node that is SHORTEST PATH LENGTH AWAY FIRST!!!
                """                  
                unvisited_indices = np.where(tree.visited.isnull() == True)[0]
                
            
                
                """ Go to index of SHORTEST PATH FIRST """
                all_lengths = []
                for ind in unvisited_indices:                 
                    parent_coords = get_parent_nodes(tree, ind, num_parents=100, parent_coords = [])
                
                    if len(parent_coords) > 0:
                        parent_coords = np.vstack(parent_coords)
                
                    all_lengths.append(len(parent_coords))                 
                node_idx = unvisited_indices[np.argmin(all_lengths)]
                
                
                
                """ Or, just go to very last position """
                #node_idx = unvisited_indices[-1]
                
                
                
                ### SKIP IF NO END_BE_COORD FROM 
                if np.isnan(tree.end_be_coord[node_idx]).any():
                     tree.visited[node_idx] = 1; iterator += 1; continue;
                    
                
                cur_coords, cur_be_start, cur_be_end, centroid_start, centroid_end, parent_coords = get_next_coords(tree, node_idx, num_parents=20)
                           
                """ Link any missing coords due to be subtractions """
                if len(cur_coords) > 3:
                    # cur_coords_connect = connect_nearby_px(cur_coords)
                    # ### move the starting index to the beginning so can sort:
                    # match_idx = np.where((cur_coords_connect == cur_coords[0]).all(axis=1))[0][0]
                    # cur_coords_connect = np.delete(cur_coords_connect, match_idx, axis=0)
                    # cur_coords_connect = np.vstack([cur_coords[0], cur_coords_connect]) 
                    # cur_coords = cur_coords_connect
                    a = 1
 
                if len(parent_coords) > 0:
                    parent_coords = connect_nearby_px(parent_coords)
                
                
                

                
                                

                # cur_seg_im = np.zeros(np.shape(input_im))   # maybe speed up here by not creating the image every time???
                # cur_seg_im[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = 1
                # cur_seg_im[parent_coords[:, 0], parent_coords[:, 1], parent_coords[:, 2]] = 1
                
            
                # pm_crop_size = crop_size * 2
                # pm_z_size = z_size * 2
                # crop_seed, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max, boundaries_crop = crop_around_centroid_with_pads(cur_seg_im, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
                # crop_seed = skeletonize_3d(crop_seed) 
                # crop_seed, output_non_bin = bridge_end_points(crop_seed, bridge_radius=2)
                

                
                """ Order coords """
                ### SKIP IF TOO SHORT for mini-steps
                if len(cur_coords) == 1:
                     tree.visited[node_idx] = 1; iterator += 1; continue;
                else:
                    cur_coords = order_coords(cur_coords)   ### ***order the points into line coordinates

                """ Split into mini-steps """
                ### Step size:
                step_size = 5; 
                step_size_first = step_size          
                if len(cur_coords) <= step_size:  ### KEEP GOING IF ONLY SMALL SEGMENT
                      step_size_first = len(cur_coords) - 1
             
                
                output_tracker = np.zeros(np.shape(input_im))
                for step in range(step_size_first, len(cur_coords), step_size):
                    
                        
                      """ DONT DO THE LAST STEP """
                      # if step_size_first > 0 and len(cur_coords) - step < step_size:
                      #     continue;
                         
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
                      output_PYTORCH = UNet_inference_PYTORCH(unet,np.asarray(crop, np.float32), crop_seed, mean_arr, std_arr, device=device, deep_supervision=deep_supervision)
            
                      """ Since it's centered around crop, ensure doesn't go overboard """
                      output_PYTORCH[boundaries_crop == 0] = 0
                
                      """ SAVE max projections"""
                      plot_save_max_project(fig_num=5, im=crop_seed, max_proj_axis=-1, title='crop seed dilated', 
                                            name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(2)_seed.png', pause_time=0.001)
                      plot_save_max_project(fig_num=2, im=output_PYTORCH, max_proj_axis=-1, title='segmentation', 
                                            name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(3)_segmentation.png', pause_time=0.001)
                      plot_save_max_project(fig_num=3, im=crop, max_proj_axis=-1, title='input', 
                                            name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(1)_input_im.png', pause_time=0.001)


                      ### HACKY - figure out how to get sizes to be better
                      prev_seg = output_tracker[box_xyz[0]:box_xyz[1], box_xyz[2]:box_xyz[3], box_xyz[4]:box_xyz[5]] 
                      output_tracker[box_xyz[0]:box_xyz[1], box_xyz[2]:box_xyz[3], box_xyz[4]:box_xyz[5]] = prev_seg + output_PYTORCH[box_over[0]:prev_seg.shape[0] + box_over[0], box_over[2]:prev_seg.shape[1] + box_over[2], box_over[4]:prev_seg.shape[2] + box_over[4]]
            

 
                # if iterator == 5:    ### ALSO IN THE 50s
                #         zzz


                                
                """ ALSO HAVE TO RESET CUR_BE_END to align with current location of mini-step and NOT actual end of segment!!!
                """
                #cur_be_end = np.vstack(expand_coord_to_neighborhood([cur_coords[step]], lower=1, upper=2))
                cur_coords, cur_be_start, cur_be_end, centroid_start, centroid_end, parent_coords = get_next_coords(tree, node_idx, num_parents=50)

                """ Link any missing coords due to be subtractions """
                if len(cur_coords) > 3:
                    cur_coords = connect_nearby_px(cur_coords)
                

                 
                x = int(centroid_end[0]); y = int(centroid_end[1]); z = int(centroid_end[2])
                cur_seg_im = np.zeros(np.shape(input_im))   # maybe speed up here by not creating the image every time???
                cur_seg_im[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = 1
 
        

                    
                ### Define size of larger crop:
                pm_crop_size = crop_size * 2
                pm_z_size = z_size * 2
                output_tracker[output_tracker > 0] = 1
                output_PYTORCH, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(output_tracker, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
                
                
                ### if not big enough, recrop with larger
                
                mult = 3
                while np.any(cur_be_start[:, 0] - box_xyz[0] < 0) or np.any(cur_be_start[:, 1] - box_xyz[2] < 0) or np.any(cur_be_start[:, 2] - box_xyz[4] < 0) or  np.any(box_xyz[1] - cur_be_start[:, 0] < 0) or np.any(box_xyz[3] - cur_be_start[:, 1] < 0) or np.any(box_xyz[5] - cur_be_start[:, 2] < 0):
                    pm_crop_size = crop_size * mult
                    pm_z_size = z_size * 3
                
                    output_tracker[output_tracker > 0] = 1
                    output_PYTORCH,  box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(output_tracker, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
                    resize = 1
                    
                    center_cube_pm = create_cube_in_im(width=8, input_size=input_size * mult, z_size=z_size * 3)
                    
                    small_cube = create_cube_in_im(width=5, input_size=input_size * mult, z_size=z_size * 3)
                    
                    mult += 1
                    
                    #print(box_x_min)
                
                

                crop_seed, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(cur_seg_im, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
                crop_seed = skeletonize_3d(crop_seed)     
                
                """ Make sure no gaps in crop_seed """
                crop_seed, output_non_bin = bridge_end_points(crop_seed, bridge_radius=2)
                #crop_seed = dilate_by_ball_to_binary(crop_seed, radius=dilation)
                #crop_seed[crop_seed > 0] = 255




                """ Get separate full crop size """
                 #parent_coords = np.vstack(parent_coords)
                if len(parent_coords)> 0:
                    parent_coords = connect_nearby_px(parent_coords)
                    
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
  
                """ REMOVE EDGE """
                # dist_xy = 0; dist_z = 0
                # edge = np.zeros(np.shape(output_PYTORCH)).astype(np.int64)
                # edge[dist_xy:pm_crop_size * 2-dist_xy, dist_xy:pm_crop_size * 2-dist_xy, dist_z:pm_z_size-dist_z] = 1
                # edge = np.where((edge==0)|(edge==1), edge^1, edge)
                # output_PYTORCH[edge == 1] = 0
  
                """ ***FIND anything that has previously been identified
                    ***EXCLUDING CURRENT CROP_SEED
                """
                im = show_tree(tree, track_trees)
                                  
                ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
                for cur_tree in all_trees:
                    im += show_tree(cur_tree, track_trees)
                
                im[im > 0] = 1
                crop_prev, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(im, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                                                      
                crop_prev = skeletonize_3d(crop_prev)
                crop_prev[crop_prev > 0] = 1
                                  
                
                
                """ WRONG??? 
                
                
                    ?????????????
                    
                    (1) need to first identify if there is any need for mini-step (i.e. are there even new segments to worry about???)
                        ***problem:
                                sub-seed is not working? b/c the crop_seed is added back in without dilation???
                    
                    (2) then, if yes, do mini-step
                    
                
                """
                
                ### EXCLUDE current crop seed
                im_sub = subtract_im_no_sub_zero(crop_prev, crop_seed)
                
                # with dilation
                #im_sub = im_sub + crop_seed
                
                im_dil = dilate_by_ball_to_binary(im_sub, radius=2)
                
                ### but add back in current crop seed (so now without dilation)
                im_dil = im_dil + crop_seed
                im_dil[im_dil > 0] = 1
                
  
                """ delete all small objects """
                labelled = measure.label(output_PYTORCH)
                cc = measure.regionprops(labelled); 
                cleaned = np.zeros(np.shape(output_PYTORCH))
                for seg in cc:
                       coord = seg['coords']; 
                       if len(coord) > 10:
                           cleaned[coord[:, 0], coord[:, 1], coord[:, 2]] = 1
  
                output_PYTORCH = cleaned
  
                
                """ add in crop seed and subtract later??? """
                output_PYTORCH = output_PYTORCH + crop_seed
                output_PYTORCH[output_PYTORCH > 0] = 1
               
                
                
                """ LINK EVERY END POINT TOGETHER USING line_nd """      
                output_PYTORCH = skeletonize_3d(output_PYTORCH)                    
                output_PYTORCH, output_non_bin = bridge_end_points(output_PYTORCH, bridge_radius=2)
                
                plot_save_max_project(fig_num=3, im=output_non_bin, max_proj_axis=-1, title='output_be', 
                                      name=s_path + filename + '_Crop_' + str(num_tree) + '_' + str(iterator) +  '_step_' + str(step) +'_(4)_output_be.png', pause_time=0.001)                                              
                    
                
                
                """ Subtract out previous segmentations, but do so smartly:
                    
                        (1) subtract out old seed and see what's left
                        (2) make into connected components
                        (3) then, add in prev_seg dilated ==> see what areas still (new areas) == 2 (i.e. overlapped with previous output still)
                        (4) if area has == 2, then keep it
                            ***this allows propagation of segmentation to OVERLAP IF there is new segments identified WITHIN the current segmentation
                    
                    """
                sub_seed = subtract_im_no_sub_zero(output_PYTORCH, crop_seed_full)
                #crop_prev[sub_seed == 0] = 0
                added = np.copy(sub_seed)
                added[im_dil > 0] = 2
                added[added == 2] = 0
                
                added = sub_seed + added
                bw_added = np.copy(added)
                bw_added[bw_added > 0] = 1
                
                labelled = measure.label(bw_added)
                cc = measure.regionprops(labelled, intensity_image=added); 
                cleaned = np.zeros(np.shape(output_PYTORCH))
                for seg in cc:
                       coord = seg['coords']
                       max_intensity = seg['max_intensity']
                       if max_intensity == 2:
                           cleaned[coord[:, 0], coord[:, 1], coord[:, 2]] = 1
  
                output_PYTORCH = cleaned                
                output_PYTORCH = output_PYTORCH + crop_seed ### add it back in
                    
                    
                
                """ Keep only what is colocalized with the center """
                # (1) use old start_coords to find only nearby segments           
                # ***or just use center cube
                coloc_with_center = output_PYTORCH + center_cube_pm
                only_coloc = find_overlap_by_max_intensity(bw=output_PYTORCH, intensity_map=coloc_with_center) 
                
                
                """ moved here: subtract out past identified regions LAST to not prevent propagation """
                only_coloc[only_coloc > 0] = 1
                sub_seed = subtract_im_no_sub_zero(only_coloc, crop_seed)
                sub_seed = subtract_im_no_sub_zero(sub_seed, im_dil)
                
                
                
                """ skip if everything was subtracted out last time: """
                if np.count_nonzero(sub_seed) < 8:
                        tree.visited[node_idx] = 1; print('Finished')                     
                        plot_save_max_project(fig_num=10, im=np.zeros(np.shape(only_coloc)), max_proj_axis=-1, title='_final_added', 
                                name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) +  '_step_' + str(step) + '_(6)_zfinal_added.png', pause_time=0.001) 
                        iterator += 1
                        continue                      
  
                else:
  
                    # (2) skeletonize the output to create "all_neighborhoods" and "all_hood_first_last"        
                    
                    
                    
                    """ Make sure to skeletonize again """
                    only_coloc = skeletonize_3d(only_coloc)    
                    only_coloc[only_coloc > 0] = 1


                    """ Link to center like adding cur_start_be """
                    # tmp_coloc = np.copy(only_coloc)
                    # tmp_coloc = tmp_coloc + small_cube
                    
                    # loc_start = np.transpose(np.where(tmp_coloc > 1))
                    
                    
                    # """ If not working (not matching a cur_start), then force the match by subtracting out the center
                    #         and then linking all end points together to the start point
                    # """
                 
                    ### ONLY LINK THE SHORTEST
                    # all_dist = []
                    # center = [pm_crop_size - 1, pm_crop_size - 1, int(pm_z_size/2 - 1)]
                    # for coord in loc_start:
                        
                    #     #print(np.linalg.norm(center - coord))
                    #     if np.linalg.norm(center - coord) <= 10:
                    #         all_dist.append(np.linalg.norm(center - coord))
                    # min_id = np.argmin(all_dist)
                    # point = loc_start[min_id]
 
                    # only_coloc[point[0], point[1], point[2]] = 4
                    
                    

                    """ Link to center to ensure  
                    
                            ***but only need to do this if <= 2 branch/endpoints?
                    
                    """
                    degrees, coordinates = bw_skel_and_analyze(only_coloc)
                    degrees[degrees == 2] = 0
                    degrees[degrees > 0] = 1
                    
                    labelled = measure.label(degrees)
                    cc_num_be = measure.regionprops(labelled)
                    
                    #if len(cc) > 2:
                    #    degrees, coordinates = bw_skel_and_analyze(only_coloc)
                        
                        
                    #else:
                    only_coloc[small_cube == 1] = 0  ### delete out small cube
                    
                    center = [pm_crop_size - 1, pm_crop_size - 1, int(pm_z_size/2 - 1)]

                    bw_added[bw_added > 0] = 1
                    
                    labelled = measure.label(only_coloc)
                    cc = measure.regionprops(labelled)
                    
                    for seg in cc:
                        blank = np.zeros(np.shape(only_coloc))
                        coords = seg['coords']
                        blank[coords[:, 0], coords[:, 1], coords[:, 2]] = 1
                        
                        degrees, coordinates = bw_skel_and_analyze(blank)
                        coord_end = np.transpose(np.vstack(np.where(degrees == 1)))
                        
                        ### ONLY LINK THE SHORTEST
                        all_dist = [];
                        all_linkers = [];
                        for coord in coord_end:
                            
                            #print(np.linalg.norm(center - coord))
                            if np.linalg.norm(center - coord) <= 8:
                                all_dist.append(np.linalg.norm(center - coord))
                                
                                line_coords = line_nd(center, coord, endpoint=False)
                                line_coords = np.transpose(line_coords)      
                                
                                all_linkers.append(line_coords)
                        
                        
                        """ if could not find an end point, then just use a nearby body coord """
                        if len(all_dist) == 0:
                          
                            all_dist = [];
                            all_linkers = [];
                            for coord in coordinates:
                                
                                #print(np.linalg.norm(center - coord))
                                if np.linalg.norm(center - coord) <= 8:
                                    all_dist.append(np.linalg.norm(center - coord))
                                    
                                    line_coords = line_nd(center, coord, endpoint=False)
                                    line_coords = np.transpose(line_coords)      
                                    
                                    all_linkers.append(line_coords)   

                        if len(all_dist) > 0:
                            min_id = np.argmin(all_dist)
                            line_coords = all_linkers[min_id]                                  
                        
                            only_coloc[line_coords[:, 0], line_coords[:, 1], line_coords[:, 2]] = 1
                            only_coloc[center[0], center[1], center[2]] = 1
                    
                    
                    """ Make sure to skeletonize again """
                    only_coloc = skeletonize_3d(only_coloc)
                    
                    
                    only_coloc, output_non_bin = bridge_end_points(only_coloc, bridge_radius=4)
                    
                    
                    """ Make into degrees """
                    degrees, coordinates = bw_skel_and_analyze(only_coloc)
                    

                  
 
                    """ Don't allow things in the past to be counted as new end points 
                    
                    
                            ***can't do this b/c have new ones coming out of these
                            
                            
                            TIGER FIX:
                                need to sort these new branchpoints and reorder the old ones, or do at the end???
                            
                    """
                    #im_dil[degrees == 0] = 0
                    #degrees[im_dil > 0] = 2
    
    
                    """ insert the current be neighborhood at center and set to correct index in "degrees"
                            or add in the endpoint like I just said, but subtract it out to keep the crop_seed separate???                              
                    """
                    # cur_end = np.copy(cur_be_end)
                    # cur_end = scale_coords_of_crop_to_full(cur_end, -1 * np.asarray(box_xyz)[0], -box_y_min, -box_z_min)
                    
                    # ### check limits to ensure doesnt go out of frame
                    # cur_end = check_limits([cur_end], pm_crop_size * 2, pm_crop_size * 2, pm_z_size)[0]
                    
                    # ### HACK: fix how so end points cant leave frame
                    # """ MIGHT GET TOO LARGE b/c of building up previous end points, so need to ensure crop """
                    # cur_end[np.where(cur_end[:, 0] >= pm_crop_size * 2), 0] = pm_crop_size * 2 - 1
                    # cur_end[np.where(cur_end[:, 1] >= pm_crop_size * 2), 1] = pm_crop_size * 2 - 1
                    # cur_end[np.where(cur_end[:, 2] >= pm_z_size), 2] = pm_z_size - 1
                    
                    # ### Then set degrees
                    #degrees[cur_end[:, 0], cur_end[:, 1], cur_end[:, 2]] = 4
                    
                    ### CATCH ERROR:
                    if degrees[pm_crop_size - 1, pm_crop_size - 1, int(pm_z_size/2 - 1)] == 0  and len(cc_num_be) <= 2:
                        #zzz
                        print('bad')                        
                        
                    ### CHANGED ABOVE TO:
                    degrees[pm_crop_size - 1, pm_crop_size - 1, int(pm_z_size/2 - 1)] = 7
                    

                    """ REMOVE EDGE end points b/c crop is bigger than possible to ever reach edges """
                    dist_xy = 2; dist_z = 1
                    edge = np.zeros(np.shape(output_PYTORCH)).astype(np.int64)
                    edge[dist_xy:pm_crop_size * 2-dist_xy, dist_xy:pm_crop_size * 2-dist_xy, dist_z:pm_z_size-dist_z] = 1
                    edge = np.where((edge==0)|(edge==1), edge^1, edge)
                    
                    edge[degrees == 2] = 0
                    degrees[edge > 0] = 0                    

                    

                    """ Also need to add in the starting point """
                    tmp_degrees = np.copy(degrees)
                    cur_start = np.copy(cur_be_start)
                    cur_start = scale_coords_of_crop_to_full(cur_start, -1 * np.asarray(box_xyz), -1 * np.asarray(box_over))
                    
                    ### check limits to ensure doesnt go out of frame
                    cur_start = check_limits([cur_start], pm_crop_size * 2, pm_crop_size * 2, pm_z_size)[0]
                    
                    ### HACK: fix how so end points cant leave frame
                    """ MIGHT GET TOO LARGE b/c of building up previous end points, so need to ensure crop """
                    cur_start[np.where(cur_start[:, 0] >= pm_crop_size * 2), 0] = pm_crop_size * 2 - 1
                    cur_start[np.where(cur_start[:, 1] >= pm_crop_size * 2), 1] = pm_crop_size * 2 - 1
                    cur_start[np.where(cur_start[:, 2] >= pm_z_size), 2] = pm_z_size - 1
                    
                    ### Then set degrees
                    tmp_degrees[cur_start[:, 0], cur_start[:, 1], cur_start[:, 2]] = 20

                    ### only choose single point
                    bw_deg = skeletonize_3d(tmp_degrees)
                    

                    
                    tmp_degrees[degrees == 0] = 0

                    loc_start = np.transpose(np.where(tmp_degrees == 20))
                    
                    
                    """ If not working (not matching a cur_start), then force the match by subtracting out the center
                            and then linking all end points together to the start point
                    """
                    if len(loc_start) == 0:
                         mid = cur_start[int(len(cur_start)/2) - 1]
                        
                         mid_hood = expand_coord_to_neighborhood([mid], lower=3, upper=3 + 1)
                         mid_hood = np.vstack(mid_hood)

                         """ MIGHT GET TOO LARGE b/c of building up previous end points, so need to ensure crop """
                         mid_hood[np.where(mid_hood[:, 0] >= pm_crop_size * 2), 0] = pm_crop_size * 2 - 1
                         mid_hood[np.where(mid_hood[:, 1] >= pm_crop_size * 2), 1] = pm_crop_size * 2 - 1
                         mid_hood[np.where(mid_hood[:, 2] >= pm_z_size), 2] = pm_z_size - 1                        


                         tmp_degrees[mid_hood[:, 0], mid_hood[:, 1], mid_hood[:, 2]] = 0
                         degrees[mid_hood[:, 0], mid_hood[:, 1], mid_hood[:, 2]] = 0
                    
                         tmp_degrees, coordinates = bw_skel_and_analyze(tmp_degrees)
                         coord_end = np.transpose(np.vstack(np.where(tmp_degrees == 1)))
                        
                         for coord in coord_end:
                            
                            print(np.linalg.norm(center - coord))
                            if np.linalg.norm(mid - coord) <= 10:
                                line_coords = line_nd(mid, coord, endpoint=False)
                                line_coords = np.transpose(line_coords)      
                                
                                degrees[line_coords[:, 0], line_coords[:, 1], line_coords[:, 2]] = 2
                                #degrees[center[0], center[1], center[2]] = 2               
                                
                                print('loop')
                         
                            
                         degrees[mid[0], mid[1], mid[2]] = 8
                    else:
                    
                        ### also set everything in neighborhood to be NOT end point
                        for point in cur_start:
                            if degrees[point[0], point[1], point[2]] == 1 or degrees[point[0], point[1], point[2]] == 3:
                                degrees[point[0], point[1], point[2]] = 2
                                print('replace')
                                
                    
                        degrees[loc_start[0][0], loc_start[0][1], loc_start[0][2]] = 20
                    




                    """ Skeletonize this to make smoother for later and not cutoff by this middle part """
                    #bw_deg = skeletonize_3d(degrees)
                    #degrees[bw_deg == 0] = 0
                    

                    
                    
                
                    
                    ###remove all the others that match this first one???
                    plot_save_max_project(fig_num=9, im=degrees, max_proj_axis=-1, title='segmentation_deleted', 
                                name=s_path + filename + '_Crop_' + str(num_tree) + '_' + str(iterator)  + '_step_' + str(step) + '_(5)_segmentation_deleted.png', pause_time=0.001) 
                    
                    
                   
                    all_neighborhoods, all_hood_first_last, root_neighborhood = get_neighborhoods(degrees, coord_root=0, scale=1, box_xyz=box_xyz, box_over=box_over, order=1)
  
    
  
                    """ Find which neighborhood_be matches with cur_be_end to set as new root
                            then delete that neighborhood from the all_neighborhoods list
                                ***pick the neighborhood that matches THE MOST
                            
                    """
                    ### set root_neighborhood so exclude from all_neighborhoods list
                    root_neighborhood = cur_be_start
                    all_lens = []
                    for neighbor_be in all_neighborhoods:
                        aset = set([tuple(x) for x in root_neighborhood])
                        bset = set([tuple(x) for x in neighbor_be])
                        intersect = np.array([x for x in aset & bset])
                        
                        print('match')
                        all_lens.append(len(intersect))
                
                
                    ### CATCH ERROR
                    if np.max(all_lens) == 0:
                        zzz
                        print('not matched')
                
                    root_neighborhood = all_neighborhoods[np.argmax(all_lens)]
                    all_neighborhoods[np.argmax(all_lens)] = []
                    



                    
                    """ For debugging """
                    check_debug = np.zeros(np.shape(output_PYTORCH))
                    idx = 0; 
                    for neighbor_be in all_neighborhoods:
                       if len(neighbor_be) > 0:
                           if (root_neighborhood[:, None] == neighbor_be).all(-1).any(): 
                               print('match')
                               cur = np.copy(neighbor_be)
                               cur = scale_coords_of_crop_to_full(cur, -1 * np.asarray(box_xyz), box_over=np.zeros(np.shape(box_over)))                           
                               ### check limits to ensure doesnt go out of frame
                               cur = check_limits([cur], pm_crop_size * 2, pm_crop_size * 2, pm_z_size)[0]
                               check_debug[cur[:, 0], cur[:, 1], cur[:, 2]] = 10
                           else:
                               cur = np.copy(neighbor_be)
                               cur = scale_coords_of_crop_to_full(cur, -1 * np.asarray(box_xyz), box_over=np.zeros(np.shape(box_over)))
                               
                               ### check limits to ensure doesnt go out of frame
                               cur = check_limits([cur], pm_crop_size * 2, pm_crop_size * 2, pm_z_size)[0]
                               
                               check_debug[cur[:, 0], cur[:, 1], cur[:, 2]] = idx + 1
                     
                       idx += 1
                       
                    # delete cur_segs *** NOT NEEDED
                    idx = 0
                    for cur_seg in all_hood_first_last:
                          #if (cur_seg[:, None] == tree.coords[node_idx]).all(-1).any():
                          #      all_hood_first_last[idx] = []
                          #else:
                             cur = np.copy(cur_seg)
                             cur = scale_coords_of_crop_to_full(cur, -1 * np.asarray(box_xyz), box_over=np.zeros(np.shape(box_over)))
                             
                             ### check limits to ensure doesnt go out of frame
                             cur = check_limits([cur], pm_crop_size * 2, pm_crop_size * 2, pm_z_size)[0]
                             check_debug[cur[:, 0], cur[:, 1], cur[:, 2]] = idx + 1         
                          
                             idx += 1
                   
                    plot_save_max_project(fig_num=10, im=check_debug, max_proj_axis=-1, title='_final_added', 
                                name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator)  + '_step_' + str(step) + '_(6)_zfinal_added.png', pause_time=0.001) 

                    
                    """ IF is empty (no following part) """
                    if len(all_neighborhoods) == 0:
                        tree.visited[node_idx] = 1;
                        print('Finished'); iterator += 1; continue
                    
                    else:
                        
                        
                        """ First drop the previous node and reorganize, cur_idx should auto update to max idx
                                only need to update parent idx to be parent of node_idx BEFORE deleting AND depth
                        """

                        
                        #root_neighborhood = cur_be_start
                        idx_to_del = np.where(tree.cur_idx == node_idx)[0][0]
                        parent = tree.parent[node_idx]
                        depth_tree = tree.depth[node_idx]  
                        
                        cur_idx = tree.cur_idx[node_idx]
                       
                        tree = tree.drop(index = idx_to_del)
                        
                        
                        """ else add to tree """
                        # cur_idx = tree.cur_idx[node_idx]
                        # depth_tree = tree.depth[node_idx] + 1
                        
                        
                        tree, cur_childs = treeify(tree, depth_tree, root_neighborhood, all_neighborhoods, all_hood_first_last, cur_idx = cur_idx, parent= parent,
                                              start=1, width_tmp=width_tmp, height_tmp=height_tmp, depth_tmp=depth_tmp)
                        
                        #tree.child[cur_idx][0] = list(np.concatenate((tree.child[cur_idx][0], cur_childs), axis=-1))
                        
                        ### set "visited" to correct value
                        for idx, node in tree.iterrows():
                              if node.visited == -1:
                                  continue
                              elif not isListEmpty(node.child):
                                  node.visited = 1
                              elif not node.visited:
                                  node.visited = np.nan    
                        
                        
                        """ set parent is visited to true """
                        tree.visited[node_idx] = 1;
                        
                        print('Finished one iteration'); plt.close('all')
                        iterator += 1
                        
                        
                        """ Save image for animation """
                        if scale_for_animation:
                             ### Just current tree???
                             im = show_tree(tree, track_trees)
                                                 
                             ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
                             for cur_tree in all_trees:
                                   im += show_tree(cur_tree, track_trees)        
                                   
                             print("Saving animation")
                             im = convert_matrix_to_multipage_tiff(im)                     
                             
   
                             im[im > 0] = 1
                             image_rescaled = rescale(im, scale_for_animation)        
                             image_rescaled[image_rescaled > 0.01] = 1   # binarize again
                             
                             imsave(s_path + filename + '_ANIMATION_crop_' + str(num_tree) + '_' + str(iterator) + '.tif', np.asarray(image_rescaled * 255, dtype=np.uint8)) 
                             imsave(s_path +  filename + '_ANIMATION_input_im_' + str(num_tree) + '_' + str(iterator) + '.tif', np.asarray(input_im_rescaled, dtype=np.uint8))
                                            
    
               
             """ Add expanded tree back into all_trees """
             all_trees[num_tree] = tree
               
             num_tree += 1 
             print('Tree #: ' + str(num_tree) + " of possible: " + str(len(all_trees)))
             
                 
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
        
        im = np.zeros(np.shape(input_im))
        for cur_tree in all_trees:
            im += show_tree(cur_tree, track_trees)
            
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
                 
            tree.child = tree.child + len(all_trees_appended) 
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

        
        
        
        
        
        
        
    