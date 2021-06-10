# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger
### TO DO:

    

     Things to add:
         - validation comparison at end
         - debug mode (not print outputs)
         - PARALLEL PROCESSING??? can do analysis of MANY trees at once??? Each iteration???
    
    
    
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


import tifffile as tiff

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
check_path = './(1) Checkpoint_unet_MEDIUM_filt7x7_b8_HD_INTERNODE/'; dilation = 1; deep_supervision = False; tracker = 1; HISTORICAL = 0;


check_path = './(2) Checkpoint_unet_MEDIUM_filt7x7_b8_HD_INTERNODE_SPS_optimizer/'; dilation = 1; deep_supervision = False; tracker = 1; HISTORICAL = 0;

z_size = 48



""" Oligos with paranodes """
input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/just first 2 REDO multiply scaling/just first 2 REDO multiply scaling_OUTPUT_DETECTRON/';   seed_crop_size=150; seed_z_size=50


""" Use perfect oligo seeds """
input_path = '/media/user/storage/Data/(1z) paranode_identification/Training_COMBINED_DATASET/test_image/'; seed_crop_size=150; seed_z_size=50
#input_path = '/media/user/storage/Data/(1z) paranode_identification/Training_COMBINED_DATASET/test_image/validation'; seed_crop_size=150; seed_z_size=50

input_path = '/media/user/storage/Data/(1z) paranode_identification/Training_COMBINED_DATASET/test_very_dense/'; seed_crop_size=150; seed_z_size=50


""" post-correction paranodes """
#input_path = '/media/user/storage/Data/(1z) paranode_identification/Training_COMBINED_DATASET/val_image/val_image_output_5x5_SPS_optimizer_correct_depth_NO_BATCH_NORM_62736/ERROR_CORR_(2)_checkpoints/'


### WITH ERROR_CORRECTIOn
#input_path = '/media/user/storage/Data/(1z) paranode_identification/Training_COMBINED_DATASET/test_image/test_image_output_5x5_HD_loss_check_(32)_62736/test_image_output_5x5_HD_loss_check_(32)_62736_ERROR_CORRECT_check_(2)58604/'
#/media/user/storage/Data/(1) snake seg project/Traces files MORE/DENSE_cell


#input_path = '/media/user/storage/Data/(1z) paranode_identification/Training_COMBINED_DATASET/test_image/analysis_output_5x5_HD_loss_check_(33)_49666/analysis_output_5x5_HD_loss_check_(33)_49666_ERROR_CORRECT_check_(2)58604/'




""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*input.tif*'))
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
images.sort(key = natsort_key1)
#examples = [dict(input=i, paranodes=i.replace('input.tif','input_OUTPUT_SEG.tif'), 
examples = [dict(input=i, paranodes=i.replace('_input.tif','_paranodes_from_MAT.tif'), 
                 val_mat=i.replace('_input.tif','.mat'), val_im=i.replace('_input.tif','_seeds.tif')) for i in images]

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

        """ (1) Loads data as sorted list of seeds """
        input_name = examples[i]['input']
        input_im = tiff.imread(input_name)
        width_tmp = np.shape(input_im)[1]
        height_tmp = np.shape(input_im)[2]
        depth_tmp = np.shape(input_im)[0]
             
        
        input_im =  np.moveaxis(input_im, 0, 2)

        
        filename = input_name.split('/')[-1].split('.')[0:-1]
        filename = '.'.join(filename)        
 
    
 
        """ Create save folder """
        s_path = check_path + filename + '_TEST_176000/'
        
        try:
            # Create target Directory
            os.mkdir(s_path)
            print("Directory " , s_path ,  " Created ") 
        except FileExistsError:
            print("Directory " , s_path ,  " already exists")
        
                
         
    
        # load segmentations
        seg_name = examples[i]['paranodes']
        paranodes = tiff.imread(seg_name)  
        paranodes =  np.moveaxis(paranodes, 0, 2)
        
        
        """ scale input im for animations """
        if scale_for_animation:
            
             input_im_rescaled = convert_matrix_to_multipage_tiff(input_im)   
             input_im_rescaled = resize(input_im_rescaled, (input_im_rescaled.shape[0] * scale_for_animation, input_im_rescaled.shape[1] * scale_for_animation, input_im_rescaled.shape[2]  * scale_for_animation))
             
            
        """ First split paranodes up where the first/last coords of every paranode becomes start of new tree!
        
                so each paranode detection can become root of many trees (increase probability of initiating a track)
                
                    HACK: maybe this could be improved? Somehow extend each local region??? So the skeleton starts longer? and not just 2 pixels?
        """
    
        """ only 50 """
        #dil_p = dilate_by_ball_to_binary(paranodes, radius = 2)   ### Tiger: removed this, b/c dilation just melds things together + eliminates structure of paranodes
        
        pixel_graph, degrees, coordinates = bw_skel_and_analyze(paranodes)
        bw_deg = degrees > 0
        labelled = measure.label(bw_deg)
        cc_paranodes = measure.regionprops(labelled, intensity_image=degrees)

        all_trees = []
        for it, seg in enumerate(cc_paranodes):
            coords = seg['coords']
            
            idx_start = np.where(degrees[coords[:, 0], coords[:, 1], coords[:, 2]] == 1)[0]
            start_coords = coords[idx_start]

            ### if super short, then just add directly, no sorting b/c no nearest neighbors
            if len(coords) <= 2:
                
                """ else add to tree """
                tree_df = pd.DataFrame()
                tree_df = treeify_nx(tree_df, [coords], tree_idx=0, disc_idx=0, parent=-1, start_tree=1)            

                all_trees.append(tree_df)
                             
            else:    
                ### ADD EACH START POINT AS NEW START POINT!!!
                ### "start points" are from "degrees" so essentially just "end points" ==> so we're making trees from each endpoint!
                for iter_start, starts in enumerate(start_coords):

                    
                    ### HACK: SKIP IF MORE THAN 2 end points, only select the first and last point
                    if len(start_coords) > 2 and (iter_start != 0 and iter_start != len(start_coords) - 1):
                        continue;
    
                    ordered, tree_order, discrete_segs = order_coords_from_start(coords, starts)
                
                    """ else add to tree """
                    tree_df = pd.DataFrame()
                    tree_df = treeify_nx(tree_df, discrete_segs, tree_idx=0, disc_idx=0, parent=-1, start_tree=1)            
            
                    all_trees.append(tree_df)

                        
                
        track_trees = np.zeros(np.shape(input_im))    
        num_tree = 0
        
        
        """ Concatenate into one big tree """
        # combined_trees = all_trees[0]
        # for small_t in all_trees[1:]:
            
        #     small_t.cur_idx = small_t.cur_idx + np.max(combined_trees.cur_idx) + 1
        #     for p_id, parent in enumerate(small_t.parent):
        #         if parent != -1:
        #             parent = parent + np.max(combined_trees.cur_idx) + 1
                    
        #         small_t.parent[p_id] = parent;
            
        #     combined_trees = pd.concat([combined_trees, small_t], ignore_index=True)
        
        # all_trees = []
        # all_trees.append(combined_trees)
        
        """ Create box in middle to only get seed colocalized with middle after splitting branchpoints """
        center_cube = create_cube_in_im(width=10, input_size=input_size, z_size=z_size)
        #small_center_cube = create_cube_in_im(width=8, input_size=input_size, z_size=z_size)
        center_cube_pm = create_cube_in_im(width=8, input_size=input_size * 2, z_size=z_size * 2)
        small_cube = create_cube_in_im(width=5, input_size=input_size * 2, z_size=z_size * 2) 
        
        resize_crop = 0
        for it, tree in enumerate(all_trees):
            
            
             matplotlib.use('Agg')
            
             """ Keep looping until everything has been visited """  
             iterator = 0;
             while np.asarray(tree.visited.isnull()).any():   
                 
                """ Time the run """
                # if iterator > 0:
                #     stop = time.perf_counter(); diff = stop - start_time; print(diff) 
                 
                # start_time = time.perf_counter() 
                
                 
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
                
                    
                """ Or, just go to very last position """
                node_idx = unvisited_indices[-1]
                
                
                """ Save order so can generate animation later """
                animation_order.append(node_idx)
                
                ### SKIP IF NO END_BE_COORD FROM 
                if np.isnan(tree.end_be_coord[node_idx]).any():
                     tree.visited[node_idx] = 1; iterator += 1; continue;
                    
                
                
                cur_coords, cur_be_start, cur_be_end, centroid_start, centroid_end, parent_coords = get_next_coords(tree, node_idx, num_parents=20)
                
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
                    
                
                
                ## ADD TIMER:
                # stop = time.perf_counter(); diff = stop - start_time; print(diff) 
                 
                # start_time = time.perf_counter() 
                
                
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
                
                x = int(centroid_end[0]); y = int(centroid_end[1]); z = int(centroid_end[2])
                cur_seg_im = np.zeros(np.shape(input_im))   # maybe speed up here by not creating the image every time???
                cur_seg_im[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = 1
                
        
                ### Define size of larger crop:
                pm_crop_size = crop_size * 2
                pm_z_size = z_size * 2
                output_tracker[output_tracker > 0] = 1
                output_PYTORCH, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(output_tracker, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                

                crop_seed, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(cur_seg_im, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                
                crop_seed = skeletonize_3d(crop_seed)     
                
                
                
                """ Make sure no gaps in crop_seed """
                crop_seed, output_non_bin = bridge_end_points(crop_seed, bridge_radius=2)
                #crop_seed = dilate_by_ball_to_binary(crop_seed, radius=dilation)
                #crop_seed[crop_seed > 0] = 255

                """ Get separate full crop size """
                 #parent_coords = np.vstack(parent_coords)
                if len(parent_coords) > 0:
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
                im = np.zeros(np.shape(input_im))
                im[all_segs[:, 0], all_segs[:, 1], all_segs[:, 2]] = 1
                
                
                im[im > 0] = 1
                crop_prev, box_xyz, box_over, boundaries_crop = crop_around_centroid_with_pads(im, y, x, z, pm_crop_size, pm_z_size, height_tmp, width_tmp, depth_tmp)                                                      
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
                

                """ add in crop seed and subtract later??? """
                output_PYTORCH = output_PYTORCH + crop_seed
                output_PYTORCH[output_PYTORCH > 0] = 1
               
                """ LINK EVERY END POINT TOGETHER USING line_nd """      
                output_PYTORCH = skeletonize_3d(output_PYTORCH)                    
                output_PYTORCH, output_non_bin = bridge_end_points(output_PYTORCH, bridge_radius=0)
                
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
                        iterator += 1;
                        continue                      
  
                else:
                    
                    """ Dilate to clean up a bit """
                    only_coloc = dilate_by_ball_to_binary(only_coloc, radius=2)                    


                    """ Make sure to skeletonize again """
                 
                    
                    only_coloc = skeletonize_3d(only_coloc)    
                    only_coloc[only_coloc > 0] = 1


                    # NEW: skeletonize and extract ordered graph
                    # then convert into tree
                    

                    pixel_graph, degrees, coordinates = bw_skel_and_analyze(only_coloc)

                    """ Also need to add in the starting point """
                    tmp_degrees = np.copy(degrees)
                    cur_start = np.copy(cur_be_start)
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
                    


                    """ To fix: 
                        
                        
                        DEC. 11th ==> start coord is being added at weird spot, so making small tiny segs
                        """

                    start = np.transpose(np.where(degrees == 20))[0]
                 
                    
                    """ Order the coordinates in degrees into discrete segments that can then be treeify-ed"""
                    ordered, discrete_segs, be_coords = order_skel_graph(degrees, start=start, end=[])
                    discrete_non_scaled = np.copy(discrete_segs)   # for debugging
                    ordered_non_scaled = np.copy(ordered)           # for debugging
                    ordered = scale_coords_of_crop_to_full(ordered, box_xyz, box_over)

                    all_lengths = []
                    for idx_s, seg in enumerate(discrete_segs):
                        seg = scale_coords_of_crop_to_full(np.vstack(seg), box_xyz, box_over)
                      
                        """ Ensure doesn't go out of limits"""
                        seg[np.where(seg[:, 0] >= width_tmp), 0] = width_tmp * 2 - 1
                        seg[np.where(seg[:, 1] >= height_tmp), 1] = height_tmp * 2 - 1
                        seg[np.where(seg[:, 2] >= depth_tmp), 2] = depth_tmp - 1  
                        
                        discrete_segs[idx_s] = seg
                        
                        discrete_non_scaled[idx_s]  = np.vstack(discrete_non_scaled[idx_s])   # for debugging
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
                        seg = scale_coords_of_crop_to_full(np.vstack(seg), box_xyz, box_over)
                        be_coords[idx_s] = seg


                    check_debug = np.zeros(degrees.shape, dtype=np.int32)
                    for idx_hist, row in enumerate(ordered_non_scaled):
                        check_debug[row[0], row[1], row[2]] = idx_hist + 1

                    plot_save_max_project(fig_num=10, im=check_debug, max_proj_axis=-1, title='_final_added', 
                                name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator)  + '_step_' + str(step) + '_(6)_zfinal_added.png', pause_time=0.001) 




                    """ Detect if branched, and eliminate all branches except most straight direction to go! """
                    linear_segs=[]; linear_segs.append(discrete_segs[0])
                
                    discrete_segs = linear_walk(discrete_segs, disc_idx=0, linear_segs=linear_segs)
                    
                    
                    ### For DEBUGGING:
                    non_ordered=[]; non_ordered.append(discrete_non_scaled[0])
                    non_ordered = linear_walk(discrete_non_scaled, disc_idx=0, linear_segs=non_ordered)    
                    
                    if len(non_ordered) != len(discrete_non_scaled):   ### only plot if something changed
                        check_debug = np.zeros(degrees.shape, dtype=np.int32)
                        for idx_hist, row in enumerate(np.vstack(non_ordered)):
                            check_debug[row[0], row[1], row[2]] = idx_hist + 1                    
                        
                        plot_save_max_project(fig_num=10, im=check_debug, max_proj_axis=-1, title='_final_added_LINEARIZED', 
                                    name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator)  + '_step_' + str(step) + '_(7)_zfinal_added_LINEARIZED.png', pause_time=0.001) 
           




                    """ HACK: stop if going crazy """
                    # if node_idx > 200:
                    #     tree.visited[node_idx] = 1;
                    #     print('Finished'); iterator += 1; continue
                    



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

        

        """ Set globally """
        matplotlib.use('Qt5Agg')
        plt.rc('xtick',labelsize=12)
        plt.rc('ytick',labelsize=12)
        #plt.rcParams['figure.dpi'] = 300
        ax_title_size = 18
        leg_size = 16
             
        
        """ Save dataframe as pickle """
        import pickle
        with open(s_path + 'all_trees.pkl', 'wb') as f:
            pickle.dump(all_trees, f)
        
        # Load back pickle
        # with open(s_path + 'all_trees.pkl', 'rb') as f:
        #      mynewlist = pickle.load(f)
                
        
        
        
        
        
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

        cleaned_paranodes = np.zeros(np.shape(paranodes))
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
                print('yo')
            
            
            
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
                
                    print(len(C))
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
                            print(len(C)/len(inter_1))
                            print(len(inter_1))
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
        mat_name = examples[ i]['val_mat']
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
                    print(len(np.where(val_im[expand[:, 0], expand[:, 1], expand[:, 2]])[0]))
                    
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
                    print(prop_diffs)
                    
                    
                    
            val_metrics['num_match'].append(num_match)
            val_metrics['len_diff'].append(len_diffs)     
            val_metrics['prop_diff'].append(prop_diffs)     

            if matched:
                total_len_diff = len(sheath) - total_lens
                total_prop_diff = total_len_diff/len(sheath)
    
                val_metrics['total_len_diff'].append(total_len_diff)     
                val_metrics['total_prop_diff'].append(total_prop_diff)                      
                        
        plt.close('all')
        
        arr_props = np.asarray(val_metrics['total_prop_diff'])
        plt.figure()
        plt.hist(arr_props)      ### negative values mean that segmentation longer than ground truth
        
        perc = len(np.where((arr_props > -0.4) & (arr_props < 0.6))[0])   # within +/- 50% proportion
        total_perc = perc/len(arr_props)
        print(total_perc)
        

        arr_props = np.asarray(val_metrics['total_len_diff'])
        plt.figure()
        plt.hist(arr_props)      ### negative values mean that segmentation longer than ground truth
        
        perc = len(np.where((arr_props > -50) & (arr_props < 50))[0])   # within +/- 50% proportion
        total_perc = perc/len(arr_props)
        print(total_perc)



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
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        


        
        
        
        # print('save entire tree')
        # ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
        
        # empty = np.zeros(np.shape(input_im))
        # im = np.zeros(np.shape(input_im))
        # for cur_tree in all_trees:
        #     im += show_tree(cur_tree, empty)
            
            
        # color_im = np.copy(im)
        # im[im > 0] = 1
        # plot_save_max_project(fig_num=7, im=im, max_proj_axis=-1, title='overall seg', 
        #                            name=s_path + filename + '_overall_segmentation_BW' + str(num_tree) + '_.png', pause_time=0.001)        

        # print("Saving after first iteration")
        # im = convert_matrix_to_multipage_tiff(im)
        # imsave(s_path + filename + '_overall_output_1st_iteration.tif', np.asarray(im * 255, dtype=np.uint8))

        # print("Saving after first iteration")
        # color_im = convert_matrix_to_multipage_tiff(color_im)
        # imsave(s_path + filename + '_overall_output_1st_iteration_COLOR.tif', np.asarray(color_im * 255, dtype=np.uint8))




           
        # all_trees_copy = all_trees.copy()           
        # all_starting_indices = [];
        # idx = 0;
        # for tree in all_trees:
        
        #     """ first clean up parent/child associations """
        #     for index, vertex in tree.iterrows():
        #          cur_idx = vertex.cur_idx
        #          children = np.where(tree.parent == cur_idx)
                 
        #          vertex.child = children[0]
                                  
        #     if idx == 0:
        #         all_trees_appended = all_trees[0]
        #         all_starting_indices.append(0)
        #         idx += 1
        #         continue
                 
            
        #     for r_id, row in enumerate(tree.child):                
        #         tree.child[r_id] = np.add(tree.child[r_id], len(all_trees_appended) ).tolist() 
        #     tree.parent = tree.parent + len(all_trees_appended) 
        #     tree.cur_idx = tree.cur_idx + len(all_trees_appended) 
            
        #     all_trees_appended = all_trees_appended.append(tree, ignore_index=True)
            
        #     all_starting_indices.append(len(all_trees_appended))
            
        #     idx += 1       




        """ *** PRIOR to this final step, also make sure to combine all non-branched together to make less vertices!!!         
                /media/user/storage/Data/(1) snake seg project/Traces files/swc files                
                ***also combine all .swc into one single file?
        
        """

        
        
        
        
        
        
        
    