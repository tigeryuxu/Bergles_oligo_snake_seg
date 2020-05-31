# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger
"""
import matplotlib
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 
#matplotlib.use('Agg')
matplotlib.use('Qt5Agg')


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
from losses_pytorch.boundary_loss import DC_and_HDBinary_loss, BDLoss, HDDTBinaryLoss


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 

from matlab_crop_function import *
from off_shoot_functions import *
from tree_functions import *
from skimage.morphology import skeletonize_3d, skeletonize


""" Define GPU to use """
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

""" Decide if use pregenerated seeds or not """
pregenerated = 1
        
"""  Network Begins: """
check_path ='./(9) Checkpoint_AdamW_batch_norm/'; dilation = 1
check_path ='./(15) Checkpoint_AdamW_batch_norm_SPATIALW/'; dilation = 1
#check_path = './(12) Checkpoint_AdamW_batch_norm_CYCLIC/'; dilation = 1
check_path = './(21) Checkpoint_AdamW_batch_norm_3x_branched/'; dilation = 1

s_path = check_path + 'TEST_inference/'
try:
    # Create target Directory
    os.mkdir(s_path)
    print("Directory " , s_path ,  " Created ") 
except FileExistsError:
    print("Directory " , s_path ,  " already exists")

input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation/'
input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation large/'

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

print('restoring weights')
check = torch.load(check_path + checkpoint)
unet = check['model_type']
unet.load_state_dict(check['model_state_dict']) 
mean_arr = check['mean_arr']
std_arr = check['std_arr']


""" Set to eval mode for batch norm """
unet.eval()
#unet.training # check if mode set correctly
unet.to(device)

input_size = 80
depth = 16

crop_size = int(input_size/2)
z_size = depth

for i in range(len(examples)):              
        """ (1) Loads data as sorted list of seeds """
        sorted_list, input_im, width_tmp, height_tmp, depth_tmp, overall_coord = load_input_as_seeds(examples, im_num=i, pregenerated=pregenerated, s_path=s_path)   

    
        """ add seeds to form roots of tree """
        """ (1) First loop through and turn each seed into segments at branch points 
            (2) Then add to list with parent/child indices
        """
        all_trees = []
        for root in sorted_list:
            tree_df, children = get_tree_from_im_list(root, input_im, width_tmp, height_tmp, depth_tmp)                              
            tmp_tree_im = np.zeros(np.shape(input_im))
            im = show_tree(tree_df, tmp_tree_im)
            plot_max(im, ax=-1)
                        
            ### set "visited" to correct value
            for idx, node in tree_df.iterrows():
                if not isListEmpty(node.child):
                    node.visited = 1
                else:
                    node.visited = np.nan            
            # append to all trees
            all_trees.append(tree_df)
            
            if len(all_trees) == 1:
                break

        matplotlib.use('Agg')
 
        track_trees = np.zeros(np.shape(input_im))    
        num_tree = 0
        for tree in all_trees:
             """ Create box in middle to only get seed colocalized with middle after splitting branchpoints """
             ball_in_middle = np.zeros([input_size,input_size, z_size])
             ball_in_middle[int(input_size/2), int(input_size/2), int(z_size/2)] = 1
             ball_in_middle_dil = dilate_by_cube_to_binary(ball_in_middle, width=10)
             center_cube = ball_in_middle_dil


             ball_in_middle = np.zeros([input_size,input_size, z_size])
             ball_in_middle[int(input_size/2), int(input_size/2), int(z_size/2)] = 1
             ball_in_middle_dil = dilate_by_cube_to_binary(ball_in_middle, width=4)
             small_center_cube = ball_in_middle_dil

                     
             """ Keep looping until everything has been visited """  
             iterator = 0    
             while np.asarray(tree.visited.isnull()).any():   
                 
                  unvisited_indices = np.where(tree.visited.isnull() == True)[0]
                  first_ind = unvisited_indices[-1]

                  """ GET ALL LAST 2 or 3 coords from parents as well """
                  parent_coords = get_parent_nodes(tree, start_ind=first_ind, num_parents=4, parent_coords=[])
                  
                  if len(parent_coords) > 0:
                      parent_coords = np.vstack(parent_coords)
                  
                  """ Get center of crop """
                  cur_coords = []

                  ### Get start of crop
                  cur_be_start = tree.start_be_coord[first_ind]
                  centroid = cur_be_start[math.floor(len(cur_be_start)/2)]
                  cur_coords.append(centroid)
 
                  ### Get middle of crop """
                  coords = tree.coords[first_ind]
                  cur_coords.append(coords)
                  
                  ### Get end of crop if it exists
                  if not np.isnan(tree.end_be_coord[first_ind]).any():   # if there's no end index for some reason, use starting one???
                      """ OR ==> should use parent??? """             
                      cur_be_end = tree.end_be_coord[first_ind]
                      centroid = cur_be_end[math.floor(len(cur_be_end)/2)]
                      cur_coords.append(centroid)                      
                  else:
                      ### otherwise, just leave ONLY the start index, and nothing else
                      cur_coords = centroid
                      cur_be_end = cur_be_start
                
                  x = int(centroid[0]); y = int(centroid[1]); z = int(centroid[2])
                  cur_coords = np.vstack(cur_coords)
                  
                  if np.shape(cur_coords)[1] == 1:
                      cur_coords = np.transpose(cur_coords)

                  cur_seg_im = np.zeros(np.shape(input_im))   # maybe speed up here by not creating the image every time???
                  cur_seg_im[cur_coords[:, 0], cur_coords[:, 1], cur_coords[:, 2]] = 1
                  
                  # add the centroid as well
                  cur_seg_im[x, y, z] = 1
                  
                  # add the parent
                  if len(parent_coords) > 0:
                      cur_seg_im[parent_coords[:, 0], parent_coords[:, 1], parent_coords[:, 2]] = 1

                  """ use centroid of object to make seed crop """
                  crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(input_im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)
                  crop_seed, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(cur_seg_im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                                                      

                    
                  """ Dilate the seed by sphere 1 to mimic training data """
                  # THIS IS ORIGINAL BALL DILATION 
                  crop_seed = dilate_by_ball_to_binary(crop_seed, radius=dilation)


                  """ Check nothing hanging off edges in seed
                  """
                  # also make sure everything is connected to middle point (no loose seeds)
                  crop_seed = convert_matrix_to_multipage_tiff(crop_seed)
                  crop_seed = np.expand_dims(crop_seed, axis=-1)
                  crop_seed = check_resized(crop_seed, depth, width_max=input_size, height_max=input_size)
                  crop_seed = crop_seed[:, :, :, 0]
                  crop_seed = convert_multitiff_to_matrix(crop_seed)
                       
                  """ Send to segmentor for INFERENCE """
                  crop = np.asarray(crop, np.uint8)
                  crop = np.asarray(crop, np.float32)
                  crop_seed[crop_seed > 0] = 255
                    
                  output_PYTORCH = UNet_inference_PYTORCH(unet, crop, crop_seed, mean_arr, std_arr, device=device)
        
        
        
                  """ SAVE max projections"""
                  plot_save_max_project(fig_num=5, im=crop_seed, max_proj_axis=-1, title='crop seed dilated', 
                                        name=s_path + 'Crop_'  + str(num_tree) + '_' + str(iterator) + '_seed.png', pause_time=0.001)
                  plot_save_max_project(fig_num=2, im=output_PYTORCH, max_proj_axis=-1, title='segmentation', 
                                        name=s_path + 'Crop_'  + str(num_tree) + '_' + str(iterator) + '_segmentation.png', pause_time=0.001)
                  plot_save_max_project(fig_num=3, im=crop, max_proj_axis=-1, title='input', 
                                        name=s_path + 'Crop_'  + str(num_tree) + '_' + str(iterator) + '_input_im.png', pause_time=0.001)

                  """ TURN SEGMENTATION INTO skeleton and assess branch points ect...                  
                          ***might need to smooth the skeleton???
                          (0) pre-process unlinked portions that are very close by???                          
                          (1) use old start_be_coord to find only nearby segments                          
                          (2) skeletonize everything                          
                          (3) find branch points + end points                          
                          (4) create tree node for every branch + end point using the treeify function???                          
                          (5) ***make sure start from last index with nan child and go backwards always???                                    
                  """                  
                  """ (0) PRE-PROCESS unlinked components that are very close by...                   
                          ***AND delete anything that has previously been identified                  
                  """                  
                 
                  """ Things to fix still:
                           - ***WHAT IS THE BEST CROPPING +/- 1???
                           - *** ADD difference between branch point vs. end point in the tree!!! ==> discern by # of children???                                      
                           *** how to deal with edges being cut off???                                          
                           
                           
                           
                           
                           more parents???
                           dont delete small things???
                           
                           
                           maybe propagate slower??? move only 20 pixels at a time???
                           
                           ***also, dilation quite slow right now

        
                    
                                ***why not linking nearby things???                 
                                
                                
                                *** why not continuing some seeds??? set the tree to terminate too early???
                                *** need the spatial W one??? b/c super close to connecting some???
                                *** what's up with the weird one that bends upwards???
                           
                           
                           
                           ***circle instead of cube subtraction??? ==> b/c creating bad cut-offs right now
                           ***linking is still missing a few obvious ones...
                           ***stop letting it propoagate like crazy if only next to nearby small ones
                           
                           *** if new branchpoints are near old one, then all count as the same branchpoint
                               ***==> REMOVED PARENT node visited below!!!
                               ****** also added a dilation below
                               
                           
                           
                           ***spatial weight loss in center???
                           
                           
                      """

                  """ REMOVE EDGE """
                  dist_xy = 10
                  dist_z = 2
                  
                  edge = np.zeros(np.shape(output_PYTORCH)).astype(np.int64)
                  edge[dist_xy:crop_size * 2-dist_xy, dist_xy:crop_size * 2-dist_xy, dist_z:z_size-dist_z] = 1
                  edge = np.where((edge==0)|(edge==1), edge^1, edge)
                  
                  output_PYTORCH[edge == 1] = 0



                  """ ***FIND anything that has previously been identified """
                  ### Just current tree???
                  im = show_tree(tree, track_trees)
                                    
                  ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
                  for cur_tree in all_trees:
                      im += show_tree(cur_tree, track_trees)
                  
                  im[im > 0] = 1
                  crop_prev, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                  crop_prev = skeletonize_3d(crop_prev)
                                    
                  im_dil = dilate_by_ball_to_binary(crop_prev, radius=3)
                  
                  ### ***but exclude the center for subtraction
                  im_sub = subtract_im_no_sub_zero(im_dil, small_center_cube)
                  

                  # subtract old parts of image that have been identified in past rounds of segmentation
                  output_PYTORCH = subtract_im_no_sub_zero(output_PYTORCH, im_sub)

                  """ delete all small objects """
                  # labelled = measure.label(output_PYTORCH)
                  # cc = measure.regionprops(labelled); 
                  # cleaned = np.zeros(np.shape(output_PYTORCH))
                  # for seg in cc:
                  #      coord = seg['coords']; 
                  #      if len(coord) > 10:
                  #          cleaned[coord[:, 0], coord[:, 1], coord[:, 2]] = 1
                           
                           
                  # output_PYTORCH = cleaned
                  
                  """ add in crop seed and subtract later??? """
                  output_PYTORCH = output_PYTORCH + crop_seed
                  output_PYTORCH[output_PYTORCH > 0] = 1
                  
                  """ only keep anything colocalized with center 
                  
                  
                  **** ONLY DILATE END POINTS THOUGH *** """
                  degrees, coordinates = bw_skel_and_analyze(output_PYTORCH)
                  end_points = np.copy(degrees); end_points[end_points != 1] = 0
                  
                  
                  ### BUT ONLY THE END POINTS WITHIN THE CENTRAL CUBE
                  # exclude_cube = np.copy(center_cube)
                  # exclude_cube[exclude_cube > 0] = -1
                  # exclude_cube[exclude_cube == 0] = 1
                  # exclude_cube[exclude_cube == -1] = 0
                  # end_points = subtract_im_no_sub_zero(end_points, exclude_cube)
                  
                  
                  end_points = dilate_by_ball_to_binary(end_points, radius=5)       
                  
                  """ added a little dilation for the output as well """
                  
                  output_PYTORCH = dilate_by_ball_to_binary(output_PYTORCH, radius=1)  
                  output_PYTORCH = output_PYTORCH + end_points
                  output_PYTORCH[output_PYTORCH > 0] = 1
                  output_PYTORCH = skeletonize_3d(output_PYTORCH)
                  output_PYTORCH[output_PYTORCH > 0] = 1  # line before this sets all values to 255
                  
                
                  # (1) use old start_coords to find only nearby segments
                  #cur_be_start               
                  # ***or just use center cube
                  coloc_with_center = output_PYTORCH + center_cube
                  only_coloc = find_overlap_by_max_intensity(bw=output_PYTORCH, intensity_map=coloc_with_center) 
                      

                  ### use to see if should skip
                  sub_seed = subtract_im_no_sub_zero(only_coloc, crop_seed)
                  
                  # if iterator == 26:
                  #      zzz

                  """ skip if everything was subtracted out last time: """
                  if np.count_nonzero(sub_seed) < 5:
                          tree.visited[first_ind] = 1;
                          print('Finished')
                          plot_save_max_project(fig_num=9, im=only_coloc, max_proj_axis=-1, title='segmentation_deleted', 
                                  name=s_path + 'Crop_'  + str(num_tree) + '_' + str(iterator) + '_segmentation_deleted.png', pause_time=0.001)                       
                          plot_save_max_project(fig_num=10, im=only_coloc, max_proj_axis=-1, title='_final_added', 
                                  name=s_path + 'Crop_'  + str(num_tree) + '_' + str(iterator) + '_zfinal_added.png', pause_time=0.001) 
                          iterator += 1
                          continue                      

                  else:

                      
                      # (2) skeletonize the output to create "all_neighborhoods" and "all_hood_first_last"                                    
                      """ Add the old crop back in to ensure correct branch_points ect... 
                      
                      and then delete the old crop that matches the original
                      
                      """
                      # only_coloc = only_coloc + crop_seed
                      # only_coloc[only_coloc > 0] = 1
                      
                      """ *** MAKE THIS INTO A FUNCTION to be shared with treeify !!! """
                      degrees, coordinates = bw_skel_and_analyze(only_coloc)
                      """ insert the current be neighborhood at center and set to correct index in "degrees"
                              or add in the endpoint like I just said, but subtract it out to keep the crop_seed separate???                              
                      """
                      cur_end = np.copy(cur_be_end)
                      cur_end = scale_coords_of_crop_to_full(cur_end, -box_x_min - 1, -box_y_min - 1, -box_z_min - 1)
                      degrees[cur_end[:, 0], cur_end[:, 1], cur_end[:, 2]] = 4
                      

                      plot_save_max_project(fig_num=9, im=degrees, max_proj_axis=-1, title='segmentation_deleted', 
                                  name=s_path + 'Crop_' + str(num_tree) + '_' + str(iterator) + '_segmentation_deleted.png', pause_time=0.001) 
                      
                      all_neighborhoods, all_hood_first_last, root_neighborhood = get_neighborhoods(degrees, coord_root=0, scale=1, box_x_min=box_x_min, box_y_min=box_y_min, box_z_min=box_z_min)

                      """ Find which neighborhood_be matches with cur_be_end to set as new root!!! 
                              Then delete all the rest of the cur_segs that match with crop_seed
                      """
                      check_debug = np.zeros(np.shape(crop))
                      idx = 0; 
                      for neighbor_be in all_neighborhoods:
                         if (cur_be_end[:, None] == neighbor_be).all(-1).any(): 
                             root_neighborhood = neighbor_be
                             print('match')
                             # delete old
                             all_neighborhoods[idx] = []
                             
                         else:
                             cur = np.copy(neighbor_be)
                             cur = scale_coords_of_crop_to_full(cur, -box_x_min - 1, -box_y_min - 1, -box_z_min - 1)
                             check_debug[cur[:, 0], cur[:, 1], cur[:, 2]] = 1
                       
                         idx += 1
                      # delete cur_segs *** NOT NEEDED
                      idx = 0
                      for cur_seg in all_hood_first_last:
                            if (cur_seg[:, None] == tree.coords[first_ind]).all(-1).any():
                                  all_hood_first_last[idx] = []
                            else:
                               cur = np.copy(cur_seg)
                               cur = scale_coords_of_crop_to_full(cur, -box_x_min - 1, -box_y_min - 1, -box_z_min - 1)
                               check_debug[cur[:, 0], cur[:, 1], cur[:, 2]] = 1                 
                            
                            idx += 1
                     
                      plot_save_max_project(fig_num=10, im=check_debug, max_proj_axis=-1, title='_final_added', 
                                  name=s_path + 'Crop_'  + str(num_tree) + '_' + str(iterator) + '_zfinal_added.png', pause_time=0.001) 
                      

                      """ IF is empty (no following part) """
                      if len(all_neighborhoods) == 0:
                          tree.visited[first_ind] = 1;
                          print('Finished')
                          iterator += 1
                          continue
                      
                      else:
                          """ else add to tree """
                          cur_idx = tree.cur_idx[first_ind]
                          depth_tree = tree.depth[first_ind] + 1
                          tree = tree
                          parent =  tree.parent[first_ind]
                          #root_neighborhood = all_neighborhoods[0]
                          
                          tree, list_ = treeify(tree, depth_tree, root_neighborhood, all_neighborhoods, all_hood_first_last, cur_idx = cur_idx, parent= cur_idx, start=1)

                          ### set "visited" to correct value
                          # for idx, node in tree.iterrows():
                          #      if not isListEmpty(node.child):
                          #          node.visited = 1
                          #      elif not node.visited:
                          #          node.visited = np.nan    
                          
                          
                          """ set parent is visited to true """
                          tree.visited[first_ind] = 1;

                          
        
                          print('Finished one iteration'); plt.close('all')
                          iterator += 1
            
             num_tree += 1 
             print('Tree #: ' + str(num_tree) + " of possible: " + str(len(all_trees)))
             
                 
             """ Save max projections and pickle file """
             im = show_tree(tree, track_trees)
             plot_save_max_project(fig_num=6, im=im, max_proj_axis=-1, title='overall seg', 
                                        name=s_path + 'overall_segmentation_' + str(num_tree) + '_.png', pause_time=0.001)

             """ Save max projections and pickle file """
             im[im > 0] = 1
             plot_save_max_project(fig_num=7, im=im, max_proj_axis=-1, title='overall seg', 
                                        name=s_path + 'overall_segmentation_BW' + str(num_tree) + '_.png', pause_time=0.001)        


        


        print('save entire tree')
        ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
        for cur_tree in all_trees:
            im += show_tree(cur_tree, track_trees)
        im[im > 0] = 1
        plot_save_max_project(fig_num=7, im=im, max_proj_axis=-1, title='overall seg', 
                                   name=s_path + 'overall_segmentation_BW' + str(num_tree) + '_.png', pause_time=0.001)        

        print("Saving after first iteration")
        final_seg_overall = convert_matrix_to_multipage_tiff(final_seg_overall)
        imsave(s_path + 'overall_output_1st_iteration.tif', np.asarray(final_seg_overall * 255, dtype=np.uint8))



       
        
        
        
        
        
        
        
    