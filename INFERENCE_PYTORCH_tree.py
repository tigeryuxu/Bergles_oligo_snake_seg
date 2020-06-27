# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger
"""
import matplotlib
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 
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


from scipy.spatial import distance 
from skimage.draw import line_nd

from skimage.transform import rescale, resize, downscale_local_mean

""" Define GPU to use """
import torch
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

""" Decide if use pregenerated seeds or not """
pregenerated = 1
        
"""  Network Begins: """
#check_path ='./(9) Checkpoint_AdamW_batch_norm/'; dilation = 1
#check_path ='./(15) Checkpoint_AdamW_batch_norm_SPATIALW/'; dilation = 1
#check_path = './(12) Checkpoint_AdamW_batch_norm_CYCLIC/'; dilation = 1
#check_path = './(21) Checkpoint_AdamW_batch_norm_3x_branched/'; dilation = 1


#check_path = './(24) Checkpoint_nested_unet_SPATIALW/'; dilation = 1
check_path = './(28) Checkpoint_nested_unet_SPATIALW_complex/'; dilation = 1; deep_supervision = False;


check_path = './(31) Checkpoint_nested_unet_SPATIALW_complex_3x3/'; dilation = 1;

check_path = './(32) Checkpoint_nested_unet_SPATIALW_complex_deep_supervision/'; dilation = 1; deep_supervision = True;

check_path = './(35) Checkpoint_nested_unet_SPATIALW_complex_SWITCH_NORM/'; dilation = 1; deep_supervision = False;


check_path = './(36) Checkpoint_nested_unet_SPATIALW_complex_SWITCH_NORM_medium/'; dilation = 1; deep_supervision = False;

s_path = check_path + 'TEST_inference/'
try:
    # Create target Directory
    os.mkdir(s_path)
    print("Directory " , s_path ,  " Created ") 
except FileExistsError:
    print("Directory " , s_path ,  " already exists")

input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation/'
input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation large/'


input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation large_25px/'

#input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation large_25px/dense/'


#input_path = 'E:/7) Bergles lab data/Traces files/seed generation large_25px/'



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

scale_for_animation = 0
#scale = 1

for i in range(len(examples)):              

        """ (1) Loads data as sorted list of seeds """
        sorted_list, input_im, width_tmp, height_tmp, depth_tmp, overall_coord, all_seeds = load_input_as_seeds(examples, im_num=i, pregenerated=pregenerated, s_path=s_path)   


        input_name = examples[i]['input']
        filename = input_name.split('/')[-1]
        filename = filename.split('.')[0:-1]
        filename = '.'.join(filename)
        
       
        """ scale input im for animations """
        if scale_for_animation:
             input_im_rescaled = convert_matrix_to_multipage_tiff(input_im)   
             input_im_rescaled = rescale(input_im_rescaled, scale_for_animation)
             
            
        """ add seeds to form roots of tree """
        """ (1) First loop through and turn each seed into segments at branch points 
            (2) Then add to list with parent/child indices
        """
    
        """ only 50 """
        all_seeds[all_seeds !=  50] = 0;
        labelled = measure.label(all_seeds)
        cc = measure.regionprops(labelled)
        all_coords_root = []
        for point in cc:
            coord_point = point['coords']
            all_coords_root.append(coord_point)
        #all_coords_root = np.vstack(all_coords_root)

        all_trees = []
        for root in sorted_list:
            tree_df, children = get_tree_from_im_list(root, input_im, width_tmp, height_tmp, depth_tmp, all_coords_root)                                          
            
            ### HACK: continue if empty
            if len(tree_df) == 0:
                continue;

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
            
            # if len(all_trees) == 1:
            #     break
 
        track_trees = np.zeros(np.shape(input_im))    
        num_tree = 0

        """ Create box in middle to only get seed colocalized with middle after splitting branchpoints """
        ball_in_middle = np.zeros([input_size,input_size, z_size])
        ball_in_middle[int(input_size/2), int(input_size/2), int(z_size/2)] = 1
        ball_in_middle_dil = dilate_by_cube_to_binary(ball_in_middle, width=10)
        center_cube = ball_in_middle_dil


        ball_in_middle = np.zeros([input_size,input_size, z_size])
        ball_in_middle[int(input_size/2), int(input_size/2), int(z_size/2)] = 1
        ball_in_middle_dil = dilate_by_cube_to_binary(ball_in_middle, width=8)
        small_center_cube = ball_in_middle_dil
        
        
        matplotlib.use('Agg')
        
        
        for tree in all_trees:
                     
             """ Keep looping until everything has been visited """  
             iterator = 0    
             while np.asarray(tree.visited.isnull()).any():   
                 
                  unvisited_indices = np.where(tree.visited.isnull() == True)[0]
                  first_ind = unvisited_indices[-1]

                  """ GET ALL LAST 2 or 3 coords from parents as well """
                  parent_coords = get_parent_nodes(tree, start_ind=first_ind, num_parents=4, parent_coords=[])
                  
                  if len(parent_coords) > 0:  # check if empty
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
                    
                  output_PYTORCH = UNet_inference_PYTORCH(unet, crop, crop_seed, mean_arr, std_arr, device=device, deep_supervision=deep_supervision)
        
        
        
                  """ SAVE max projections"""
                  plot_save_max_project(fig_num=5, im=crop_seed, max_proj_axis=-1, title='crop seed dilated', 
                                        name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator) + '_seed.png', pause_time=0.001)
                  plot_save_max_project(fig_num=2, im=output_PYTORCH, max_proj_axis=-1, title='segmentation', 
                                        name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_segmentation.png', pause_time=0.001)
                  plot_save_max_project(fig_num=3, im=crop, max_proj_axis=-1, title='input', 
                                        name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_input_im.png', pause_time=0.001)

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
                                                      
                           *** STOP THE SPAGHETTI FROM FORMING ***                           
                                   ***something to do with the dilation???
                                   ***something to do with the fact that need to visit EVERYTHING now... grr...
                                   ***OR can add a failsafe ==> if FOV doesn't move by x-y pixels, then terminate
                           
                            
                           *** Try starting from 0th instead of -1 for iteration
                           
                           *** maybe backup each branchpoint on the subsequent iteration??? b/c missing some turns...
                               ***actually, was due to the cropping being -1!!!
                                                           
                           ***don't subtract in z-axis off the top and bottom of the crop???
                           
                                                                                
                           Today's updates:
                               (1) use line_nd to connect end_points that touch during dilation(or just pixel coords match)
                               (2) loop through and make sure doesn't connect any lines to itself
                               (3) expand the current end point so that new nearby things are not saved as branches
                               
                               ***with these changes, remove the removal of small objects???


                                ***one sided HD loss???


                            



                            ***gets confused in the center with super bright fluorescence
                            ***try to do with automated seeds
                            
                            
                            
                            
                            ***missing the T-crossing after changed from 4 to 5???
                            
                            
                            
                            
                            
                            
                            
                            
                            crop 95 of -1 forward prop ==> maybe set end points at all connection points???
                            
                            ***how best to connect objects only once??? and not to spuriously connect???
                            
                            ***maybe don't just crop the middle of the seg??? at least for the dense network???
                            
                            
                            
                            
                            
                            
                            
                            ***REMOVE BODY CONNECTIONS???
                            
                            
                            
                            
                            
                            *** removed dist_xy and dist_Z??? ==> but misses some due to travelling too fast???
                            
                            ***To check:
                                
                                
                                
                                
                                0_38 ==> okay...
                                0_59 ==> okay later
                                
                                0_90
                                
                            *** TRY automatic ==> missing too many starting seeds
                            
                            
                            ***maybe neighborhood line connect is too lenient right now?
                            
                            
                            ***if turn corner abruptly, often misses other side of turn
                                ***maybe subtract 10 pixels from the placement of every seed???
                            
                            
                            
                            
    
    
    
                      """

                  """ REMOVE EDGE """
                  dist_xy = 0
                  dist_z = 0
                  
                  edge = np.zeros(np.shape(output_PYTORCH)).astype(np.int64)
                  edge[dist_xy:crop_size * 2-dist_xy, dist_xy:crop_size * 2-dist_xy, dist_z:z_size-dist_z] = 1
                  edge = np.where((edge==0)|(edge==1), edge^1, edge)
                  
                  output_PYTORCH[edge == 1] = 0



                  """ ***FIND anything that has previously been identified
                  
                      ***EXCLUDING CURRENT CROP_SEED
                  
                  """
                  ### Just current tree???
                  im = show_tree(tree, track_trees)
                                    
                  ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
                  for cur_tree in all_trees:
                      im += show_tree(cur_tree, track_trees)
                  
                  im[im > 0] = 1
                  crop_prev, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                  crop_prev = skeletonize_3d(crop_prev)
                                    

                  ### EXCLUDE current crop seed
                  im_sub = subtract_im_no_sub_zero(crop_prev, crop_seed)

                    
                  im_dil = dilate_by_ball_to_binary(im_sub, radius=3)
                  
                  
                  ### but add back in current crop seed (so now without dilation)
                  im_dil = im_dil + crop_seed
                  im_dil[im_dil > 0] = 1
                  
                  
                  
                  ### ***but exclude the center for subtraction
                  #im_sub = subtract_im_no_sub_zero(im_dil, small_center_cube)
                  
                  

                 
                  # subtract old parts of image that have been identified in past rounds of segmentation
                  output_PYTORCH = subtract_im_no_sub_zero(output_PYTORCH, im_dil)


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
                  
                  ### (1) Find cc of every object in the current crop
                  labelled = measure.label(output_PYTORCH)
                  cc = measure.regionprops(labelled)

                  all_seg = []
                  for cur in cc:
                      cur_seg = {
                              "coords": cur['coords'],
                              "center_be": [],
                              "expand_be": [],
                              "bridges": [],
                          }
                      all_seg.append(cur_seg)
                          

                  ### (2) Find the end points AND(???) branchpoints
                  degrees, coordinates = bw_skel_and_analyze(output_PYTORCH)
                  be_points = np.copy(degrees); be_points[be_points == 2] = 0; be_points[be_points > 0] = 1;
                  
                  labelled = measure.label(be_points)
                  cc_be = measure.regionprops(labelled)
                  
                  ### (3) get pixel indices of each be AND get the expanded version of the be neighborhood as well 
                  # match the end point to the list of coords
                  for seg in all_seg:
                      cur_seg = seg['coords']
                      for be in cc_be:
                          cur_be = be['coords']
                          if (cur_seg[:, None] == cur_be).all(-1).any():
                              
                              seg["center_be"].append(cur_be)
                              
                              ### expand the cur_be
                              neighborhood_be = expand_coord_to_neighborhood(cur_be, lower=5, upper=6)
                              if len(neighborhood_be) > 0:
                                  neighborhood_be = np.vstack(neighborhood_be)
                              seg["expand_be"].append(neighborhood_be)
                              
                              
                  ### (4) loop through each cc and see if be neighborhood hits nearby cc EXCLUDING itself
                  ### if it hits, use line_nd to make connection
                  empty = np.zeros(np.shape(crop))
                  for cur_seg, cur_idx in zip(all_seg, range(len(all_seg))):
                      cur_expand = cur_seg['expand_be']
                      if len(cur_expand) > 0:
                          for cur_ex, idx_outer in zip(cur_expand, range(len(cur_expand))):  # loop through each be of current seg
                               for next_seg, next_idx in zip(all_seg, range(len(all_seg))):  # loop through all other segs
                                   if cur_idx == next_idx:
                                       continue;   ### don't try to match with self
                                   
                                   ### (a) try to find an expanded neighborhood that matches
                                   match = 0
                                   next_expand = next_seg['expand_be']
                                   if len(next_expand) > 0:
                                       for be_ex, idx_inner in zip(next_expand, range(len(next_expand))): # loop through each be of next seg             
                                             if (cur_ex[:, None] == be_ex).all(-1).any():  
                                                 
                                                 next_be = next_seg['center_be'][idx_inner][0]
                                                 cur_be = cur_seg['center_be'][idx_outer][0]
                                                 
                                                 ### DRAW LINE
                                                 line_coords = line_nd(cur_be, next_be, endpoint=False)
                                                 line_coords = np.transpose(line_coords)
                                                 cur_seg['bridges'].append(line_coords)
                                                 match = 1
                                                 #print('bridge')
        
                          
                                   ### (b) then try to find coords that match ==> ***IF MATCHED, find CLOSEST
                                   if not match:
                                        next_coords = next_seg['coords']
                                        if len(next_coords) > 0 and (cur_ex[:, None] == next_coords).all(-1).any():
                                            cur_be = cur_seg['center_be'][idx_outer][0]
                                            # find distance to all coords
                                           
                                            cur = np.transpose(np.vstack(cur_be))
                                            dist = distance.cdist(cur, next_coords)
                                            min_idx = np.argmin(dist)
                                           
                                            closest_point = next_coords[min_idx]
                                          
                                            ### DRAW LINE
                                            line_coords = line_nd(cur_be, closest_point, endpoint=False)
                                            line_coords = np.transpose(line_coords)
                                            cur_seg['bridges'].append(line_coords)     
                                           
                                            print('body bridge')
                                            print(cur_be)
                                           
                                           
                                      
                                      
            
                  ### debug: ensure proper points inserted
                  """ get output image """
                  output = np.zeros(np.shape(crop))
                  for seg, idx in zip(all_seg, range(len(all_seg))):
                       cur_expand = seg['bridges']
                       if len(cur_expand) > 0:
                           cur_expand = np.vstack(cur_expand)
                           output[cur_expand[:, 0], cur_expand[:, 1], cur_expand[:, 2]] = 5

                       cur_seg = seg['coords']
                       if len(cur_seg) > 0:
                            cur_seg = np.vstack(cur_seg)
                            output[cur_seg[:, 0], cur_seg[:, 1], cur_seg[:, 2]] = idx + 1
                          
                       # cur_be = seg['center_be']
                       # if len(cur_be) > 0:
                       #     cur_be = np.vstack(cur_be)
                       #     output[cur_be[:, 0], cur_be[:, 1], cur_be[:, 2]] = 2
                           
                       # cur_be = seg['expand_be']
                       # if len(cur_be) > 0:
                       #      cur_be = np.vstack(cur_be)
                       #      output[cur_be[:, 0], cur_be[:, 1], cur_be[:, 2]] = 1
                           
                           
                  #plot_max(output, ax=-1)


                  plot_save_max_project(fig_num=3, im=output, max_proj_axis=-1, title='output_be', 
                                        name=s_path + filename + '_Crop_' + str(num_tree) + '_' + str(iterator) + '_output_be.png', pause_time=0.001)
                  output[output > 0] = 1
                  
                  output_PYTORCH = output
                                            
                  
                  
                  """ only keep anything colocalized with center 

                  **** ONLY DILATE END POINTS THOUGH *** """
                  # degrees, coordinates = bw_skel_and_analyze(output_PYTORCH)
                  # end_points = np.copy(degrees); end_points[end_points != 1] = 0
                  
                  
                  # ### BUT ONLY THE END POINTS WITHIN THE CENTRAL CUBE
                  # # exclude_cube = np.copy(center_cube)
                  # # exclude_cube[exclude_cube > 0] = -1
                  # # exclude_cube[exclude_cube == 0] = 1
                  # # exclude_cube[exclude_cube == -1] = 0
                  # # end_points = subtract_im_no_sub_zero(end_points, exclude_cube)
                  
                  
                  # end_points = dilate_by_ball_to_binary(end_points, radius=5)       
                  
                  # """ added a little dilation for the output as well """
                  # #output_PYTORCH = dilate_by_ball_to_binary(output_PYTORCH, radius=1)  
                  # output_PYTORCH = output_PYTORCH + end_points
                  # output_PYTORCH[output_PYTORCH > 0] = 1
                  # output_PYTORCH = skeletonize_3d(output_PYTORCH)
                  # output_PYTORCH[output_PYTORCH > 0] = 1  # line before this sets all values to 255
                  
                
                  # (1) use old start_coords to find only nearby segments
                  #cur_be_start               
                  # ***or just use center cube
                  coloc_with_center = output_PYTORCH + center_cube
                  only_coloc = find_overlap_by_max_intensity(bw=output_PYTORCH, intensity_map=coloc_with_center) 
                      

                  ### use to see if should skip
                  sub_seed = subtract_im_no_sub_zero(only_coloc, crop_seed)
                  
                  # if iterator == 100:
                  #       zzz

                  """ skip if everything was subtracted out last time: """
                  if np.count_nonzero(sub_seed) < 5:
                          tree.visited[first_ind] = 1;
                          print('Finished')
                          plot_save_max_project(fig_num=9, im=only_coloc, max_proj_axis=-1, title='segmentation_deleted', 
                                  name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_segmentation_deleted.png', pause_time=0.001)                       
                          plot_save_max_project(fig_num=10, im=only_coloc, max_proj_axis=-1, title='_final_added', 
                                  name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_zfinal_added.png', pause_time=0.001) 
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
                      
                      
                      
                      
                      ### HACK: fix how so end points cant leave frame
                      """ MIGHT GET TOO LARGE b/c of building up previous end points, so need to ensure crop """
                      cur_end[np.where(cur_end[:, 0] >= crop_size * 2), 0] = crop_size * 2 - 1
                      cur_end[np.where(cur_end[:, 1] >= crop_size * 2), 1] = crop_size * 2 - 1
                      cur_end[np.where(cur_end[:, 2] >= depth), 2] = depth - 1
                      
                      
                      
                      
                      
                      
                      ### Then set degrees
                      degrees[cur_end[:, 0], cur_end[:, 1], cur_end[:, 2]] = 4
                      
                      
                      
                      
                      ###remove all the others that match this first one???
                      
                                           

                      plot_save_max_project(fig_num=9, im=degrees, max_proj_axis=-1, title='segmentation_deleted', 
                                  name=s_path + filename + '_Crop_' + str(num_tree) + '_' + str(iterator) + '_segmentation_deleted.png', pause_time=0.001) 
                      
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
                                  name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator) + '_zfinal_added.png', pause_time=0.001) 
                      

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
        
        
        

       
        
        
        
        
        
        
        
    