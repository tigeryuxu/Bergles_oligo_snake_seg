# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================


 ***NEED TO INSTALL SCIPY???
 
 
 Try reducing network size
 

@author: Tiger



To try:
     - ridge filter for seed generation
     - dilate while training?
     - interpolate while training?


     - advancing cone instead of a square to colocalize broken fragments???



"""



""" ALLOWS print out of results on compute canada """
#from keras import backend as K
#cfg = K.tf.ConfigProto()
#cfg.gpu_options.allow_growth = True
#K.set_session(K.tf.Session(config=cfg))

import matplotlib
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt

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

import kornia

torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 



from matlab_crop_function import *
from off_shoot_functions import *


""" Define GPU to use """
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

""" Decide if use pregenerated seeds or not """
pregenerated = 1
        

# Initialize everything with specific random seeds for repeatability
"""  Network Begins:
"""

check_path ='./(9) Checkpoint_AdamW_batch_norm/'; dilation = 1
check_path ='./(15) Checkpoint_AdamW_batch_norm_SPATIALW/'; dilation = 1
#s_path='./Checkpoints_BELUGA/SCALED_5_Bergles_cropped_forward_prop_DILATED/';  dilation = 2
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


""" Make seeds sparse by deleting branch points"""
#degrees, coordinates = bw_skel_and_analyze(cropped_seed)
#branch_points = np.copy(degrees); branch_points[branch_points != 3] = 0
#cropped_seed[branch_points > 0] = 0

for i in range(len(examples)):              
        """ (1) Loads data as sorted list of seeds """
        sorted_list, input_im, width_tmp, height_tmp, depth_tmp, overall_coord = load_input_as_seeds(examples, im_num=i, pregenerated=pregenerated, s_path=s_path)   

        matplotlib.use('Agg')
        
        # initializes empty arrays
        final_seg_overall = np.zeros(np.shape(input_im))
        each_individual_fiber_trace_coords = []
            
        for seed_idx in range(len(sorted_list)):
            
             """ Creates empty array to track current seed trace """
             trace_mask = np.zeros(np.shape(input_im))
             trace = sorted_list[seed_idx]
             
             """ also skip trace if too short ==> thres == 4 for 1st ieration """
             if len(trace) < 5: continue;
             
             centroid = []
             for idx_trace in range(len(trace)):
                  trace_mask[trace[idx_trace][0], trace[idx_trace][1], trace[idx_trace][2]] = 1
                  if idx_trace == int(len(trace)/2):
                       centroid = [trace[idx_trace][0], trace[idx_trace][1], trace[idx_trace][2]]


             """ FOR LARGER IMAGE WILL GO OUT OF MEMORY IF DONT CROP HERE """
             if pregenerated:
                  overall_coord = sorted_list[0][0]
             
             """ MAYBE DILATE AS CROPS INSTEAD """    
             x = int(overall_coord[0]); y = int(overall_coord[1]); z = int(overall_coord[2])
             crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(trace_mask, y, x, z, 
                                                                                        crop_size=500, z_size=100, height=height_tmp, width=width_tmp, depth=depth_tmp)
             
             
             crop_trace_mask = crop
             
             """ add endpoints to a new list of points to visit (seed_idx) """
             degrees, coordinates = bw_skel_and_analyze(crop_trace_mask)
             end_points = np.copy(degrees); end_points[end_points != 1] = 0
             coords_end_points = np.transpose(np.nonzero(end_points))

             """ FIX ==> also added to scale cropped images instead """
             new_ep_coords = []
             for end_point in coords_end_points:
                  ep_center = scale_coords_of_crop_to_full(end_point, box_x_min, box_y_min, box_z_min)
                  new_ep_coords.append(ep_center)   
             coords_end_points = new_ep_coords
             
                            
             list_seed_centers = []
             for ep in coords_end_points:
                  list_seed_centers.append(ep)
             #list_seed_centers.append(seed_center)
             already_visited = []
             
             """ also, create an empty array to keep track of what has been seeded/segmented """
             #track_seg = np.zeros(np.shape(input_im))
             track_seg = trace_mask
             
             """ Create sphere in middle to expand net of matching """
             ball_in_middle = np.zeros([input_size,input_size, z_size])
             ball_in_middle[int(input_size/2), int(input_size/2), int(z_size/2)] = 1
             ball_in_middle_dil = dilate_by_cube_to_binary(ball_in_middle, width=20)
             dil_end_points = ball_in_middle_dil
             
             
             """ Create seed limiting box """
             ball_in_middle = np.zeros([input_size,input_size, z_size])
             ball_in_middle[int(input_size/2), int(input_size/2), int(z_size/2)] = 1
             ball_in_middle_dil = dilate_by_cube_to_binary(ball_in_middle, width=20)
             seed_limiting_box = ball_in_middle_dil             
             

             """ Create box in middle to only get seed colocalized with middle after splitting branchpoints """
             ball_in_middle = np.zeros([input_size,input_size, z_size])
             ball_in_middle[int(input_size/2), int(input_size/2), int(z_size/2)] = 1
             ball_in_middle_dil = dilate_by_cube_to_binary(ball_in_middle, width=4)
             center_cube = ball_in_middle_dil
             
             
             
             """ Keep looping until there are no more seed centers to visit """  
             iterator = 0            
             while len(np.unique(list_seed_centers)) != 0:
                  
                  """ Garbage collection """
                  crop = []; crop_trace_mask = []; degrees = []; centroid = [];
                  end_points = []; labelled = []; only_colocalized_mask = []; trace_mask = [];
                  
                  batch_x = []; batch_y = []; weights = [];
                  x = int(list_seed_centers[0][0]); y = int(list_seed_centers[0][1]); z = int(list_seed_centers[0][2])

                  """ More strict skipping """
                  if iterator == 0:
                       track_seg_old = []
                       
                  else:
                       if iterator > 500 and track_seg_old[x,y,z] > 0:
                            print('skipped becuase track seg old')
                            already_visited.append(list_seed_centers[0])
                            del list_seed_centers[0]
                            iterator += 1
                            continue;    

                  """ use centroid of object to make seed crop """
                  crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(input_im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)
                  crop_seed, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(track_seg, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                                                      

                  # crop_seed = dilate_by_ball_to_binary(crop_seed, radius=1)    
                  # """ limit to only seed in the middle minus all branchpoints """
                  # #center_cube
                  # crop_seed[crop_seed > 0] = 1                  
                  # test = skeletonize_3d(crop_seed);  test[test > 0] = 1
                  # degrees, coordinates = bw_skel_and_analyze(test)
                  # branch_points = np.copy(degrees); branch_points[branch_points != 3] = 0   
                  
                  # test[branch_points > 0 ] = 0 
                  # coloc_with_end_points = test + center_cube
                  
                  # only_coloc = find_overlap_by_max_intensity(bw=test, intensity_map=coloc_with_end_points) 
                  # crop_seed = only_coloc
                       
                  """ Delete seeds that are too small """
                  if np.count_nonzero(crop_seed) <= 10:
                       crop_seed[:, :, :] = 0
                  
                    
                  """ Dilate the seed by sphere 2 to mimic training data """
                  # THIS IS ORIGINAL BALL DILATION 
                  crop_seed = dilate_by_ball_to_binary(crop_seed, radius=dilation)

                  
                  """ Limit seed to only smaller size!!! 40 x 40 box in the middle
                       to cleave off edges. Then use to 
                  """
                  
                  #crop_seed[seed_limiting_box == 0] = 0;

                       
                  # also make sure everything is connected to middle point (no loose seeds)
                  crop_seed = convert_matrix_to_multipage_tiff(crop_seed)
                  crop_seed = np.expand_dims(crop_seed, axis=-1)
                  crop_seed = check_resized(crop_seed, depth, width_max=input_size, height_max=input_size)
                  crop_seed = crop_seed[:, :, :, 0]
                  crop_seed = convert_multitiff_to_matrix(crop_seed)
                  
                       
                  """ Send to segmentor!!! """
                  #batch_x = []; batch_y = []; weights = [];
                  crop = np.asarray(crop, np.uint8)
                  crop = np.asarray(crop, np.float32)
                  crop_seed[crop_seed > 0] = 255

                    
                  output_PYTORCH = UNet_inference_PYTORCH(unet, crop, crop_seed, mean_arr, std_arr, device=device)
                  depth_last_tmp = output_PYTORCH
  
        
                  """ SAVE max projections"""
                  plot_save_max_project(fig_num=5, im=crop_seed, max_proj_axis=-1, title='crop seed dilated', 
                                        name=s_path + 'Crop_' + str(seed_idx) + '_' + str(iterator) + '_seed.png', pause_time=0.001)
                  plot_save_max_project(fig_num=2, im=depth_last_tmp, max_proj_axis=-1, title='segmentation', 
                                        name=s_path + 'Crop_' + str(seed_idx) + '_' + str(iterator) + '_segmentation.png', pause_time=0.001)
                  plot_save_max_project(fig_num=3, im=crop, max_proj_axis=-1, title='input', 
                                        name=s_path + 'Crop_' + str(seed_idx) + '_' + str(iterator) + '_input_im.png', pause_time=0.001)


                  """ Also save/extract paranodes """
                  paranodes = np.copy(depth_last_tmp)

                  depth_last_tmp[depth_last_tmp > 0] = 1
                  seg_test = depth_last_tmp
          
                  """ Clean up segmentation by imopen/imclose """
                  #""" Delete anything that is only 10 pixel large """
                  #intensity_map = np.copy(seg_test)
                  #intensity_map[seg_test > 0] = 2
                  #seg_test = find_overlap_by_max_intensity(bw=seg_test, intensity_map=intensity_map, min_size_obj=10)                 
                       
                  """ also only keep segments that are near to end points of the original seed """
                  crop_seed[crop_seed > 0] = 1
                  coloc_with_end_points = dil_end_points + seg_test
                  bw_coloc = coloc_with_end_points > 0
                  only_coloc = find_overlap_by_max_intensity(bw=seg_test, intensity_map=coloc_with_end_points) 
                  
                  # FIXED, instead of adding giant circles into seg_test final
                  seg_test[only_coloc == 0] = 0
                  seg_test = only_coloc
                  
                 
                  """ Only keep paranodes that are colocalized as well """
                  #paranodes[paranodes < 2] = 0
                  #paranodes[paranodes > 0] = 1
                  #paranodes[seg_test == 0] = 0
                  
                  """ and delete paranodes from seg_test for stopping error propagation """
                  #seg_test[paranodes > 0] = 0

                  """ Clean up segmentation by imopen/imclose """
                  seg_test[crop_seed > 0] = 1
                  seg_test = dilate_by_ball_to_binary(seg_test, radius=2)
                  #seg_test = erode_by_ball_to_binary(seg_test, radius=3)    
                       
                       
                  """ Must skeletonize segmentation before saving in track_seg because otherwise will grow iteratively each time with new dilation
                       so instead, must save ANOTHER array for tracking dilated segmentations if want to keep those...
                  """
                  seg_test = skeletonize_3d(seg_test); seg_test[seg_test > 0] = 1
                  #crop_seed = skeletonize_3d(crop_seed);  crop_seed[crop_seed > 0] = 1

                  """ EXTRA IN SECOND ITERATION ==> DELETE ALL SEGMENTATIONS THAT HAVE ALREADY BEEN IDENTIFIED
                      BUT, EXCLUDING THOSE WITHIN THE ENDPOINT BEING STUDIED
                  """
                  # get crop from previous segmentations
                  track_seg_crop_delete = final_seg_overall[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] + track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max]
                  track_seg_crop_delete[track_seg_crop_delete > 0] = 1
                  track_seg_crop_delete = dilate_by_ball_to_binary(track_seg_crop_delete, radius=2)

                  # subtract old parts of image that have been identified in past rounds of segmentation
                  seg_test = subtract_im_no_sub_zero(seg_test, track_seg_crop_delete)
       
                  plot_save_max_project(fig_num=9, im=seg_test, max_proj_axis=-1, title='segmentation_deleted', 
                              name=s_path + 'Crop_' + str(seed_idx) + '_' + str(iterator) + '_segmentation_deleted.png', pause_time=0.001)
                         
                                    
                  """ Add segmentation to track_seg array *** must take actual crop coords from cropping function above"""
                  track_seg_old = []
                  track_seg_old = np.copy(track_seg)

                  if iterator == 0:
                       crop_seed = skeletonize_3d(crop_seed);  crop_seed[crop_seed > 0] = 1
                       track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] = track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] + crop_seed + seg_test
                  else:
                       """ Add paranodes """
                       #paranodes = skeletonize_3d(paranodes); paranodes[paranodes > 0] = 1
                       #seg_test[paranodes > 0] = 2
                       
                       """ Also prevent overlap so can have paranodes in value == 2 """
                       prevent_overlap = track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max]
                       seg_test[prevent_overlap > 0] = 0
                       
                       track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] = track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] + seg_test
                  
                  """ Now get a crop that includes what has already been segmented in the area """
                  track_seg_crop = np.copy(seg_test)
                  track_seg_crop[track_seg_crop > 0] = 1
                  crop_seed = dilate_by_ball_to_binary(crop_seed, radius=5)
                  
                  # error checking to see if empty array
                  skip, already_visited, list_seed_centers = check_empty(track_seg_crop, already_visited, list_seed_centers, reason='skipped becuase deleted')
                  if skip: iterator +=1; continue;
                  
                  """ Then start looking for end-points and append them to the list of points to go to """
                  degrees, coordinates = bw_skel_and_analyze(track_seg_crop)
                  degrees[crop_seed == 1] = 0    # NEW TIGER ADDED to subtract out old end points

                  branch_points = np.copy(degrees); branch_points[branch_points != 3] = 0
                  end_points = np.copy(degrees); end_points[end_points != 1] = 0
                  
                  
                  """get coordinates of end points and convert to seed_center on larger image! """
                  """ Make sure to only get the SINGLE middle point, and NOT the entire endpoint object """
                  labelled = measure.label(end_points)
                  cc_end_points = measure.regionprops(labelled); new_ep_coords = []
                  for end_point in cc_end_points:
                       ep_center = end_point['centroid']; ep_center = np.asarray(ep_center)
                       ep_center = scale_coords_of_crop_to_full(ep_center, box_x_min, box_y_min, box_z_min)
                       new_ep_coords.append(ep_center)   
                       
                  """ Include points on edge of image as end_points in "new_ep_coords" array """
                  # error checking to see if empty array
                  skip, already_visited, list_seed_centers = check_empty(coordinates, already_visited, list_seed_centers, reason='skipped becuase no coordinates')
                  if skip: iterator +=1; continue;
                       
                  coordinates = np.delete(coordinates, 0, axis=0) # removes first zeros                                 
                  max_crop_size = crop_size * 2
                  max_z_size = z_size
                  for min_max_check in coordinates:
                       if min_max_check[0] == 0 or min_max_check[1] == 0 or min_max_check[2] == 0 or min_max_check[0] == max_crop_size or min_max_check[1] == max_crop_size or min_max_check[2] == max_z_size:                                            
                            min_max_check_scaled = scale_coords_of_crop_to_full(min_max_check, box_x_min, box_y_min, box_z_min)
                            new_ep_coords.append(min_max_check_scaled)

                  """ Skip if empty """
                  skip, already_visited, list_seed_centers = check_empty(new_ep_coords, already_visited, list_seed_centers, reason='skipped becuase no coordinates')
                  if skip: iterator +=1; continue;
                  
                  """ append the point that we just went to """
                  already_visited.append(list_seed_centers[0])
                  del list_seed_centers[0]
                  
                  """ Check if point already exists in previous segmentation. If so, skip it"""
                  if list_seed_centers:
                       append_lists = np.append(list_seed_centers, new_ep_coords, axis=0)
                       unique_seed_centers = np.unique(append_lists, axis=0)
                  else:
                       unique_seed_centers = new_ep_coords
                    
                  """ also check if already visited. If so, skip it """
                  not_visited = []
                  for unique in unique_seed_centers:
                       bool_visited = 0
                       for visited in already_visited:
                            if np.array_equal(unique, visited):
                                 bool_visited = 1
                                 break
                            # IF NOT MORE THAN average 5 pixels away in some dimension, then exclude
                            elif np.mean(np.abs(unique - visited)) < 2 :
                                 bool_visited = 1
                                 break
                       if not bool_visited:
                         not_visited.append(unique)
 
                  list_seed_centers = not_visited
                  print('Finished one ep cycle'); plt.close('all')
                  iterator += 1
               
             """ Garbage collection """
             track_seg_old = [];     
             
             final_seg_overall = final_seg_overall + track_seg 
             
             coords_track_seg = np.transpose(np.nonzero(track_seg))
             each_individual_fiber_trace_coords.append(coords_track_seg)
             
             print('Seed_idx #: ' + str(seed_idx) + " of possible: " + str(len(sorted_list)))
                   
             """ Save max projections and pickle file """
             plot_save_max_project(fig_num=6, im=final_seg_overall, max_proj_axis=-1, title='overall seg', 
                                        name=s_path + 'overall_segmentation_' + str(seed_idx) + '_.png', pause_time=0.001)
             if seed_idx <= 1:
                  plot_save_max_project(fig_num=7, im=input_im, max_proj_axis=-1, title='input overall', 
                                        name=s_path + 'input.png', pause_time=0.001)
             
             """ Garbage collection """
             track_seg = []
        
        """ Garbage ==> can't save pickle for enormous input """
        #save_pkl(final_seg_overall, s_path, 'overall_segmentation.pkl')
        print("Saving after first iteration")
        final_seg_overall = convert_matrix_to_multipage_tiff(final_seg_overall)
        imsave(s_path + 'overall_output_1st_iteration.tif', np.asarray(final_seg_overall * 255, dtype=np.uint8))












        """ Create sphere in middle to expand net of matching """
        ball_in_middle = np.zeros([input_size,input_size, z_size])
        ball_in_middle[int(input_size/2), int(input_size/2), int(z_size/2)] = 1
        ball_in_middle_dil = dilate_by_cube_to_binary(ball_in_middle, width=30)
        dil_end_points = ball_in_middle_dil












#         """ If change to right teeter, must set delete from list as right teeter as well """

#          ## Getting back the objects:        
#          #with open(s_path + 'overall_segmentation.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
#          #    loaded = pickle.load(f)
#          #    final_seg_overall = loaded[0] 


#         final_seg_overall = convert_multitiff_to_matrix(final_seg_overall)

#         """ Second iteration of analysis, same as before, but now use end points from previous mask
#                  ***AND SUBTRACT OUT THE ORIGINAL SPACE THAT HAS BEEN SURVEYED ALREADY EVERY TIME YOU RUN THE ANALYSIS
#         """
#         second_iter_each_individual_fiber_trace_coords = []
#         second_iter_branch_number = []
#         trace_idx = 0
#         min_trace_size = 20 # pixels from 1st iteration, or skip here
        
#         each_individual_fiber_trace_coords = sorted(each_individual_fiber_trace_coords, key=len, reverse=True)  
#         for trace in each_individual_fiber_trace_coords:
             
#              trace = each_individual_fiber_trace_coords[trace_idx]
             
#              """ also skip trace if too short ==> thres == 10 for 2nd ieration """
#              if len(trace) < min_trace_size: trace_idx += 1; continue;             
             
#              trace_mask = np.zeros(np.shape(input_im))
#              trace_idx += 1
# #             for idx_trace in range(len(trace)):
# #                  trace_mask[trace[idx_trace][0], trace[idx_trace][1], trace[idx_trace][2]] = 1
# #                  
# #             """ add endpoints to a new list of points to visit (seed_idx) """
# #             degrees, coordinates = bw_skel_and_analyze(trace_mask)
# #             end_points = np.copy(degrees); end_points[end_points != 1] = 0
# #             coords_end_points = np.transpose(np.nonzero(end_points))
             
             
#              centroid = []
#              for idx_trace in range(len(trace)):
#                   trace_mask[trace[idx_trace][0], trace[idx_trace][1], trace[idx_trace][2]] = 1
#                   if idx_trace == int(len(trace)/2):
#                        centroid = [trace[idx_trace][0], trace[idx_trace][1], trace[idx_trace][2]]


#              """ FOR LARGER IMAGE WILL GO OUT OF MEMORY IF DONT CROP HERE """
#              #labelled = measure.label(trace_mask)
#              #cc_coloc = measure.regionprops(labelled)
     
#              #overall_coord = np.asarray(cc_coloc[0]['centroid']);
#              centroid[0] = int(centroid[0]);
#              centroid[1] = int(centroid[1]);
#              centroid[2] = int(centroid[2]);
             
#              """ MAYBE DILATE AS CROPS INSTEAD """    
#              x = int(overall_coord[0])
#              y = int(overall_coord[1])
#              z = int(overall_coord[2])
#              crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(trace_mask, y, x, z, 
#                                      crop_size=500, z_size=100, height=height_tmp, width=width_tmp, depth=depth_tmp)
#              crop_trace_mask = crop
             
#              """ add endpoints to a new list of points to visit (seed_idx) """
#              degrees, coordinates = bw_skel_and_analyze(crop_trace_mask)
#              end_points = np.copy(degrees); end_points[end_points != 1] = 0
#              coords_end_points = np.transpose(np.nonzero(end_points))

#              """ FIX ==> also added to scale cropped images instead """
#              new_ep_coords = []
#              for end_point in coords_end_points:
#                   #ep_center = end_point['centroid']; ep_center = np.asarray(ep_center)
#                   ep_center = scale_coords_of_crop_to_full(end_point, box_x_min, box_y_min, box_z_min)
#                   new_ep_coords.append(ep_center)   
#              coords_end_points = new_ep_coords
             
                  
#              """ Then loop exactly as same above except for the subtraction of the trace_seg_mask from iteration #1 """
#              """ Now start looping through each seed point to crop out the image """
#              """ Make a list of centroids to keep track of all the new locations to visit """
#              for seed_idx in range(len(coords_end_points)): # extra 2nd iter
#                   seed_center = coords_end_points[seed_idx]  # extra 2nd iter
#                   seed_center = np.asarray(seed_center)  
#                   seed_center[0] = int(seed_center[0]); seed_center[1] = int(seed_center[1]); seed_center[2] = int(seed_center[2])
     
#                   list_seed_centers = []
#                   list_seed_centers.append(seed_center)
#                   already_visited = []
                               
#                   """ EXTRA: 2nd iter: use trace_mask instead """
#                   track_seg = trace_mask
     
#                   """ Keep looping until there are no more seed centers to visit """  
#                   iterator = 0
#                   while list_seed_centers:   
#                        """ Garbage collection """
#                        crop = []; crop_trace_mask = []; degrees = [];
#                        end_points = []; labelled = []; only_colocalized_mask = []; #trace_mask = [];

#                        batch_x = []; batch_y = []; weights = [];
#                        """ *************** switch to teeter right in 2nd iteration by always sampling the LAST seed_center idx *********** """
#                        x = int(list_seed_centers[0][0]); y = int(list_seed_centers[0][1]); z = int(list_seed_centers[0][2])
#                        #height_tmp = width_tmp = input_tmp_size   # NEED TO FIX THIS, get actual input im size
#                        #depth_tmp = depth_tmp_size
     
     
#                        """ EXTRA: 2nd iter remove this, use trace_mask instead """
#                        #if iterator == 0: track_seg_old = np.zeros(np.shape(input_im))
#                        """ ALSO EXTRA FOR 2nd iteration ==> change iterator > 0 instead of iterator > 3"""     
#                        if iterator > 1000 and track_seg_old[x,y,z] > 0:
#                             print('skipped')
#                             already_visited.append(list_seed_centers[0])
#                             del list_seed_centers[0]
#                             iterator += 1
#                             continue;    
     
#                        """ use centroid of object to make seed crop """
#                        crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(input_im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)
#                        crop_seed, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(track_seg, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                                                      
     
                            
#                        # crop_seed = dilate_by_ball_to_binary(crop_seed, radius=1)    
#                        # """ limit to only seed in the middle minus all branchpoints """
#                        # #center_cube
#                        # crop_seed[crop_seed > 0] = 1                  
#                        # test = skeletonize_3d(crop_seed);  test[test > 0] = 1
#                        # degrees, coordinates = bw_skel_and_analyze(test)
#                        # branch_points = np.copy(degrees); branch_points[branch_points != 3] = 0   
                       
#                        # test[branch_points > 0 ] = 0 
#                        # coloc_with_end_points = test + center_cube
                       
#                        # only_coloc = find_overlap_by_max_intensity(bw=test, intensity_map=coloc_with_end_points) 
#                        # crop_seed = only_coloc
                            
#                        if np.count_nonzero(crop_seed) <= 10:
#                             crop_seed[:, :, :] = 0                           
     
#                        """ Dilate the seed by sphere 2 to mimic training data """
#                        # THIS IS ORIGINAL BALL DILATION 
#                        crop_seed = dilate_by_ball_to_binary(crop_seed, radius=dilation)
     
                       
#                        """ Limit seed to only smaller size!!! 40 x 40 box in the middle """
                       
#                        #crop_seed[seed_limiting_box == 0] = 0;
     
                            
#                        # also make sure everything is connected to middle point (no loose seeds)
#                        crop_seed = convert_matrix_to_multipage_tiff(crop_seed)
#                        crop_seed = np.expand_dims(crop_seed, axis=-1)
#                        crop_seed = check_resized(crop_seed, depth, width_max=input_size, height_max=input_size)
#                        crop_seed = crop_seed[:, :, :, 0]
#                        crop_seed = convert_multitiff_to_matrix(crop_seed)
                       
                            
#                        """ Send to segmentor!!! """
#                        #batch_x = []; batch_y = []; weights = [];
#                        crop = np.asarray(crop, np.uint8)
#                        crop = np.asarray(crop, np.float32)
#                        crop_seed[crop_seed > 0] = 255
     
#                        depth_last_tmp, batch_x, batch_y, weights, input_im_save, output_softMax = UNet_inference(crop, crop_seed, batch_x, batch_y, weights, mean_arr, std_arr, x_3D, y_3D_, weight_matrix_3D, softMaxed, training, num_truth_class)
             
          
                       
#                        """ Also save/extract paranodes """
#                        paranodes = np.copy(depth_last_tmp)
                       
                       
#                        depth_last_tmp[depth_last_tmp > 0] = 1
#                        seg_test = depth_last_tmp
                       
#                        """ SAVE max projections"""
#                        plot_save_max_project(fig_num=11, im=crop_seed, max_proj_axis=-1, title='crop seed dilated', 
#                                              name=s_path + '_' + '_2nd_iter_' + str(trace_idx) +  '_' + str(seed_idx) + '_' + str(iterator) + '_seed_crop.png', pause_time=0.001)
#                        plot_save_max_project(fig_num=12, im=depth_last_tmp, max_proj_axis=-1, title='segmentation', 
#                                              name=s_path + '_' + '_2nd_iter_' + str(trace_idx) +  '_' + str(seed_idx) + '_' + str(iterator) + '_segmentation.png', pause_time=0.001)
#                        plot_save_max_project(fig_num=13, im=crop, max_proj_axis=-1, title='input', 
#                                              name=s_path + '_' + '_2nd_iter_' + str(trace_idx) +  '_' + str(seed_idx) + '_' + str(iterator) + '_input_crop.png', pause_time=0.001)
                       

                       
#                        """ also only keep segments that are near to end points of the original seed """
#                        crop_seed[crop_seed > 0] = 1
#                        #crop_seed = skeletonize_3d(crop_seed);  crop_seed[crop_seed > 0] = 1
#                        #
#                        #degrees, coordinates = bw_skel_and_analyze(crop_seed)
#                        #end_points = np.copy(degrees); end_points[end_points != 1] = 0                  
#                        #dil_end_points = dilate_by_ball_to_binary(end_points, radius=10)
                      
#                        coloc_with_end_points = dil_end_points + seg_test
#                        bw_coloc = coloc_with_end_points > 0

#                        #only_coloc = find_overlap_by_max_intensity(bw=bw_coloc, intensity_map=coloc_with_end_points)   # WRONG!!!
#                        only_coloc = find_overlap_by_max_intensity(bw=seg_test, intensity_map=coloc_with_end_points) 
                  
                    
#                        seg_test[only_coloc == 0] = 0
#                        seg_test = only_coloc
                       
#                        """ Only keep paranodes that are colocalized as well """
#                        paranodes[paranodes < 2] = 0
#                        paranodes[paranodes > 0] = 1
#                        paranodes[seg_test == 0] = 0
                       
#                        """ and delete paranodes from seg_test for stopping error propagation """
#                        seg_test[paranodes > 0] = 0
#                        #if np.count_nonzero(paranodes):
#                        #     zzzzzz
                                              
  
#                        """ Clean up segmentation by imopen/imclose """
#                        seg_test[crop_seed > 0] = 1
#                        seg_test = dilate_by_ball_to_binary(seg_test, radius=2)
#                        #seg_test = erode_by_ball_to_binary(seg_test, radius=3)     

                     
#                        """ Must skeletonize segmentation before saving in track_seg because otherwise will grow iteratively each time with new dilation
#                             so instead, must save ANOTHER array for tracking dilated segmentations if want to keep those...
#                        """
#                        seg_test = skeletonize_3d(seg_test); seg_test[seg_test > 0] = 1
#                        #crop_seed = skeletonize_3d(crop_seed);  crop_seed[crop_seed > 0] = 1
                       
                       

#                        """ no longer EXTRA IN SECOND ITERATION ==> DELETE ALL SEGMENTATIONS THAT HAVE ALREADY BEEN IDENTIFIED
#                            BUT, EXCLUDING THOSE WITHIN THE ENDPOINT BEING STUDIED
#                        """
#                        ## end point being studied:                      
#                        #cur_ep = np.zeros(np.shape(crop_seed))
#                        #cur_ep[x - box_x_min,y - box_y_min,z - box_z_min] = 1
#                        #dil_cur_ep = dilate_by_cube_to_binary(cur_ep, width=5)
                       
#                        # get crop from previous segmentations
#                        #track_seg_crop_delete = final_seg_overall[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max]
#                        #track_seg_crop_delete = dilate_by_ball_to_binary(track_seg_crop_delete, radius=2)
                       
#                        ## exclude regions of the end point (so keep them in the final analysis)
#                        #track_seg_crop_delete = subtract_im_no_sub_zero(track_seg_crop_delete, dil_cur_ep)
                       
#                        # subtract old parts of image that have been identified in past rounds of segmentation
#                        #seg_test = subtract_im_no_sub_zero(seg_test, track_seg_crop_delete)
                       
#                        """ Changed to this from first round """      
#                        # get crop from previous segmentations
#                        track_seg_crop_delete = final_seg_overall[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] + track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max]
#                        track_seg_crop_delete[track_seg_crop_delete > 0] = 1
#                        track_seg_crop_delete = dilate_by_ball_to_binary(track_seg_crop_delete, radius=2)
     
#                        # subtract old parts of image that have been identified in past rounds of segmentation
#                        seg_test = subtract_im_no_sub_zero(seg_test, track_seg_crop_delete)                       
                            
                       
#                        plot_save_max_project(fig_num=14, im=seg_test, max_proj_axis=-1, title='segmentation_deleted', 
#                                              name=s_path + '_' + '_2nd_iter_' + str(trace_idx) + '_' + str(seed_idx) + '_' + str(iterator) + '_segmentation_deleted.png', pause_time=0.001)
         
  
#                        """ Add paranodes """
#                        paranodes = skeletonize_3d(paranodes); paranodes[paranodes > 0] = 1
#                        seg_test[paranodes > 0] = 2

                       
#                        """ Also prevent overlap so can have paranodes in value == 2 """
#                        prevent_overlap = track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max]
#                        seg_test[prevent_overlap > 0] = 0
                       
                       
#                        """ Add segmentation to track_seg array *** must take actual crop coords from cropping function above"""
#                        track_seg_old = np.copy(track_seg)
#                        track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] = track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] + seg_test
                       
#                        """ Now get a crop that includes what has already been segmented in the area """
#                        #track_seg_crop = np.copy(track_seg[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max])
#                        #track_seg_crop[track_seg_crop > 0] = 1
#                        track_seg_crop = np.copy(seg_test)
#                        track_seg_crop[track_seg_crop > 0] = 1

#                        crop_seed = dilate_by_ball_to_binary(crop_seed, radius=5)
                       

#                        """ EXTRA: also delete old areas here """
#                        seg_test_deleted = track_seg_crop - track_seg_crop_delete
#                        seg_test_deleted[seg_test_deleted < 0] = 0
#                        track_seg_crop = seg_test_deleted
                       
                       
#                        """ Delete anything that is only 1 pixel large """
#                        intensity_map = np.copy(track_seg_crop)
#                        intensity_map[track_seg_crop > 0] = 2
#                        track_seg_crop = find_overlap_by_max_intensity(bw=track_seg_crop, intensity_map=intensity_map, min_size_obj=1)                 
                       
#                        # error checking to see if empty array
#                        skip, already_visited, list_seed_centers = check_empty(track_seg_crop, already_visited, list_seed_centers, reason='skipped becuase deleted')
#                        if skip: iterator +=1; continue;                  
     
#                        """ Then start looking for end-points and append them to the list of points to go to """
#                        degrees, coordinates = bw_skel_and_analyze(track_seg_crop)
#                        degrees[crop_seed == 1] = 0    # NEW TIGER ADDED to subtract out old end points

                       
#                        branch_points = np.copy(degrees); branch_points[branch_points != 3] = 0
#                        end_points = np.copy(degrees); end_points[end_points != 1] = 0
                       
#                        """get coordinates of end points and convert to seed_center on larger image! """
#                        """ Make sure to only get the SINGLE middle point, and NOT the entire endpoint object """
#                        labelled = measure.label(end_points)
#                        cc_end_points = measure.regionprops(labelled); new_ep_coords = []
#                        for end_point in cc_end_points:
#                             ep_center = end_point['centroid']; ep_center = np.asarray(ep_center)
#                             ep_center = scale_coords_of_crop_to_full(ep_center, box_x_min, box_y_min, box_z_min)
#                             new_ep_coords.append(ep_center)   
                            
#                        """ Include points on edge of image as end_points in "new_ep_coords" array """
#                        # error checking to see if empty array
#                        skip, already_visited, list_seed_centers = check_empty(coordinates, already_visited, list_seed_centers, reason='skipped becuase no coordinates')
#                        if skip: iterator +=1; continue;
                            
#                        coordinates = np.delete(coordinates, 0, axis=0) # removes first zeros
#                        max_crop_size = crop_size * 2
#                        max_z_size = z_size
#                        for min_max_check in coordinates:
#                             if min_max_check[0] == 0 or min_max_check[1] == 0 or min_max_check[2] == 0 or min_max_check[0] == max_crop_size or min_max_check[1] == max_crop_size or min_max_check[2] == max_z_size:                                            
#                                  min_max_check_scaled = scale_coords_of_crop_to_full(min_max_check, box_x_min, box_y_min, box_z_min)
#                                  new_ep_coords.append(min_max_check_scaled)
                                 
#                        """ Skip if empty """
#                        skip, already_visited, list_seed_centers = check_empty(new_ep_coords, already_visited, list_seed_centers, reason='skipped becuase no coordinates')
#                        if skip: iterator +=1; continue;

                                 
#                        """ append the point that we just went to """
#                        already_visited.append(list_seed_centers[0])
#                        del list_seed_centers[0]
                       
#                        """ Check if point already exists in previous segmentation. If so, skip it"""
#                        if list_seed_centers and len(np.unique(new_ep_coords)) > 0:
#                             append_lists = np.append(list_seed_centers, new_ep_coords, axis=0)
#                             unique_seed_centers = np.unique(append_lists, axis=0)
#                        else:
#                             unique_seed_centers = new_ep_coords
                         
#                        """ also check if already visited. If so, skip it """
#                        not_visited = []
#                        for unique in unique_seed_centers:
#                             bool_visited = 0
#                             for visited in already_visited:
#                                  if np.array_equal(unique, visited):
#                                       bool_visited = 1
#                                       break
#                                  # IF NOT MORE THAN average 2 pixels away in some dimension, then exclude
#                                  elif np.mean(np.abs(unique - visited)) < 2 :
#                                       bool_visited = 1
#                                       break
#                             if not bool_visited:
#                               not_visited.append(unique)
      
#                        list_seed_centers = not_visited
#                        print('Finished one ep cycle'); plt.close('all')
#                        iterator += 1

#                   """ Garbage collection """
#                   track_seg_old = [];     
                       
#                   final_seg_overall = final_seg_overall + track_seg 
                  
#                   coords_track_seg = np.transpose(np.nonzero(track_seg))
#                   second_iter_each_individual_fiber_trace_coords.append(coords_track_seg)
#                   second_iter_branch_number.append(trace_idx - 1)
                  
#                   print('Seed_idx #: ' + str(seed_idx))

#                   """ Garbage collection """
#                   track_seg = [];     

                        
#              """ Save max projections and pickle file """
#              plot_save_max_project(fig_num=20, im=final_seg_overall, max_proj_axis=-1, title='overall seg', 
#                                         name=s_path + 'overall_segmentation_2nd_iter_' + str(trace_idx - 1) + '.png', pause_time=0.001)
#              final_seg_overall[final_seg_overall > 0] = 1
#              plot_save_max_project(fig_num=21, im=final_seg_overall, max_proj_axis=-1, title='input overall', 
#                                         name=s_path + 'overall_segmentation_2nd_iter_BINARY_' + str(trace_idx - 1) + '.png', pause_time=0.001)
                        
             
#         """ Save as outputs """
#         print("Saving post-processed distance thresheded images")
#         final_seg_overall = convert_matrix_to_multipage_tiff(final_seg_overall)
#         imsave(s_path + 'overall_output_2nd_iteration.tif', np.asarray(final_seg_overall * 255, dtype=np.uint8))
#         #final_seg_overall = convert_matrix_to_multipage_tiff(final_seg_overall)
#         #imsave(s_path + 'overall_output.tif', final_seg_overall)
        
 






     # (1) Add take away the mean value elim
     # (2) Always sort from largest to smallest seeds and start with largest, (also put in min threshold?)



        """ Post-processing: """
        combine_first_and_second_iter = []        
        for trace_idx in range(len(each_individual_fiber_trace_coords)):
             trace = each_individual_fiber_trace_coords[trace_idx]
             
             for second_trace_idx in range(len(second_iter_branch_number)):
                  branch_num = second_iter_branch_number[second_trace_idx]
                  branch_coords = second_iter_each_individual_fiber_trace_coords[second_trace_idx]
                  
                  if trace_idx != branch_num:
                       continue;
                  elif branch_num > trace_idx:
                       break;
                  else:   # add to branch if the branch_num == trace_idx
                       trace = np.append(trace, branch_coords, axis=0)
                       trace = np.unique(trace, axis=0)
                       
                       print('appended')
                       
             combine_first_and_second_iter.append(trace)           
             
        """ Create array from list of each individual trace 
            Maybe should sort by size??? so plot the largest sheaths first, and then if the small ones coloc, then ignore them?
        """
        sorted_list = sorted(combine_first_and_second_iter, key=len, reverse=True) 
        all_traces_rainbow = np.zeros(np.shape(input_im))
        for trace in sorted_list:
             rand_val = randint(1,10)
             for idx_trace in range(len(trace)):
                  # only add to trace rainbow if NOT equal to zero in that spot
                  if all_traces_rainbow[trace[idx_trace][0], trace[idx_trace][1], trace[idx_trace][2]] == 0:
                         all_traces_rainbow[trace[idx_trace][0], trace[idx_trace][1], trace[idx_trace][2]] = rand_val          
        
       
        plot_save_max_project(fig_num=100, im=all_traces_rainbow, max_proj_axis=-1, title='trace rainbow', 
                          name=s_path + 'overall_TRACE_RAINBOW.png', pause_time=0.001)
        
        """ Also save pickle files"""
        save_pkl(all_traces_rainbow, s_path, 'FINAL_overall_TRACE_RAINBOW.pkl')
        save_pkl(each_individual_fiber_trace_coords, s_path, 'FINAL_1st_iter_individual_trace_coords.pkl')
        save_pkl(second_iter_branch_number, s_path, 'FINAL_2nd_iter_branch_nums.pkl')
        save_pkl(second_iter_each_individual_fiber_trace_coords, s_path, 'FINAL_2nd_iter_individual_trace_coords.pkl')


                               
        """ Save as outputs """
        print("Saving trace rainbow")
        save_rainbow = convert_matrix_to_multipage_tiff(all_traces_rainbow)
        imsave(s_path + 'overall_TRACE_RAINBOW.tif', np.asarray(save_rainbow * 255, dtype=np.uint8))

        # In imageJ ==> select colorscale glaseby_on_dark ==> convert to RGB ==> then make 3D projection

        
        
        
        
        
        
        
        
        
    