# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:55:06 2019

@author: tiger
"""

import numpy as np
from data_functions_CLEANED import *
from data_functions_3D import *
import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu

import skimage.morphology
from skimage.exposure import equalize_adapthist
from matlab_crop_function import *
import cv2 as cv2
from skimage.filters import threshold_local


import torch
import scipy     

""" (1) Load input and parse into seeds """
def load_input_as_seeds(examples, im_num, pregenerated, s_path='./'):
     """ Load input image """
     input_name = examples[im_num]['input']
     input_im = open_image_sequence_to_3D(input_name, width_max='default', height_max='default', depth='default')
     
     """ also detect shape of input_im and adapt accordingly """
     width_tmp = np.shape(input_im)[1]
     height_tmp = np.shape(input_im)[2]
     depth_tmp = np.shape(input_im)[0]
     
     input_im = convert_multitiff_to_matrix(input_im)
 
     """ Decide whether to use auto seeds or pregenerated seeds"""
     if pregenerated:
          
          seed_name = examples[im_num]['seeds']
          all_seeds = open_image_sequence_to_3D(seed_name, width_max='default', height_max='default', depth='default')             
               
          labelled=np.uint8(all_seeds)
          labelled = np.moveaxis(labelled, 0, -1)
          overall_coord = []

          
     else:        
          """ Plotting as interactive scroller """
          only_colocalized_mask, overall_coord = GUI_cell_selector(input_im, crop_size=100, z_size=30,
                                                                    height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_tmp)
          """ or auto-create seeds """
          all_seeds, cropped_seed, binary = create_auto_seeds(input_im, only_colocalized_mask, overall_coord, 
                                        crop_size=100, z_size=30, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_tmp)
          
          plot_save_max_project(fig_num=88, im=cropped_seed, max_proj_axis=-1, title='all_seeds', 
                                          name=s_path + 'all_seeds.png', pause_time=0.001)
          plot_save_max_project(fig_num=89, im=binary, max_proj_axis=-1, title='all_seeds_binary', 
                                          name=s_path + 'all_seeds_binary.png', pause_time=0.001)
          labelled = measure.label(all_seeds)
 

     """ Now start looping through each seed point to crop out the image """
     """ Make a list of centroids to keep track of all the new locations to visit """
     cc_seeds = measure.regionprops(labelled)
     list_seeds = []
     for cc in cc_seeds:
          list_seeds.append(cc['coords'])
     sorted_list = sorted(list_seeds, key=len, reverse=True)  
 
     return sorted_list, input_im, width_tmp, height_tmp, depth_tmp, overall_coord



""" If resized, check to make sure no straggling non-attached objects """
def check_resized(im, depth, width_max, height_max):
   middle_idx = np.zeros([depth, width_max, height_max])
   
   # make a square to colocalize with later
   square_size = 4
   middle_idx[int(depth/2) - square_size: int(depth/2) + square_size, int(width_max/2) - square_size: int(width_max/2) + square_size, int(height_max/2) - square_size: int(height_max/2) + square_size] = 1

   for channel_idx in range(len(im[0, 0, 0, :])):
        ch_orig = np.copy(im[:, :, :, channel_idx])
        coloc = middle_idx + ch_orig
        if channel_idx == 2: # if there is paranodes channel, also add fibers in along with it to connect
             elim_outside = np.copy(middle_idx)
             square_size = int(width_max/2 - width_max/2 * 0.1)
             elim_outside = np.ones([depth, width_max, height_max])
             elim_outside[int(depth/2) - int(depth/2 - depth/2*0.1): int(depth/2) + int(depth/2 - depth/2*0.1), int(width_max/2) - square_size: int(width_max/2) + square_size, int(height_max/2) - square_size: int(height_max/2) + square_size] = 0
             #elim_outside[elim_outside == 1] = -1
             #elim_outside[elim_outside == 0] = 1
             #elim_outside[elim_outside == -1] = 0
             
             
             check_outside = elim_outside + coloc
             if 2 in np.unique(check_outside):
                  ch_orig = np.zeros(np.shape(middle_idx))  # SKIPS if touches the external boundary
                  print('skipped')
                  #print(np.unique(check_outside))
             else:
                  coloc = coloc + im[:, :, :, 1]
                  print('passed')

        bw_coloc = np.copy(coloc)
        bw_coloc[bw_coloc > 0] = 1
                         
        only_coloc = find_overlap_by_max_intensity(bw=bw_coloc, intensity_map=coloc)
        ch_orig[only_coloc == 0] = 0
        
        im[:, :, :, channel_idx] = ch_orig  
        
   return im


""" subtract 2 arrays from one another and setting sub zeros to 0 """
def subtract_im_no_sub_zero(arr1, arr2):
   deleted = arr1 - arr2
   deleted[deleted < 0] = 0
   return deleted


""" Given coords of shape x, y, z in a cropped image, scales back to size in full size image """
def scale_coords_of_crop_to_full(coords, box_x_min, box_y_min, box_z_min):
        coords[0] = int(coords[0]) + box_x_min   # SCALING the ep_center
        coords[1] = int(coords[1]) + box_y_min
        coords[2] = int(coords[2]) + box_z_min
        scaled = coords
        return scaled  

""" check if an array is truly empty """
def check_empty(array, already_visited, list_seed_centers, reason='deleted'):
    skip = 0
    if len(np.unique(array)) == 0:
         print(reason)
         already_visited.append(list_seed_centers[0])
         del list_seed_centers[0]
         skip = 1
         return skip, already_visited, list_seed_centers
    else:
         return skip, already_visited, list_seed_centers

""" uses an intensity map (that indicates where overalp occured), identifies segments that overlapped """
def find_overlap_by_max_intensity(bw, intensity_map, min_size_obj=0):
   labelled = measure.label(bw)
   cc_coloc = measure.regionprops(labelled, intensity_image=intensity_map)

   only_coloc = np.zeros(np.shape(intensity_map))
   for end_point in cc_coloc:
        max_val = end_point['max_intensity']
        coords = end_point['coords']
        if max_val > 1 and len(coords) > min_size_obj:
             for c_idx in range(len(coords)):
                  only_coloc[coords[c_idx,0], coords[c_idx,1], coords[c_idx,2] ] = 1
       
   return only_coloc

""" Creates maximum projection and saves it"""
def plot_save_max_project(fig_num, im, max_proj_axis=-1, title='default', name='default', pause_time=0.001):                    
     ma = np.amax(im, axis=max_proj_axis)
     plt.figure(fig_num); plt.imshow(ma); plt.title(title); plt.pause(pause_time);
     plt.savefig(name)

""" dilates image by a spherical ball of size radius """
def erode_by_ball_to_binary(input_im, radius):
     ball_obj = skimage.morphology.ball(radius=radius)
     input_im = skimage.morphology.erosion(input_im, selem=ball_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" dilates image by a spherical ball of size radius """
def dilate_by_ball_to_binary(input_im, radius):
     ball_obj = skimage.morphology.ball(radius=radius)
     input_im = skimage.morphology.dilation(input_im, selem=ball_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" dilates image by a spherical ball of size radius """
def dilate_by_disk_to_binary(input_im, radius):
     ball_obj = skimage.morphology.disk(radius=radius)
     input_im = skimage.morphology.dilation(input_im, selem=ball_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" dilates image by a cube of size width """
def dilate_by_cube_to_binary(input_im, width):
     cube_obj = skimage.morphology.cube(width=width)
     input_im = skimage.morphology.dilation(input_im, selem=cube_obj)  
     input_im[input_im > 0] = 1
     return input_im

""" erodes image by a cube of size width """
def erode_by_cube_to_binary(input_im, width):
     cube_obj = skimage.morphology.cube(width=width)
     input_im = skimage.morphology.erosion(input_im, selem=cube_obj)  
     input_im[input_im > 0] = 1
     return input_im


""" Applies CLAHE to a 2D image """           
def apply_clahe_by_slice(crop, depth):
     clahe_adjusted_crop = np.zeros(np.shape(crop))
     for slice_idx in range(depth):
          slice_crop = np.asarray(crop[:, :, slice_idx], dtype=np.uint8)
          adjusted = equalize_adapthist(slice_crop, kernel_size=None, clip_limit=0.01, nbins=256)
          clahe_adjusted_crop[:, :, slice_idx] = adjusted
                 
     crop = clahe_adjusted_crop * 255
     return crop


""" Or load as normal truth to get seeds """
def load_truth_seeds(input_counter, only_colocalized_mask, i, input_tmp_size, depth_tmp_size, num_truth_class, load_class):
        """ Load truth image, either BINARY or MULTICLASS """
        truth_name = examples[input_counter[i]]['truth']   
        truth_im, weighted_labels = load_class_truth_3D(truth_name, num_truth_class, input_size=input_tmp_size, depth=depth_tmp_size, spatial_weight_bool=0)
        cytosol_truth = truth_im[:, :, :, load_class]

        cytosol_reordered = convert_multitiff_to_matrix(cytosol_truth)
        
        """ Now find bounding box around center point and expand outwards to find things ATTACHED to middle body """
        """ create seeds by subtracting out large - small cell body masks """
        dilated_image_large = dilate_by_cube_to_binary(only_colocalized_mask, width=40)
        
        dilated_image_small = dilate_by_cube_to_binary(only_colocalized_mask, width=8)

        """ subtract large - small to create seeds """
        mask = dilated_image_large - dilated_image_small
        all_seeds = np.copy(cytosol_reordered)
        all_seeds[mask == 0] = 0
        
        return cytosol_reordered
   
     
""" automatically create seeds for analysis """
def create_auto_seeds(input_im, only_colocalized_mask, overall_coord, crop_size, z_size, height_tmp, width_tmp, depth_tmp):

        """ Don't use user selected point b/c may vary each time. Use the centroid of the dilated object """
        labelled = measure.label(only_colocalized_mask)
        cc_coloc = measure.regionprops(labelled)

        overall_coord = np.asarray(cc_coloc[0]['centroid']);
        overall_coord[0] = int(overall_coord[0]);
        overall_coord[1] = int(overall_coord[1]);
        overall_coord[2] = int(overall_coord[2]);
        
        """ MAYBE DILATE AS CROPS INSTEAD """    
        x = int(overall_coord[0])
        y = int(overall_coord[1])
        z = int(overall_coord[2])
        crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(input_im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)
        only_colocalized_mask_crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(only_colocalized_mask, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)
     
        """ Instead of loading truth image, just generate from binary image """
        
        """ Added adaptive cropping instead! on a per slice basis 
                  ***note: variable "C" is very finnicky... and determines a lot of threshold behavior
        """        
        binary = np.zeros(np.shape(crop));
        for crop_idx in range(len(crop[0, 0, :])):
             cur_crop = crop[:, :, crop_idx]

             thresh_adapt = cv2.adaptiveThreshold(np.asarray(cur_crop, dtype=np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                  cv2.THRESH_BINARY, blockSize=25,C=-30) 
             
             #thresh_adapt = threshold_local(cur_crop, block_size=35, method='gaussian',
                                             #offset=0, mode='reflect', param=None, cval=0)
             
             #np.unique(thresh_adapt)
             #cur_crop = crop[:, :, crop_idx] > thresh_adapt
             binary[:, :, crop_idx] = thresh_adapt
            
        
        #plt.figure(); plt.imshow(thresh_adapt)
        
        
        #thresh = threshold_otsu(crop)
        #binary_otsu = crop > thresh
        cytosol_reordered = binary
        cytosol_reordered[cytosol_reordered > 0] = 1
        cytosol_reordered = skeletonize_3d(cytosol_reordered)
        cytosol_reordered[cytosol_reordered > 0] = 1
                                
        """ Now find bounding box around center point and expand outwards to find things ATTACHED to middle body """
        """ create seeds by subtracting out large - small cell body masks """
        #dilated_image_large = dilate_by_ball_to_binary(only_colocalized_mask_crop, radius=20)
        
        dilated_image_small = dilate_by_ball_to_binary(only_colocalized_mask_crop, radius=10)

        """ subtract large - small to create seeds """
        #mask = dilated_image_large - dilated_image_small
        mask = dilated_image_small
        all_seeds = np.copy(cytosol_reordered)
        #all_seeds[mask == 0] = 0
        all_seeds[mask == 1] = 0

        
        """ only keep things that are connected to the center cell body directly! """
        dilated_image_small = dilate_by_ball_to_binary(dilated_image_small, radius=2)

        overlaped = all_seeds + dilated_image_small
        cropped_seed = find_overlap_by_max_intensity(bw=all_seeds, intensity_map=overlaped, min_size_obj=10)        

        """ OPTIONAL:  *** can take out if makes seeds too sparse or loses too many seeds
             
             Delete all branch points to make more clean seeds """
        degrees, coordinates = bw_skel_and_analyze(cropped_seed)
        branch_points = np.copy(degrees); branch_points[branch_points != 3] = 0
        cropped_seed[branch_points > 0] = 0
        
        """ restore crop """
        all_seeds = np.zeros(np.shape(input_im))
        all_seeds[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] = cropped_seed
        
        return all_seeds, cropped_seed, binary


""" run UNet inference """
def UNet_inference(crop, crop_seed, batch_x, batch_y, weights, mean_arr, std_arr, x_3D, y_3D_, weight_matrix_3D, softMaxed, training, num_classes):
   """ Combine seed mask with input im"""
   input_im_and_seeds = np.zeros(np.shape(crop) + (2, ))
   input_im_and_seeds[:, :, :, 0] = crop
   input_im_and_seeds[:, :, :, 1] = crop_seed  
   
   depth_first = np.zeros([np.shape(input_im_and_seeds)[2], np.shape(input_im_and_seeds)[0], np.shape(input_im_and_seeds)[1], np.shape(input_im_and_seeds)[3]])
   for slice_idx in range(len(input_im_and_seeds[0, 0, :, 0])):
        depth_first[slice_idx, :, :, :] = input_im_and_seeds[:, :, slice_idx,  :] 
   input_im_and_seeds = depth_first

   input_im_save = np.copy(input_im_and_seeds)
   input_im_and_seeds = normalize_im(input_im_and_seeds, mean_arr, std_arr) 

   batch_x.append(input_im_and_seeds)
   batch_y.append(np.zeros([np.shape(crop)[2], np.shape(crop)[0], np.shape(crop)[1], num_classes]))
   weights.append(np.zeros([np.shape(crop)[2], np.shape(crop)[0], np.shape(crop)[1], num_classes]))
   
   feed_dict_TEST = {x_3D:batch_x, y_3D_:batch_y, training:1, weight_matrix_3D:weights}
   feed_dict = feed_dict_TEST
   output_test = softMaxed.eval(feed_dict=feed_dict)
   output_test_save = output_test[0]  # takes only 1st of batch
   seg_test = np.argmax(output_test, axis = -1)[0]   # takes only 1st of batch          
       
   depth_last_tmp = np.zeros([np.shape(seg_test)[1], np.shape(seg_test)[2], np.shape(seg_test)[0]])
   for slice_idx in range(len(seg_test)):
        depth_last_tmp[:, :, slice_idx] = seg_test[slice_idx,  :, :] 
        
        
   output_test_depth_last = np.zeros([np.shape(output_test_save)[1], np.shape(output_test_save)[2], np.shape(output_test_save)[0], np.shape(output_test_save)[3]])
   for slice_idx in range(len(output_test_save[:, 0, 0, 0])):
        output_test_depth_last[:, :, slice_idx, :] = output_test_save[slice_idx, :, :,  :]
   output_softMax = output_test_depth_last
        
   return depth_last_tmp, batch_x, batch_y, weights, input_im_save, output_softMax



""" run UNet inference """
def UNet_inference_PYTORCH(unet, crop, crop_seed, mean_arr, std_arr, device=None):
   """ Combine seed mask with input im"""
   input_im_and_seeds = np.zeros(np.shape(crop) + (2, ))
   input_im_and_seeds[:, :, :, 0] = crop
   input_im_and_seeds[:, :, :, 1] = crop_seed  


   """ Rearrange channels """
   input_im_and_seeds = np.moveaxis(input_im_and_seeds, -1, 0)
   input_im_and_seeds = np.moveaxis(input_im_and_seeds, -1, 1)
   
   """ Normalization """
   input_im_and_seeds = (input_im_and_seeds - mean_arr)/std_arr
           
   inputs = torch.tensor(input_im_and_seeds, dtype = torch.float, device=device, requires_grad=False)
   """ Expand dims """
   inputs = inputs.unsqueeze(0) 
  
   
   """ forward + backward + optimize """
   output = unet(inputs)
   output = output.data.cpu().numpy()


   
   depth_last_tmp = np.moveaxis(output[0], 1, -1)
   depth_last_tmp = np.moveaxis(depth_last_tmp, 0, -1)     
           
   output = np.argmax(depth_last_tmp, axis = -1)   # takes only 1st of batch          
       
 
   return output



""" Prompts user with scrolling tile to select cell of interest """
def GUI_cell_selector(depth_last, crop_size, z_size,  height_tmp, width_tmp, depth_tmp):

        """ Interactive click event to select seed point """
        def onclick(event):
             global ix, iy
             ix, iy = event.xdata, event.ydata
             print('x = %d, y = %d'%(ix, iy))
              
             global coords
             coords.append((ix, iy))
              
             if len(coords) == 2:
                 fig.canvas.mpl_disconnect(cid)
              
             #return coords
              
        """ pausing click??? """
        def onclick_unpause(event):
             global pause
             pause = False
             
        fig, ax = plt.subplots(1, 1)
        tracker = IndexTracker(ax, depth_last)
        fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        plt.show()
     
        global coords    # GLOBAL VARIABLES INSIDE FUNCTIONS NEED TO BE DECLARED IN EVERY FUNCTION SO THE SCOPE WORKS
        coords = []
        
        """ Pause event to give time to add points to image """
        cid = fig.canvas.mpl_connect('button_press_event', onclick)
        print("Select ONE seed point, can scroll through slices with mouse wheel")
        #print(input_name)
        global pause
        pause = True
        while pause:
            plt.pause(1)
            cid = fig.canvas.mpl_connect('button_press_event', onclick_unpause)
     
        fig.canvas.mpl_disconnect(cid)   # DISCONNECTS CLICKING EVENT
         
        """ ^^^ for above, should also get z-axis positon, and ONLY keeps FIRST coord """
        z_position = tracker.ind
        overall_coord = [int(coords[0][1]), int(coords[0][0]), z_position]
        plt.close(1)
         

        """ Faster is to crop """
        x = overall_coord[0]
        y = overall_coord[1]
        z = overall_coord[2]
        depth_last_crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(depth_last, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)
        #input_im = crop   

        """ Binarize and then use distant transform to locate cell bodies """
        thresh = threshold_otsu(depth_last_crop)
        binary = depth_last_crop > thresh
           
        """ Maybe faster/more efficient way is to just do distance transform on each 2D slice in stack"""
        #dist1 = scipy.ndimage.distance_transform_edt(binary, sampling=[1,1,1])
             
        print('Distance transform')
        # DO SLICE BY SLICE
        dist1 = np.zeros(np.shape(binary))
        for slice_idx in range(len(binary[0, 0, :])):
            tmp = scipy.ndimage.distance_transform_edt(binary[:, :, slice_idx], sampling=1)
            dist1[:, :, slice_idx] = tmp
             
        print('Distance transform completed')
         
        # Then threshold based on distance transform
        thresh = 10 # pixel distance
        binary = dist1 > thresh
                 
        
        """ restore crop """
        binary_restore = np.zeros(np.shape(depth_last))
        binary_restore[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] = binary
        
        
        """ Then colocalize with coords to get mask """
        labelled = measure.label(binary_restore)
        cc_overlap = measure.regionprops(labelled)
         
        match = 0
        matching_blob_coords = []
        for cc in cc_overlap:
             coords_blob = cc['coords']
             for idx in coords_blob:     
                 if (idx == np.asarray(overall_coord)).all():
                     match = 1
             if match:
                 match = 0
                 matching_blob_coords = coords_blob
                 print('matched')            
                 
        only_colocalized_mask = np.zeros(np.shape(depth_last))
        for idx in range(len(matching_blob_coords)):
             only_colocalized_mask[matching_blob_coords[idx][0], matching_blob_coords[idx][1], matching_blob_coords[idx][2]] = 255
         
        return only_colocalized_mask, overall_coord