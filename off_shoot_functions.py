# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 11:55:06 2019

@author: tiger
"""

import numpy as np
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.plot_functions_CLEANED import *
from tree_functions import *

import matplotlib.pyplot as plt

from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_triangle, try_all_threshold, threshold_local

import skimage.morphology
from skimage.exposure import equalize_adapthist
from matlab_crop_function import *
import cv2 as cv2
from skimage.filters import threshold_local
from skimage.feature import hessian_matrix, hessian_matrix_eigvals
from skimage.filters import gaussian

import torch
import scipy     



""" (1) Load input and parse into seeds """
def load_input_as_seeds(examples, im_num, pregenerated, s_path='./', seed_crop_size=100, seed_z_size=80):
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
              
          
          all_seeds_no_50 = np.copy(all_seeds)
          all_seeds_no_50[all_seeds_no_50 == 50] = 0
          labelled=np.uint8(all_seeds_no_50)
          labelled = np.moveaxis(labelled, 0, -1)
          overall_coord = []

          all_seeds = convert_multitiff_to_matrix(all_seeds)
          
     else:        
          """ Plotting as interactive scroller """
          only_colocalized_mask, overall_coord = GUI_cell_selector(input_im, crop_size=seed_crop_size, z_size=seed_z_size,
                                                                    height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_tmp, thresh=1)
          """ or auto-create seeds """
          all_seeds, cropped_seed, binary, all_seeds_no_50 = create_auto_seeds(input_im, only_colocalized_mask, overall_coord, 
                                        seed_crop_size=seed_crop_size, seed_z_size=seed_z_size, height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_tmp)
          
          plot_save_max_project(fig_num=88, im=cropped_seed, max_proj_axis=-1, title='all_seeds', 
                                          name=s_path + 'all_seeds.png', pause_time=0.001)
          plot_save_max_project(fig_num=89, im=binary, max_proj_axis=-1, title='all_seeds_binary', 
                                          name=s_path + 'all_seeds_binary.png', pause_time=0.001)
          labelled = measure.label(all_seeds_no_50)
 

     """ Now start looping through each seed point to crop out the image """
     """ Make a list of centroids to keep track of all the new locations to visit """
     cc_seeds = measure.regionprops(labelled)
     list_seeds = []
     for cc in cc_seeds:
          list_seeds.append(cc['coords'])
     sorted_list = sorted(list_seeds, key=len, reverse=True)  
 
     return sorted_list, input_im, width_tmp, height_tmp, depth_tmp, overall_coord, all_seeds, all_seeds_no_50


                    

""" If resized, check to make sure no straggling non-attached objects """
def check_resized(im, depth, width_max, height_max):
  
   im = convert_matrix_to_multipage_tiff(im)
   im = np.expand_dims(im, axis=-1)    
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

        im = im[:, :, :, 0]
        im = convert_multitiff_to_matrix(im)

        
   return im


""" subtract 2 arrays from one another and setting sub zeros to 0 """
def subtract_im_no_sub_zero(arr1, arr2):
   deleted = arr1 - arr2
   deleted[deleted < 0] = 0
   return deleted




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
 
    
 
    
""" Extract ridges from 3D image """        
def ridge_filter_3D(im, sigma=3):
        #crop = gaussian(crop, sigma=1)
        H_elems = hessian_matrix(im, sigma=sigma)
        #i1, i2 = hessian_matrix_eigvals(hxx, hxy, hyy)
        
        eigs = hessian_matrix_eigvals(H_elems)
    
        LambdaAbs1=abs(eigs[0]);
        LambdaAbs2=abs(eigs[1]);
        LambdaAbs3=abs(eigs[2]);     
        
        return LambdaAbs1, LambdaAbs2, LambdaAbs3
     
     
 
    
 
     
""" automatically create seeds for analysis """
def create_auto_seeds(input_im, only_colocalized_mask, overall_coord, seed_crop_size, seed_z_size, height_tmp, width_tmp, depth_tmp):

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
        crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(input_im, y, x, z, seed_crop_size, seed_z_size, height_tmp, width_tmp, depth_tmp)
        only_colocalized_mask_crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(only_colocalized_mask, y, x, z, seed_crop_size, seed_z_size, height_tmp, width_tmp, depth_tmp)
     
        
        """ Subtract out center to make cleaner for thresholding """
        dilated_image_small = dilate_by_ball_to_binary(only_colocalized_mask_crop, radius=5)
        dilated_image_small = dilate_by_ball_to_binary(dilated_image_small, radius=5)
        dilated_image_small = dilate_by_ball_to_binary(dilated_image_small, radius=5)
        
        
        crop[dilated_image_small > 0] = 0
        
        from skimage.filters import frangi, gaussian, meijering, sato, hessian
        fran = frangi(crop,  sigmas=range(1, 6, 1), black_ridges=False, alpha=0.5, beta=0.5, gamma=15)
        
        
        #crop = gaussian(crop, sigma=2)
        #fran = sato(crop,  sigmas=range(3, 6, 3), black_ridges=False)
        
        #fran = meijering(crop, black_ridges=False)
        
        #fran = hessian(crop, black_ridges=False)
         
     
     
        """ smooth first??? """
        # def subtract_background(image, radius=5, light_bg=False):
        #         from skimage.morphology import white_tophat, black_tophat, ball, opening
        #         str_el = ball(radius=radius) #you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time        
        #         return white_tophat(image, str_el)
        
        
        fran = gaussian(fran, sigma=1)
        
        
        """ Use hessian??? """
        #LambdaAbs1, LambdaAbs2, LambdaAbs3 = ridge_filter_3D(im=crop, sigma=3)

        #LambdaAbs1, LambdaAbs2, LambdaAbs3 = ridge_filter_3D(im=LambdaAbs2, sigma=2)    ### REMOVE FOR NEURON
        #LambdaAbs1, LambdaAbs2, LambdaAbs3 = ridge_filter_3D(im=LambdaAbs2, sigma=1)

        #crop = LambdaAbs2;   # maybe this should be lambda 3???

        thresh = threshold_otsu(fran) 
        thresh = thresh - thresh * 0.75
        
        #thresh = threshold_triangle(fran)
        
        #thresh = 0.12   ### FOR NEURON SEGMENTATION
        binary = fran > thresh
        
        # plot_max(binary, ax=-1)
        # plot_max(fran, ax=-1)
        
        
     
        
        
        
        
        
        # from skimage.exposure import equalize_adapthist
        
        # from skimage import exposure
        
        # crop_rescale  = exposure.rescale_intensity(crop, in_range='image', out_range=(0, 1))
        
        # adapt = equalize_adapthist(crop_rescale)
        
 
        
        # thresh = threshold_otsu(crop_rescale)
        
        # thresh = threshold_triangle(crop_rescale)
        
        # thresh = 0.12   ### FOR NEURON SEGMENTATION
        # binary = crop_rescale > thresh
        
        # str_el = ball(radius=2) #you can also use 'ball' here to get a slightly smoother result at the cost of increased computing time        
        # opened = opening(crop, selem=str_el, out=None)
 
    
        #plot_max(binary, ax=-1)
        
        # fig, ax = plt.subplots(1, 1)
        # tracker = IndexTracker(ax, binary_otsu)
        # fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
        # plt.show()        
                
        """ Instead of loading truth image, just generate from binary image """
        
        """ Added adaptive cropping instead! on a per slice basis 
                  ***note: variable "C" is very finnicky... and determines a lot of threshold behavior
        """        
        # binary = np.zeros(np.shape(crop));
        # for crop_idx in range(len(crop[0, 0, :])):
        #       cur_crop = crop[:, :, crop_idx]

        #       #thresh_adapt = cv2.adaptiveThreshold(np.asarray(cur_crop, dtype=np.uint8), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        #       #                                    cv2.THRESH_BINARY, blockSize=25,C=-30) 
             
        #       thresh_adapt = threshold_local(cur_crop, block_size=35, method='gaussian',
        #                                       offset=0, mode='reflect', param=None, cval=0)
             
                
              
              
        #       #np.unique(thresh_adapt)
        #       #cur_crop = crop[:, :, crop_idx] > thresh_adapt
        #       #binary[:, :, crop_idx] = cur_crop
              
              
        #       binary[:, :, crop_idx] = thresh_adapt
            
        
        #plt.figure(); plt.imshow(thresh_adapt)
        
        
        # thresh = threshold_otsu(binary)
        # binary_otsu = binary > thresh
        
        
        
        
        cytosol_reordered = binary
        cytosol_reordered[cytosol_reordered > 0] = 1
        cytosol_reordered = skeletonize_3d(cytosol_reordered)
        cytosol_reordered[cytosol_reordered > 0] = 1
                                
        """ Now find bounding box around center point and expand outwards to find things ATTACHED to middle body """
        """ create seeds by subtracting out large - small cell body masks """
        #dilated_image_large = dilate_by_ball_to_binary(only_colocalized_mask_crop, radius=20)
        
        #dilated_image_small = dilate_by_ball_to_binary(only_colocalized_mask_crop, radius=10)
        dilated_image_small = dilate_by_ball_to_binary(dilated_image_small, radius=5)

        """ subtract dilated nucleus from image """
        #mask = dilated_image_large - dilated_image_small
        mask = dilated_image_small
        all_seeds = np.copy(cytosol_reordered)
        #all_seeds[mask == 0] = 0
        all_seeds[mask == 1] = 0

        
        """ only keep things that are connected to the center cell body directly! 
        
                ***cleans up by size later too
        """
        dilated_image_expanded = dilate_by_ball_to_binary(dilated_image_small, radius=2)

        overlaped = all_seeds + dilated_image_expanded
        cropped_seed = find_overlap_by_max_intensity(bw=all_seeds, intensity_map=overlaped, min_size_obj=10)        

        """ OPTIONAL:  *** can take out if makes seeds too sparse or loses too many seeds
             
             Delete all branch points to make more clean seeds """
        # degrees, coordinates = bw_skel_and_analyze(cropped_seed)
        # branch_points = np.copy(degrees); branch_points[branch_points != 3] = 0
        # cropped_seed[branch_points > 0] = 0
        
        """ restore crop """
        all_seeds_no_50 = np.zeros(np.shape(input_im))
        all_seeds_no_50[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] = cropped_seed
        
        
        
        """ set cell as root coords for later """
        all_seeds = np.copy(all_seeds_no_50)
        cell_body = dilated_image_small
        cropped_seed[overlaped > 1] = 2
        all_seeds[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] = cropped_seed
        
        
        
        
        return all_seeds, cropped_seed, binary, all_seeds_no_50


""" run UNet inference """
def UNet_inference_PYTORCH(unet, crop, crop_seed, mean_arr, std_arr, device=None, deep_supervision=False):
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
   
   
   if deep_supervision:
       output = output[-1]
   
   output = output.data.cpu().numpy()


   
   depth_last_tmp = np.moveaxis(output[0], 1, -1)
   depth_last_tmp = np.moveaxis(depth_last_tmp, 0, -1)     
           
   output = np.argmax(depth_last_tmp, axis = -1)   # takes only 1st of batch          
       
 
   return output



""" Prompts user with scrolling tile to select cell of interest """
def GUI_cell_selector(depth_last, crop_size, z_size,  height_tmp, width_tmp, depth_tmp, thresh=0):

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
         
        only_colocalized_mask = []
        if thresh:

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