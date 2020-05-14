# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 20:19:27 2019

@author: Tiger


Allows person to select cell nuclei for every image

3 things left to fix:
    
    (1) How to pause for user to select points
    (2) Why can't see the output figure for the last figure?
    (3) Set it up so can identify all the coords first, and THEN run the analysis for ALL the coords

"""

from matplotlib import *
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from natsort import natsort_keygen, ns
from skimage import measure
import pickle as pickle
import os

from random import randint

from plot_functions import *
from data_functions import *
from data_functions_3D import *
#from post_process_functions import *
from UNet import *
from UNet_3D import *
import glob, os
#from tifffile import imsave


""" Interactive click event to select seed point """
def onclick(event):
   global ix, iy
   ix, iy = event.xdata, event.ydata
   print('x = %d, y = %d'%(ix, iy))
    
   global coords
   coords.append((ix, iy))
    
   if len(coords) == 2:
       fig.canvas.mpl_disconnect(cid)
    
   return coords
    
""" pausing click??? """
def onclick_unpause(event):
   global pause
   pause = False
    
    
#s_path = './Checkpoints_Normal_3D/1) Check_SMALL_patches_no_alter_dilate_filter_555/'
#input_path = '../20190713_FINAL_raw_images_with_all_classes/patches_skel/'
input_path = 'C:/Users/Tiger/Documents/GitHub/Bergles Lab Projects/Oligo morphology/Raw data/'

input_path = './Raw data/'
input_path = './singleOLdata/'

sav_dir = input_path
               
""" Load filenames from zip """
images = glob.glob(os.path.join(input_path,'*.tif'))
examples = [dict(input=i,truth=i.replace('_input_','_truth_')) for i in images]
#examples = [dict(input=i,truth=i.replace('input.tif','_pos_truth.tif')) for i in images]  # if want to load only positive examples

for i in range(len(examples)):
    input_name = examples[i]['input']
    #print(input_name)
    #input_im = np.asarray(Image.open(input_name), dtype=np.float32)
    input_im = open_image_sequence_to_3D(input_name, input_size='default', depth='default')
    
    depth_last = np.zeros([np.shape(input_im)[1], np.shape(input_im)[2], np.shape(input_im)[0]])
    for slice_idx in range(len(input_im)):
        depth_last[:, :, slice_idx] = input_im[slice_idx,  :, :]        
        
    input_im = []   # garbage collection
    
    """ Plotting as interactive scroller """
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, depth_last)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

    coords = []
    
    """ Pause event to give time to add points to image """
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    print("Select ONE seed point, can scroll through slices with mouse wheel")
    pause = True
    while pause:
        plt.pause(2)
        cid = fig.canvas.mpl_connect('button_press_event', onclick_unpause)

    fig.canvas.mpl_disconnect(cid)   # DISCONNECTS CLICKING EVENT
    
     
    """ ^^^ for above, should also get z-axis positon, and ONLY keeps FIRST coord """
    z_position = tracker.ind
    
    overall_coord = [int(coords[0][1]), int(coords[0][0]), z_position]
    
    plt.close(1)
    
    """ Binarize and then use distant transform to locate cell bodies """
    from scipy import ndimage as ndi
    from skimage.morphology import watershed
    from skimage.feature import peak_local_max
    from skimage.filters import threshold_otsu
    thresh = threshold_otsu(depth_last)
    binary = depth_last > thresh
       
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
    
    """ Plotting as interactive scroller """
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, binary)
    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()
    
    """ Then colocalize with coords to get mask """
    labelled = measure.label(binary)
    cc_overlap = measure.regionprops(labelled)
    
    match = 0
    matching_blob_coords = []
    for cc in cc_overlap:
        coords_blob = cc['coords']
        print(len(coords_blob))
        for idx in coords_blob:
            #print(idx)

            if (idx == np.asarray(overall_coord)).all():
                match = 1
        if match:
            match = 0
            matching_blob_coords = coords_blob
            print('matched')            
            
            #break;
            
    only_colocalized_mask = np.zeros(np.shape(depth_last))
    for idx in range(len(matching_blob_coords)):
        only_colocalized_mask[matching_blob_coords[idx][0], matching_blob_coords[idx][1], matching_blob_coords[idx][2]] = 255
    
    
    #only_colocalized_mask = np.asarray(only_colocalized_mask, np.dtype('uint8'))
    
    """ Plotting as interactive scroller """
#    fig, ax = plt.subplots(1, 1)
#    tracker = IndexTracker(ax, only_colocalized_mask)
#    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
#    plt.show()
#    
    
    binary = []; labelled = []; dist1 = [];
    
    
    """ Save mask as 2nd channel in input image??? or just as separate image for now """    
    depth_last = []
    print("Saving image of mask")
    output = convert_matrix_to_multipage_tiff(only_colocalized_mask)
    split_name = input_name.split('.')[0]
    imsave(split_name + '_input_cellMASK.tif', output)
 
    



