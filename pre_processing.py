"""
Created on Tue Jan  2 12:29:40 2018

@author: Tiger
"""

#import tensorflow as tf
from matplotlib import *
import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from skimage import measure

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from UNet import *

import cv2
from scipy import ndimage as ndi
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu


""" Image Adjust """
def im_adjust(red, sz=20):
    #Subtract background:
    if sz > 0:   # DO NOTHING IF ROLLING BALL == 0 size
        blur = cv2.GaussianBlur(red,(5,5), 1)
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz));   # get disk structuring element
        background = cv2.morphologyEx(blur, cv2.MORPH_OPEN, se)
        I2 = blur - background;
        blur = I2;
        red = blur;
        

    """ Need to do CLAHE but not working...? """
    #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    #cl1 = clahe.apply(blur)
    #from skimage import data, exposure, img_as_float
    #cl1 = exposure.equalize_adapthist(blur, kernel_size=None, clip_limit=0.01, nbins=256)
    thresh = threshold_otsu(red)
    binary = red > thresh
    return binary, red

""" Get rid of small pieces of O4 """
def dilate_red(binary):
    binary = binary.astype(np.uint8)
    sz = 60;
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz));   # get disk structuring element
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    sz = 60;  # gives 1092 cells, whereas sz = 20 ==> gives 1758 cells
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz));   # get disk structuring element
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    
    #plt.figure(); plt.imshow(Image.fromarray(opening * 255))
    #plt.imsave('final_image' + str(im_num) + '.tif', (new_fibers * 255).astype(np.uint16))
    return opening


""" Get rid of small pieces of O4 """
def dilate_red_QL(binary):
    binary = binary.astype(np.uint8)
    sz = 60;
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (sz, sz));   # get disk structuring element
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return closing

""" DAPI_count """
def DAPI_count(DAPI, background_sz):
    #Subtract background:
    DAPI_blur = cv2.GaussianBlur(DAPI,(5,5), 1)
    if background_sz < 100: # if image is very high magnification, don't bother
        sz = background_sz;
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*sz-1, 2*sz-1));   # get disk structuring element
        background = cv2.morphologyEx(DAPI_blur, cv2.MORPH_OPEN, se)
        
        I2 = DAPI_blur - background;
        DAPI_blur = I2;

    thresh = threshold_otsu(DAPI_blur)
    DAPI_binary = DAPI_blur > thresh
    #plt.figure(); plt.imshow(Image.fromarray(DAPI_binary * 255))
    
    """ do watershed """
    
    DAPI_binary = DAPI_binary.astype(np.uint8)
    
    """ DOESN'T WORK
    # sure background area
    kernel = np.ones((3,3), np.uint8)
    sure_bg = cv2.dilate(DAPI_binary, kernel, iterations=3)
    
    # find sure foreground area
    dist_transform = cv2.distanceTransform(DAPI_binary, cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)
    
    # find unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    
    # marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)
    
    # Add 1 to all label so that sure background is not 0, but 1
    markers = markers + 1
    
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    
    markers = cv2.watershed(DAPI_binary, markers)
    DAPI_binary[markers == -1] = [255, 0, 0]
    """
    # Now we want to separate the two objects in image
    # Generate the markers as local maxima of the distance to the background
    distance = ndi.distance_transform_edt(DAPI_binary)
    local_maxi = peak_local_max(distance, indices=False, min_distance = 5, labels=DAPI_binary)
    markers_new = ndi.label(local_maxi)[0]
    labels = watershed(-distance, markers_new, mask=DAPI_binary)
    #plt.figure(); plt.imshow(Image.fromarray(labels.astype(np.uint8)))
    return labels



def pre_process(input_arr, im_num, DAPI_size, rolling_ball_size, name='default', sav_dir=''):

    input_arr = np.asarray(input_arr)
    red = input_arr[:, :, 0]
    DAPI = input_arr[:, :, 2]
    
    binary, blur = im_adjust(red, rolling_ball_size)      # im adjust
    opening = dilate_red(binary) # dilate red, can make optional more lenient or not
    labels = DAPI_count(DAPI, background_sz=DAPI_size)  # count DAPI + do watershed
    
    """ Initiate list of CELL OBJECTS """
    threshold_DAPI = 0.3
    #DAPIsize = 20
     
    counter = 0
    counter_DAPI = 0
    
    """ Eliminate anything smaller than criteria """
    binary_all_fibers = labels > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=labels)
        
    final_counted = np.zeros(labels.shape)
    final_match_O4 = np.zeros(labels.shape)
    for Q in range(len(cc_overlap)):
       overlap_coords = cc_overlap[Q]['coords']
       area = cc_overlap[Q]['Area']
       perim = cc_overlap[Q]['perimeter'] 
       
       metric = 4*math.pi*area/(perim * perim)
       #print(angle)
       if metric > threshold_DAPI and (area > DAPI_size and area < 100000):
          #print(angle)
          match_O4 = 0
          counter_DAPI = counter_DAPI + 1
          for T in range(len(overlap_coords)):
             final_counted[overlap_coords[T,0], overlap_coords[T,1]] = Q
             
             if opening[overlap_coords[T,0], overlap_coords[T,1]] == 1:
                 match_O4 = 1
                 
          if match_O4:
             counter = counter + 1
             for T in range(len(overlap_coords)):
                 final_match_O4[overlap_coords[T,0], overlap_coords[T,1]] = Q
             
    """ Saving """
    candidates = final_match_O4 > 0
    plt.imsave(sav_dir + 'candidates' + str(im_num) + '_' + name + '.tif', (candidates))
        
    return candidates, counter, counter_DAPI, blur


def pre_process_QL(input_arr, im_num, DAPIsize, rolling_ball_size, name='default', sav_dir=''):

    input_arr = np.asarray(input_arr)
    red = input_arr[:, :, 0]
    DAPI = input_arr[:, :, 2]
    
    binary, blur = im_adjust(red, rolling_ball_size)      # im adjust
    opening = dilate_red_QL(binary) # dilate red, can make optional more lenient or not
    labels = DAPI_count(DAPI, background_sz=50)  # count DAPI + do watershed
    
    """ Initiate list of CELL OBJECTS """
    threshold_DAPI = 0.3
    #DAPIsize = 20
     
    counter = 0
    counter_DAPI = 0
    
    """ Eliminate anything smaller than criteria """
    binary_all_fibers = labels > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=labels)
        
    final_counted = np.zeros(labels.shape)
    final_match_O4 = np.zeros(labels.shape)
    for Q in range(len(cc_overlap)):
       overlap_coords = cc_overlap[Q]['coords']
       area = cc_overlap[Q]['Area']
       perim = cc_overlap[Q]['perimeter'] 
       
       metric = 4*math.pi*area/(perim * perim)
       #print(angle)
       if metric > threshold_DAPI and (area > DAPIsize and area < 100000):
          #print(angle)

          match_O4 = 0
          counter_DAPI = counter_DAPI + 1
          for T in range(len(overlap_coords)):
             final_counted[overlap_coords[T,0], overlap_coords[T,1]] = Q
             
             if opening[overlap_coords[T,0], overlap_coords[T,1]] == 1:
                 match_O4 = 1
                 
          if match_O4:
             counter = counter + 1
             for T in range(len(overlap_coords)):
                 final_match_O4[overlap_coords[T,0], overlap_coords[T,1]] = Q
             
    """ Saving """
    candidates = final_match_O4 > 0
    plt.imsave(sav_dir + 'candidates' + str(im_num) + '_' + name + '.tif', (candidates))
        
    return candidates, counter, counter_DAPI, blur
