# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:25:15 2017

@author: Tiger
"""

""" Retrieves validation images
"""

import numpy as np
from PIL import Image
from os import listdir
from os.path import isfile, join
from skimage import measure
from natsort import natsort_keygen, ns
import os
import pickle
import scipy.io as sio
from tifffile import imsave

import zipfile
import bz2

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from UNet import *

from skan import skeleton_to_csgraph
from skimage.morphology import skeletonize_3d

""" Take input bw image and returns coordinates and degrees pixel map, where
         degree == # of pixels in nearby CC space
                 more than 3 means branchpoint
                 == 2 means skeleton normal point
                 == 1 means endpoint
         coordinates == z,x,y coords of the full skeleton object
         
    *** works for 2D and 3D inputs ***
"""

def bw_skel_and_analyze(bw):
     if bw.ndim == 3:
          skeleton = skeletonize_3d(bw)
     elif bw.ndim == 2:
          skeleton = skeletonize(bw)
     skeleton[skeleton > 0] = 1
    
     
     if skeleton.any() and np.count_nonzero(skeleton) > 1:
          try:
               pixel_graph, coordinates, degrees = skeleton_to_csgraph(skeleton)
          except:
               pixel_graph = np.zeros(np.shape(skeleton))
               coordinates = []
               degrees = np.zeros(np.shape(skeleton))               
     else:
          pixel_graph = np.zeros(np.shape(skeleton))
          coordinates = []
          degrees = np.zeros(np.shape(skeleton))
          
     return degrees, coordinates
     
     
""" Convert voxel list to array """
def convert_vox_to_matrix(voxel_idx, zero_matrix):
    for row in voxel_idx:
        #print(row)
        zero_matrix[(row[0], row[1], row[2])] = 1
    return zero_matrix


""" For plotting the output as an interactive scroller"""
class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, :, self.ind])
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


""" Only keeps objects in stack that are 5 slices thick!!!"""
def slice_thresh(output_stack, slice_size=5):
    binary_overlap = output_stack > 0
    labelled = measure.label(binary_overlap)
    cc_overlap = measure.regionprops(labelled)
    
    all_voxels = []
    all_voxels_kept = []; total_blebs_kept = 0
    all_voxels_elim = []; total_blebs_elim = 0
    total_blebs_counted = len(cc_overlap)
    for bleb in cc_overlap:
        cur_bleb_coords = bleb['coords']
    
        # get only z-axis dimensions
        z_axis_span = cur_bleb_coords[:, -1]
    
        min_slice = min(z_axis_span)
        max_slice = max(z_axis_span)
        span = max_slice - min_slice
    
        """ ONLY KEEP OBJECTS that span > 5 slices """
        if span >= slice_size:
            print("WIDE ENOUGH object") 
            if len(all_voxels_kept) == 0:   # if it's empty, initialize
                all_voxels_kept = cur_bleb_coords
            else:
                all_voxels_kept = np.append(all_voxels_kept, cur_bleb_coords, axis = 0)
                
            total_blebs_kept = total_blebs_kept + 1
        else:
            print("NOT wide enough")
            if len(all_voxels_elim) == 0:
                print("came here")
                all_voxels_elim = cur_bleb_coords
            else:
                all_voxels_elim = np.append(all_voxels_elim, cur_bleb_coords, axis = 0)
                
            total_blebs_elim = total_blebs_elim + 1
       
        if len(all_voxels) == 0:   # if it's empty, initialize
            all_voxels = cur_bleb_coords
        else:
            all_voxels = np.append(all_voxels, cur_bleb_coords, axis = 0)
            
    print("Total kept: " + str(total_blebs_kept) + " Total eliminated: " + str(total_blebs_elim))
    
    
    """ convert voxels to matrix """
    all_seg = convert_vox_to_matrix(all_voxels, np.zeros(output_stack.shape))
    all_blebs = convert_vox_to_matrix(all_voxels_kept, np.zeros(output_stack.shape))
    all_eliminated = convert_vox_to_matrix(all_voxels_elim, np.zeros(output_stack.shape))
    
    return all_seg, all_blebs, all_eliminated


""" Find vectors of movement and eliminate blobs that migrate """
def distance_thresh(all_blebs_THRESH, average_thresh=15, max_thresh=15):
    
    # (1) Find and plot centroid of each 2D image object:
    centroid_matrix_3D = np.zeros(np.shape(all_blebs_THRESH))
    for i in range(len(all_blebs_THRESH[0, 0, :])):
        bin_cur_slice = all_blebs_THRESH[:, :, i] > 0
        label_cur_slice = measure.label(bin_cur_slice)
        cc_overlap_cur = measure.regionprops(label_cur_slice)
        
        for obj in cc_overlap_cur:
            centroid_matrix_3D[(int(obj['centroid'][0]),) + (int(obj['centroid'][1]),) + (i,)] = 1   # the "i" puts the centroid in the correct slice!!!
        
        #print(i)
        
    # (2) use 3D cc_overlap to find clusters of centroids
    binary_overlap = all_blebs_THRESH > 0
    labelled = measure.label(binary_overlap)
    cc_overlap_3D = measure.regionprops(labelled)
        
    all_voxels_kept = []; num_kept = 0
    all_voxels_elim = []; num_elim = 0
    for obj3D in cc_overlap_3D:
        
        slice_idx = np.unique(obj3D['coords'][:, -1])
        
        cropped_centroid_matrix = centroid_matrix_3D[:, :, min(slice_idx) : max(slice_idx) + 1]
        
        mask = np.ones(np.shape(cropped_centroid_matrix))

        translate_z_coords = obj3D['coords'][:, 0:2]
        z_coords = obj3D['coords'][:, 2:3]  % min(slice_idx)   # TRANSLATES z-coords to 0 by taking modulo of smallest slice index!!!
        translate_z_coords = np.append(translate_z_coords, z_coords, -1)
        
        obj_mask = convert_vox_to_matrix(translate_z_coords, np.zeros(cropped_centroid_matrix.shape))
        mask[obj_mask == 1] = 0 

        tmp_centroids = np.copy(cropped_centroid_matrix)  # contains only centroids that are masked by array above
        tmp_centroids[mask == 1] = 0
        
        
        ##mask = np.ones(np.shape(centroid_matrix_3D))
        ##obj_mask = convert_vox_to_matrix(obj3D['coords'], np.zeros(output_stack.shape))
        ##mask[obj_mask == 1] = 0 
    
        ##tmp_centroids = np.copy(centroid_matrix_3D)  # contains only centroids that are masked by array above
        ##tmp_centroids[mask == 1] = 0
        
        cc_overlap_cur_cent = measure.regionprops(np.asarray(tmp_centroids, dtype=np.int))  
        
        list_centroids = []
        for centroid in cc_overlap_cur_cent:
            if len(list_centroids) == 0:
                list_centroids = centroid['coords']
            else:
                list_centroids = np.append(list_centroids, centroid['coords'], axis = 0)
    
        sorted_centroids = sorted(list_centroids,key=lambda x: x[2])  # sort by column 3
        
        
        """ Any object with only 1 or less centroids is considered BAD, and is eliminated"""
        if len(sorted_centroids) <= 1:
            num_elim = num_elim + 1
            
            if len(all_voxels_elim) == 0:   # if it's empty, initialize
                all_voxels_elim = obj3D['coords']
            else:
                all_voxels_elim = np.append(all_voxels_elim, obj3D['coords'], axis = 0)
            continue;
        
    
        # (3) Find distance from 1st - 2nd - 3rd - 4th - 5th ect... centroids
        all_distances = []
        for i in range(len(sorted_centroids) - 1):
            center_1 = sorted_centroids[i]
            center_2 = sorted_centroids[i + 1]
            
            # Find distance:
            dist = math.sqrt(sum((center_1 - center_2)**2))           # DISTANCE FORMULA
            #print(dist)
            all_distances.append(dist)
        average_dist = sum(all_distances)/len(all_distances)
        max_dist = max(all_distances)
        
        
        # (4) If average distance is BELOW thresdhold, then keep the 3D cell body!!!
        # OR, if max distance moved > 15 pixels
        #print("average dist is: " + str(average_dist))
        if average_dist < average_thresh or max_dist < max_thresh:
            if len(all_voxels_kept) == 0:   # if it's empty, initialize
                all_voxels_kept = obj3D['coords']
            else:
                all_voxels_kept = np.append(all_voxels_kept, obj3D['coords'], axis = 0)
            
            num_kept = num_kept + 1
        else:
            num_elim = num_elim + 1
            
            if len(all_voxels_elim) == 0:   # if it's empty, initialize
                all_voxels_elim = obj3D['coords']
            else:
                all_voxels_elim = np.append(all_voxels_elim, obj3D['coords'], axis = 0)
            
        print("Finished distance thresholding for: " + str(num_elim + num_kept) + " out of " + str(len(cc_overlap_3D)) + " images")
    
    final_bleb_matrix = convert_vox_to_matrix(all_voxels_kept, np.zeros(all_blebs_THRESH.shape))
    elim_matrix = convert_vox_to_matrix(all_voxels_elim, np.zeros(all_blebs_THRESH.shape))
    print('Kept: ' + str(num_kept) + " eliminated: " + str(num_elim))
    
    return final_bleb_matrix, elim_matrix



""" converts a matrix into a multipage tiff to save!!! """
def convert_matrix_to_multipage_tiff(matrix):
    rolled = np.rollaxis(matrix, 2, 0).shape  # changes axis to be correct sizes
    tiff_image = np.zeros((rolled), 'uint8')
    for i in range(len(tiff_image)):
        tiff_image[i, :, :] = matrix[:, :, i]
        
    return tiff_image



def csv_num_sheaths_violin():
    
    categories = ['H1_v_AI', 'H1_v_H2', 'H1_v_H3', 'H1_v_MATLAB'];

    categories = ['H1_v_AI', 'H1_v_H2', 'e'];
      
    import csv
    with open('num_sheaths_data.csv', 'r') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
        newDF = pd.DataFrame()  

        i = 0        
        for row in spamreader:         

            row_int = []
            print (', '.join(row))
            for t in row[0]:
                if t != ',':
                    row_int.append(int(t))

            y = [i] * len(row_int)
                  
            df_ = pd.DataFrame(row_int, index=y, columns=categories[i+1:i+2])
            newDF = newDF.append(df_, ignore_index = True)

            i = i + 1
    #save_pkl(all_jaccard, '', 'all_jaccard' + name + '.pkl')
    plt.figure()
    sns.stripplot( data=newDF, jitter=True, orient='h');
    plt.xlabel('Jaccard')    

    plt.figure()
    sns.violinplot( data=newDF, jitter=True, orient='h');
    plt.xlabel('Jaccard')    

    plt.figure()
    sns.boxplot( data=newDF, orient='h');
    plt.xlabel('Jaccard')  

""" Parses through the validation zip to return counters that index to which files have
    fibers and which do not
    (speeds up processing time for batch so don't have to do this every time)    
"""
def parse_validation(myzip_val, onlyfiles_val, counter_val):
    
    counter_fibers = []
    counter_blank = []
    for T in range(len(counter_val)):
        """ Get validation images """
        index = counter_val[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_ZIP(myzip_val, filename)
        
        """ Check to see if contains fibers or not """
        if np.count_nonzero(truth_val[:, :, 1]) > 0:     # IF YES, there are fibers
            counter_fibers.append(T)

        elif np.count_nonzero(truth_val[:, :, 1]) == 0:
            counter_blank.append(T)
            
            
    return counter_fibers, counter_blank


""" Parses through the validation zip to return counters that index to which files have
    fibers and which do not
    (speeds up processing time for batch so don't have to do this every time)    
"""
def parse_validation_QL(myzip_val, onlyfiles_val, counter_val):
    
    counter_fibers = []
    counter_blank = []
    for T in range(len(counter_val)):
        """ Get validation images """
        index = counter_val[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_ZIP_Zpickle(myzip_val, filename)
        
        """ Check to see if contains fibers or not """
        if np.count_nonzero(truth_val[:, :, 1]) > 0:     # IF YES, there are fibers
            counter_fibers.append(T)

        elif np.count_nonzero(truth_val[:, :, 1]) == 0:
            counter_blank.append(T)
            
            
    return counter_fibers, counter_blank



""" changes QL images to cropped size """
def check_shape_QL(input_im, truth_im, len_im, width_im):
    input_arr = Image.fromarray(input_im.astype(np.uint8))
    
    resized = resize_adaptive(input_arr, 1500, method=Image.BICUBIC)
    resized_arr = np.asarray(resized)
    
    labelled = measure.label(resized_arr[:,:, 1])
    cc = measure.regionprops(labelled)
                   
    DAPI_idx = cc[0]['centroid']            
    # extract CROP outo of everything     
    len_im = 1024
    width_im = 640
    input_crop, coords = adapt_crop_DAPI(resized, DAPI_idx, length=len_im, width=width_im)
    
    """ resize the truth as well """
    truth_resized = Image.fromarray(truth_im[:, :, 1])
    truth_resized = resize_adaptive(truth_resized, 1500, method=Image.BICUBIC)
    
    truth_crop, coords_null = adapt_crop_DAPI(truth_resized, DAPI_idx, length=len_im, width=width_im)
    truth_crop[truth_crop > 0] = 1
    
    truth_whole = np.ones([len_im, width_im, 2])
    truth_null = truth_whole[:,:,0]
    
    truth_null[truth_crop == 1] = 0
    truth_whole[:,:,1] = truth_crop
    truth_whole[:,:,0] = truth_null
    
    truth_im = truth_whole
    input_im = input_crop
    
    return input_im, truth_im

""" returns a batch of validation images from the zip file """

def get_batch_val(myzip, onlyfiles_val, counter,  mean_arr, std_arr, batch_size=2):
    
    np.random.shuffle(counter)
    batch_x_val = [];
    batch_y_val = [];
    batch_weights = [];
    for T in range(int(batch_size)):
        """ Get validation images """
        index = counter[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_ZIP(myzip, filename)
        
        if input_val.shape[1] > 1500:
            input_val, truth_val = check_shape_QL(input_val, truth_val, len_im=1024, width_im=640)
        
        
        """ Normalize the image first """
        input_val = normalize_im(input_val, mean_arr, std_arr) 
        
        
        fiber_label = np.copy(truth_val[:, :, 1])
        """ Make spatial weighting map """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)

        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_val)
        weighted_labels[:, :, 1] = sp_weighted_labels
        
        
        
        batch_x_val.append(input_val)
        batch_y_val.append(truth_val)
        batch_weights.append(weighted_labels)
                    
    return batch_x_val, batch_y_val, batch_weights

""" returns a batch of validation images from the zip file """
def get_batch_val_QL(myzip, onlyfiles_val, counter,  mean_arr, std_arr, batch_size=2):
    
    np.random.shuffle(counter)
    batch_x_val = [];
    batch_y_val = [];
    batch_weights = [];
    for T in range(int(batch_size)):
        """ Get validation images """
        index = counter[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_ZIP_Zpickle(myzip, filename)
    
        input_val = input_val[0:1440, 0:1920, :]
        truth_val = truth_val[0:1440, 0:1920, :]    
        """ Normalize the image first """
        input_val = normalize_im(input_val, mean_arr, std_arr) 
        
        
        fiber_label = np.copy(truth_val[:, :, 1])
        """ Make spatial weighting map """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)

        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_val)
        weighted_labels[:, :, 1] = sp_weighted_labels
        
        
        
        batch_x_val.append(input_val)
        batch_y_val.append(truth_val)
        batch_weights.append(weighted_labels)
                    
    return batch_x_val, batch_y_val, batch_weights


""" returns a batch of validation images from the zip file """
def get_batch_val_bz(input_path, onlyfiles_val, counter,  mean_arr, std_arr, batch_size=2):
    
    np.random.shuffle(counter)
    batch_x_val = [];
    batch_y_val = [];
    batch_weights = [];
    for T in range(int(batch_size)):
        """ Get validation images """
        index = counter[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training_bz(input_path, filename)
    
        input_val = input_val[0:1440, 0:1920, :]
        truth_val = truth_val[0:1440, 0:1920, :]    
        """ Normalize the image first """
        input_val = normalize_im(input_val, mean_arr, std_arr) 
        
        
        fiber_label = np.copy(truth_val[:, :, 1])
        """ Make spatial weighting map """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)

        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_val)
        weighted_labels[:, :, 1] = sp_weighted_labels
        
        
        
        batch_x_val.append(input_val)
        batch_y_val.append(truth_val)
        batch_weights.append(weighted_labels)
                    
    return batch_x_val, batch_y_val, batch_weights


""" returns a batch of validation images from the zip file """
def get_batch_val_normal(input_path, onlyfiles_val, counter,  mean_arr, std_arr, batch_size=2):
    
    np.random.shuffle(counter)
    batch_x_val = [];
    batch_y_val = [];
    batch_weights = [];
    for T in range(int(batch_size)):
        """ Get validation images """
        index = counter[T]
        filename = onlyfiles_val[index]
        
        input_val, truth_val = load_training(input_path, filename)
    
        input_val = input_val[0:1440, 0:1920, :]
        truth_val = truth_val[0:1440, 0:1920, :]    
        """ Normalize the image first """
        input_val = normalize_im(input_val, mean_arr, std_arr) 
        
        
        fiber_label = np.copy(truth_val[:, :, 1])
        """ Make spatial weighting map """
        sp_weighted_labels = spatial_weight(fiber_label,edgeFalloff=10,background=0.01,approximate=True)

        """ Create a matrix of weighted labels """
        weighted_labels = np.copy(truth_val)
        weighted_labels[:, :, 1] = sp_weighted_labels
        
        
        
        batch_x_val.append(input_val)
        batch_y_val.append(truth_val)
        batch_weights.append(weighted_labels)
                    
    return batch_x_val, batch_y_val, batch_weights

""" Returns names of all files in folder path 
    Also returns counter to help with randomization
"""
def read_file_names(input_path):    
    # Read in file names
    onlyfiles = [ f for f in listdir(input_path) if isfile(join(input_path,f))]   
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    onlyfiles.sort(key = natsort_key1)

    return onlyfiles


""" Returns names of all files in zipfile 
    Also returns counter to help with randomization
"""
def read_zip_names(input_path, filename):    
    # Read in file names
    myzip = zipfile.ZipFile(input_path + filename, 'r')
    onlyfiles = myzip.namelist()
    natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
    onlyfiles.sort(key = natsort_key1)
    counter = list(range(len(onlyfiles)))  # create a counter, so can randomize it

    return myzip, onlyfiles, counter


""" Saving the objects """
def save_pkl(obj_save, s_path, name):
    with open(s_path + name, 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([obj_save], f)

"""Getting back the objects"""
def load_pkl(s_path, name):
    with open(s_path + name, 'rb') as f:  # Python 3: open(..., 'rb')
      loaded = pickle.load(f)
      obj_loaded = loaded[0]
      return obj_loaded


""" Load training data from zip archive """
def load_training_ZIP(myzip, filename):
    contents = pickle.load(myzip.open(filename))
    concate_input = contents[0]
    
    if np.shape(concate_input)[-1] < 6:   # originally only had 5
        input_im =concate_input[:, :, 0:3]
        truth_im =concate_input[:, :, 3:5]    
    else:                                 # now have extra fiber channel
        input_im =concate_input[:, :, 0:4]
        truth_im =concate_input[:, :, 4:6]    

    return input_im, truth_im
        

""" Load training data from zip archive """
def load_training_ZIP_Zpickle(myzip, filename):
    tmp = myzip.open(filename)
    contents = []
    with bz2.open(tmp, 'rb') as f:
        loaded_object = pickle.load(f)
        contents = loaded_object[0]
    concate_input = contents
    
    if np.shape(concate_input)[-1] < 6:   # originally only had 5
        input_im =concate_input[:, :, 0:3]
        truth_im =concate_input[:, :, 3:5]    
    else:                                 # now have extra fiber channel
        input_im =concate_input[:, :, 0:4]
        truth_im =concate_input[:, :, 4:6]    

    return input_im, truth_im

""" Load training data """
def load_training(s_path, filename):

    # Getting back the objects:
    with open(s_path + filename, 'rb') as f:  # Python 3: open(..., 'rb')
        loaded = pickle.load(f)
        concate_input = loaded[0]

    if np.shape(concate_input)[-1] < 6:   # originally only had 5
        input_im =concate_input[:, :, 0:3]
        truth_im =concate_input[:, :, 3:5]    
    else:                                 # now have extra fiber channel
        input_im =concate_input[:, :, 0:4]
        truth_im =concate_input[:, :, 4:6]    
    
    return input_im, truth_im

""" Load training zipped data """
def load_training_bz(s_path, filename):

    # Getting back the objects:
    contents = []
    with bz2.open(s_path + filename, 'rb') as f:
        loaded_object = pickle.load(f)
        contents = loaded_object[0]
    concate_input = contents

    if np.shape(concate_input)[-1] < 6:   # originally only had 5
        input_im =concate_input[:, :, 0:3]
        truth_im =concate_input[:, :, 3:5]    
    else:                                 # now have extra fiber channel
        input_im =concate_input[:, :, 0:4]
        truth_im =concate_input[:, :, 4:6]    
    
    return input_im, truth_im
        
        

def get_validate(test_input_path, DAPI_path, mask_path, mean_arr, std_arr):
        
  batch_x = []
  batch_y = []
  # Read in file names
  onlyfiles_mask = [ f for f in listdir(mask_path) if isfile(join(mask_path,f))]   
  natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order
  onlyfiles_mask.sort(key = natsort_key1)

  # Read in file names
  onlyfiles_DAPI = [ f for f in listdir(DAPI_path) if isfile(join(DAPI_path,f))]
  onlyfiles_DAPI.sort(key = natsort_key1)
      
  # Read in truth image names
  onlyfiles_test = [ f for f in listdir(test_input_path) if isfile(join(test_input_path,f))] 
  onlyfiles_test.sort(key = natsort_key1)    
    
  input_arr = readIm_counter(test_input_path,onlyfiles_test, 0)
  DAPI_arr = readIm_counter(DAPI_path,onlyfiles_DAPI, 0)
  mask_arr = readIm_counter(mask_path,onlyfiles_mask, 0)
            
  """
      Then split the DAPI into pixel idx list
      then cycle through each DAPI
      then CROP a rectangle around the input_arr, the DAPI_mask, AND the fibers_mask
      
      find corresponding fibers by adding (+1) to the value of the DAPI pixel
      then concatenate DAPI_mask to the input_arr
  """
  #DAPI_arr[DAPI_arr > 0] = 1
  DAPI_tmp = np.asarray(DAPI_arr, dtype=float)
  labelled = measure.label(DAPI_tmp)
  cc = measure.regionprops(labelled)
  
  # SHOULD RANDOMIZE THE COUNTER      
  counter_DAPI = list(range(len(cc)))  # create a counter, so can randomize it
  counter_DAPI = np.array(counter_DAPI)
  np.random.shuffle(counter_DAPI)
  
  N = 0
  while N < len(counter_DAPI):  
      DAPI_idx = cc[counter_DAPI[N]]['centroid']
      
      # extract CROP outo of everything          
      DAPI_crop = adapt_crop_DAPI(DAPI_arr, DAPI_idx, length=704, width=480)                    
      truth_crop = adapt_crop_DAPI(mask_arr, DAPI_idx, length=704, width=480)
      input_crop = adapt_crop_DAPI(input_arr, DAPI_idx, length=704, width=480)          
      
      """ Find fibers (truth_mask should already NOT contain DAPI, so don't need to get rid of it)
          ***however, the DAPI pixel value of DAPI_center should be the SAME as fibers pixel value + 1
      """
      val_at_center = DAPI_tmp[DAPI_idx[0].astype(int), DAPI_idx[1].astype(int)] 
      val_fibers = val_at_center + 1
      
      # Find all the ones that are == val_fibers
      truth_crop[truth_crop != val_fibers] = 0
      truth_crop[truth_crop == val_fibers] = 255
      
      # then split into binary classifier truth:
      fibers = np.copy(truth_crop)
      fibers = np.expand_dims(fibers, axis=3)
      
      null_space = np.copy(truth_crop)
      null_space[null_space == 0] = -1
      null_space[null_space > -1] = 0
      null_space[null_space == -1] = 1
      null_space = np.expand_dims(null_space, axis=3)
      
      combined = np.append(null_space, fibers, -1)
      
      """ Eliminate all other DAPI """
      DAPI_crop[DAPI_crop != val_at_center] = 0
      DAPI_crop[DAPI_crop == val_at_center] = 1
       
      """ Delete green channel by making it the DAPI_mask instead """
      input_crop[:, :, 1] = DAPI_crop

      """ Normalize here"""
      input_crop = normalize_im(input_crop, mean_arr, std_arr)  
      concate_input = input_crop
      
      """ set inputs and truth """
      batch_x.append(concate_input)
      batch_y.append(combined)
              
      N = N + 1
  return batch_x, batch_y






      
""" Adaptive cropping
        Inputs:
            - im ==> original full-scale image
            - DAPI_center ==> idx of center of DAPI point
            - length ==> width of SQUARE around the DAPI point to crop
        if nears an edge (index out of bounds) ==> will crop the opposite direction however many pixels are left  
"""
def adapt_crop_DAPI(im, DAPI_center, length=704, width=480):
    
    # first find limits of image
    w_lim, h_lim = im.size
    
    # then find limits of crop
    top = DAPI_center[0] - length/2
    bottom = DAPI_center[0] + length/2
    left = DAPI_center[1] - width/2
    right = DAPI_center[1] + width/2
    
    """ check if it's even possible to create this square
    """
    total = (bottom - top) * (right - left)
    if total > w_lim * h_lim:
        print("CROP TOO BIG")
        #throw exception
    
    """ if it's possible to create the square, adjust the excess as needed
    """
    if h_lim - bottom < 0:  # out of bounds height
        excess = bottom - h_lim
        bottom = h_lim  # reset the bottom to the MAX
        top = top - excess  # raise the top
    
    if top < 0: # out of bounds height
        excess = top * (-1)
        top = 0
        bottom = bottom + excess
        
    if w_lim - right < 0: # out of bounds width
        excess = right - w_lim
        right = w_lim
        left = left - excess
    
    if left < 0: # out of bounds width
        excess = left * (-1)
        left = 0
        right = right + excess
    
    """ CROP and convert to float array
    """
    cropped_im = im.crop((left, top, right, bottom))
    cropped_im = np.asarray(cropped_im, dtype=float)
    
    coords = [top, bottom, left, right]
    return cropped_im, coords



""" Adaptive cropping FOR ARRAYS
        Inputs:
            - im ==> original full-scale image
            - DAPI_center ==> idx of center of DAPI point
            - length ==> width of SQUARE around the DAPI point to crop
        if nears an edge (index out of bounds) ==> will crop the opposite direction however many pixels are left  
"""
def adapt_crop_DAPI_ARRAY(im, DAPI_center, length=704, width=480):
    
    # first find limits of image
    w_lim, h_lim = im.shape
    
    # then find limits of crop
    top = DAPI_center[0] - length/2
    bottom = DAPI_center[0] + length/2
    left = DAPI_center[1] - width/2
    right = DAPI_center[1] + width/2
    
    """ check if it's even possible to create this square
    """
    total = (bottom - top) * (right - left)
    if total > w_lim * h_lim:
        print("CROP TOO BIG")
        #throw exception
    
    """ if it's possible to create the square, adjust the excess as needed
    """
    if h_lim - bottom < 0:  # out of bounds height
        excess = bottom - h_lim
        bottom = h_lim  # reset the bottom to the MAX
        top = top - excess  # raise the top
    
    if top < 0: # out of bounds height
        excess = top * (-1)
        top = 0
        bottom = bottom + excess
        
    if w_lim - right < 0: # out of bounds width
        excess = right - w_lim
        right = w_lim
        left = left - excess
    
    if left < 0: # out of bounds width
        excess = left * (-1)
        left = 0
        right = right + excess
    

    """ Ensure within boundaries """
    top = int(top)
    bottom = int(bottom)
    right = int(right)
    left = int(left)
    add_l = length - (bottom - top)
    add_w = width - (right - left)
    
    if add_l: bottom = bottom + add_l
    if add_w: right = right + add_w
    
    """ CROP """    
    cropped_im = im[int(left):int(right), int(top):int(bottom)] 
    #cropped_im = np.asarray(cropped_im, dtype=float)
    
    coords = [top, bottom, left, right]
    return cropped_im, coords

"""
    Find standard deviation + mean
"""
def calc_avg_mod(input_path, onlyfiles_mask):


    array_of_ims = []    
    for i in range(len(onlyfiles_mask)):
        filename = onlyfiles_mask[i]
        input_im, truth_im = load_training(input_path, filename)
    
        array_of_ims.append(input_im)
        
    mean_arr = np.mean(array_of_ims)  # saves the mean_arr to be used for cross-validation
    array_of_ims = array_of_ims - mean_arr
    std_arr = np.std(array_of_ims)  # saves the std_arr to be used for cross-validation
    array_of_ims = array_of_ims / std_arr 


    return mean_arr, std_arr      



"""
    Find standard deviation + mean
"""
def calc_avg(input_path):
    
    cwd = os.getcwd()     # get current directory
    os.chdir(input_path)   # change path
    
    # Access all PNG files in directory
    allfiles=os.listdir(os.getcwd())
    imlist=[filename for filename in allfiles if  filename[-4:] in [".tif",".TIF"]]

    array_of_ims = []
    # Build up list of images that have been casted to float
    for im in imlist:
        im_org = Image.open(im)
        im_res = resize(im_org, h=4104, w=4104)
        imarr=np.array(im_res,dtype=np.float)
        array_of_ims.append(imarr)

    mean_arr = np.mean(array_of_ims)  # saves the mean_arr to be used for cross-validation
    array_of_ims = array_of_ims - mean_arr
    std_arr = np.std(array_of_ims)  # saves the std_arr to be used for cross-validation
    array_of_ims = array_of_ims / std_arr 

    os.chdir(cwd)

    return mean_arr, std_arr      


"""
    To normalize by the mean and std
"""
def normalize_im(im, mean_arr, std_arr):
    normalized = (im - mean_arr)/std_arr 
    return normalized        
    
"""
    if RGB ==> num_dims == 3
"""
def crop_im(mask, num_dims, width, height):
    left = 0; up =  0
    right = width; down = height 
          
    new_c = mask.crop((left, up, right, down))         
    new_c = np.array(new_c)
    if num_dims == 2:
        new_c = np.expand_dims(new_c, axis=3)
    all_crop = new_c
    
    return all_crop

def resize(im, size_h=8208, size_w=8208, method=Image.BICUBIC):
    
    im = im.resize([size_h,size_w], resample=method)
    return im


def resize_adaptive(img, basewidth, method=Image.BICUBIC):
    wpercent = (basewidth / float(img.size[0]))
    hsize = int((float(img.size[1]) * float(wpercent)))
    img = img.resize((basewidth, hsize), resample=method)
    return img


""" Reads in image at path "mypath" and for certain file number "fileNum"
"""
def readIm_counter(mypath, onlyfiles, fileNum, size_h=8208, size_w=8208): 
    curpath = mypath + onlyfiles[fileNum]
    im = Image.open(curpath)      
    return im

""" Reads in image at path "mypath" and for certain file number "fileNum"
"""
def readIm_counter_DAPI(mypath, onlyfiles, fileNum): 
    curpath = mypath + onlyfiles[fileNum]
    im = Image.open(curpath).convert('L')
    return im

""" Reads in image at path "mypath" and for certain file number "fileNum"
"""
def readIm_counter_MATLAB(mypath, onlyfiles, fileNum): 
    curpath = mypath + onlyfiles[fileNum]
    mat_contents = sio.loadmat(curpath)
    mask = mat_contents['save_im']    
    #mask = Image.fromarray(mask)    
    return mask


""" Reads in image at path "mypath" and for certain file number "fileNum"
"""
def readIm_ZIP(mypath, onlyfiles, fileNum,  size_h=8208, size_w=8208): 
    curpath = mypath + onlyfiles[fileNum]
    im = Image.open(curpath)
    im = resize(im, h=size_h, w=size_w)        
    return im