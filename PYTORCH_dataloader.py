# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:10:03 2020

@author: tiger
"""


import torchvision.transforms.functional as TF
import random
from torch.utils import data
import torch
import time
import numpy as np

import scipy
import math
import tifffile as tifffile

# import torchio
# from torchio.transforms import (
#     RescaleIntensity,
#     RandomFlip,
#     RandomAffine,
#     RandomElasticDeformation,
#     RandomMotion,
#     RandomBiasField,
#     RandomBlur,
#     RandomNoise,
#     Interpolation,
#     Compose
# )
# from torchio import Image, Subject, ImagesDataset


""" Extended functions for SNAKE_SEG """

""" Load data directly from tiffs with seed mask """
class Dataset_tiffs_snake_seg(data.Dataset):
  def __init__(self, list_IDs, examples, mean, std, sp_weight_bool=0, transforms=0, dist_loss=0):
        'Initialization'
        #self.labels = labels
        self.list_IDs = list_IDs
        self.examples = examples
        self.transforms = transforms
        self.mean = mean
        self.std = std
        self.sp_weight_bool = sp_weight_bool
        self.dist_loss = dist_loss

  def apply_transforms(self, image, labels):
        inputs = image
        inputs = torch.tensor(inputs, dtype = torch.float,requires_grad=False)
        labels = torch.tensor(labels, dtype = torch.long, requires_grad=False)         
 
        subject_a = Subject(
                one_image=Image(None,  torchio.INTENSITY, inputs),   # *** must be tensors!!!
                a_segmentation=Image(None, torchio.LABEL, labels))
          
        subjects_list = [subject_a]

        subjects_dataset = ImagesDataset(subjects_list, transform=self.transforms)
        subject_sample = subjects_dataset[0]
          
          
        X = subject_sample['one_image']['data'].numpy()
        Y = subject_sample['a_segmentation']['data'].numpy()
        
        return X[0], Y[0]    
    
    
  def append_seed_mask(self, image, seed_crop):
     
        #seed_crop[seed_crop > 0] = 1

        """ Added because need to check to make sure no objects leaving out of frame during small crop 
        
                *** DOUBLE CHECK THIS IN MATLAB???
        
        """
        #seed_crop = np.expand_dims(seed_crop, axis=0)
        #seed_crop = check_resized(seed_crop, depth, width_max=input_size, height_max=input_size)
        #seed_crop = seed_crop[:, :, :, 0]

    
        """ Append seed to input """
        temp = np.zeros((2, ) + np.shape(image))
        temp[0,...] = image
        seed_crop[seed_crop > 0] = 255
        temp[1,...] = seed_crop
                             
        return temp
    
  # def create_dist_loss(self, labels):
  #        posmask = labels
  #        negmask = ~posmask
  #        spatial_weight = scipy.ndimage.distance_transform_cdt(posmask) + scipy.ndimage.distance_transform_cdt(negmask)

  #        return spatial_weight
     
        
     
  def create_spatial_weight_mat(self, labels, edgeFalloff=10,background=0.01,approximate=True):
       
         if approximate:   # does chebyshev
             dist1 = scipy.ndimage.distance_transform_cdt(labels)
             dist2 = scipy.ndimage.distance_transform_cdt(np.where(labels>0,0,1))    # sets everything in the middle of the OBJECT to be 0
                     
         else:   # does euclidean
             dist1 = scipy.ndimage.distance_transform_edt(labels, sampling=[1,1,1])
             dist2 = scipy.ndimage.distance_transform_edt(np.where(labels>0,0,1), sampling=[1,1,1])
             
         """ DO CLASS WEIGHTING instead of spatial weighting WITHIN the object """
         dist1[dist1 > 0] = 0.5
     
         dist = dist1+dist2
         attention = math.e**(1-dist/edgeFalloff) + background   # adds background so no loses go to zero
         attention /= np.average(attention)
         return np.reshape(attention,labels.shape)    
     

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        input_name = self.examples[ID]['input']
        truth_name = self.examples[ID]['truth']
        seed_name = self.examples[ID]['seed_crop']

        X = tifffile.imread(input_name)
        Y = tifffile.imread(truth_name)
        seed_crop = tifffile.imread(seed_name)
        Y[Y > 0] = 1
        
        
        """ Get spatial weight matrix """
        if self.sp_weight_bool:
             spatial_weight = self.create_spatial_weight_mat(Y)
             
        elif self.dist_loss:
             spatial_weight = self.create_dist_loss(Y)
        else:
             spatial_weight = []
             
             
           
        
        """ Do normalization here??? """
        #X  = (X  - self.mean)/self.std
        
        """ If want to do transform on CPU """
        if self.transforms:
             X, Y = self.apply_transforms(X, Y)  
        
        """ Append seed mask """
        X = self.append_seed_mask(X, seed_crop) 


        """ If want to do lr_finder """
        # X = np.asarray(X, dtype=np.float32)
        # X = (X - self.mean)/self.std    
        # #X = np.expand_dims(X, axis=0)
        # X = torch.tensor(X, dtype = torch.float, requires_grad=False)
        # Y = torch.tensor(Y, dtype = torch.long, requires_grad=False) 

            
        return X, Y, spatial_weight






























""" Calculate Jaccard on the GPU """
def jacc_eval_GPU_torch(output, truth, ax_labels=-1, argmax_truth=1):
    
      output = torch.argmax(output,axis=1)
      intersection = torch.sum(torch.sum(output * truth, axis=ax_labels),axis=ax_labels)
      union = torch.sum(torch.sum(torch.add(output, truth)>= 1, axis=ax_labels),axis=ax_labels) + 0.0000001
      jaccard = torch.mean(intersection / union)  # find mean of jaccard over all slices     
          
      
      ### per image ==> is the same???
      # all_jacc = [];
      # for o_single, t_single in zip(output, truth):
      #     o_single = torch.argmax(o_single,axis=0)
      #     intersection = torch.sum(torch.sum(o_single * t_single, axis=ax_labels),axis=ax_labels)
      #     union = torch.sum(torch.sum(torch.add(o_single, t_single)>= 1, axis=ax_labels),axis=ax_labels) + 0.0000001
      #     jaccard = torch.mean(intersection / union)  # find mean of jaccard over all slices   
      #     all_jacc.append(jaccard.cpu().data.numpy())

          
      return jaccard

""" Define transforms"""


def initialize_transforms(p=0.5):
     transforms = [
           RandomFlip(axes = 0, flip_probability = 0.5, p = p, seed = None),
           
           RandomAffine(scales=(0.9, 1.1), degrees=(10), isotropic=False,
                        default_pad_value='otsu', image_interpolation=Interpolation.LINEAR,
                        p = p, seed=None),
           
           # *** SLOWS DOWN DATALOADER ***
           #RandomElasticDeformation(num_control_points = 7, max_displacement = 7.5,
           #                         locked_borders = 2, image_interpolation = Interpolation.LINEAR,
           #                         p = 0.5, seed = None),
           RandomMotion(degrees = 10, translation = 10, num_transforms = 2, image_interpolation = Interpolation.LINEAR,
                        p = p, seed = None),
           
           RandomBiasField(coefficients=0.5, order = 3, p = p, seed = None),
           
           RandomBlur(std = (0, 4), p = p, seed=None),
           
           RandomNoise(mean = 0, std = (0, 0.25), p = p, seed = None),
           RescaleIntensity((0, 255))
           
     ]
     transform = Compose(transforms)
     return transform




def initialize_transforms_simple(p=0.5):
     transforms = [
           RandomFlip(axes = 0, flip_probability = 0.5, p = p, seed = None),
           
           #RandomAffine(scales=(0.9, 1.1), degrees=(10), isotropic=False,
           #             default_pad_value='otsu', image_interpolation=Interpolation.LINEAR,
           #             p = p, seed=None),
           
           # *** SLOWS DOWN DATALOADER ***
           #RandomElasticDeformation(num_control_points = 7, max_displacement = 7.5,
           #                         locked_borders = 2, image_interpolation = Interpolation.LINEAR,
           #                         p = 0.5, seed = None),
           #RandomMotion(degrees = 10, translation = 10, num_transforms = 2, image_interpolation = Interpolation.LINEAR,
           #             p = p, seed = None),
           
           #RandomBiasField(coefficients=0.5, order = 3, p = p, seed = None),
           
           #RandomBlur(std = (0, 4), p = p, seed=None),
           
           #RandomNoise(mean = 0, std = (0, 0.25), p = p, seed = None),
           #RescaleIntensity((0, 255))
           
     ]
     transform = Compose(transforms)
     return transform



""" Do pre-processing on GPU
          ***can't do augmentation/transforms here because of CPU requirement for torchio

"""
def transfer_to_GPU(X, Y, device, mean, std, transforms = 0):
     """ Put these at beginning later """
     mean = torch.tensor(mean, dtype = torch.float, device=device, requires_grad=False)
     std = torch.tensor(std, dtype = torch.float, device=device, requires_grad=False)
     
     """ Convert to Tensor """
     inputs = torch.tensor(X, dtype = torch.float, device=device, requires_grad=False)
     labels = torch.tensor(Y, dtype = torch.long, device=device, requires_grad=False)           

     """ Normalization """
     inputs = (inputs - mean)/std
                
     """ Expand dims """
     inputs = inputs.unsqueeze(1)   

     return inputs, labels



""" Load data directly from tiffs """
class Dataset_tiffs(data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, list_IDs, examples, mean, std, transforms = 0):
        'Initialization'
        #self.labels = labels
        self.list_IDs = list_IDs
        self.examples = examples
        self.transforms = transforms
        self.mean = mean
        self.std = std

  def apply_transforms(self, image, labels):
        #inputs = np.asarray(image, dtype=np.float32)
        inputs = image

 
        inputs = torch.tensor(inputs, dtype = torch.float,requires_grad=False)
        labels = torch.tensor(labels, dtype = torch.long, requires_grad=False)         
 
        subject_a = Subject(
                one_image=Image(None,  torchio.INTENSITY, inputs),   # *** must be tensors!!!
                a_segmentation=Image(None, torchio.LABEL, labels))
          
        subjects_list = [subject_a]

        subjects_dataset = ImagesDataset(subjects_list, transform=self.transforms)
        subject_sample = subjects_dataset[0]
          
          
        X = subject_sample['one_image']['data'].numpy()
        Y = subject_sample['a_segmentation']['data'].numpy()
        
        return X[0], Y[0]    

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.list_IDs)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        ID = self.list_IDs[index]

        # Load data and get label
        #X = torch.load('data/' + ID + '.pt')
        #y = self.labels[ID]

 
        input_name = self.examples[ID]['input']
        truth_name = self.examples[ID]['truth']

        X = tifffile.imread(input_name)
        #X = np.expand_dims(X, axis=0)
        Y = tifffile.imread(truth_name)
        Y[Y > 0] = 1
        #Y = np.expand_dims(Y, axis=0)

        """ Do normalization here??? """
        #X  = (X  - self.mean)/self.std


        """ If want to do transform on CPU """
        if self.transforms:
             X, Y = self.apply_transforms(X, Y)  
        
             
        return X, Y

