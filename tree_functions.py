#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 26 23:26:11 2020

@author: user
"""


import numpy as np
from matlab_crop_function import *
from off_shoot_functions import *

from plot_functions_CLEANED import *
from data_functions_CLEANED import *
from data_functions_3D import *
import pandas as pd




""" Returns the coordinates of the parents of the current starting index """
def get_parent_nodes(tree, start_ind, num_parents, parent_coords):
    
    if num_parents == 0 or start_ind == -1:
        return parent_coords
    
    else:
        parent_ind = tree.parent[start_ind] 
        
        if parent_ind == -1:
            print("hit bottom of tree")
            return parent_coords
        
        parent_coords.append(tree.coords[parent_ind])
        
        parent_coords.append(tree.start_be_coord[parent_ind][math.floor(len(tree.start_be_coord[parent_ind])/2)])
      
        if not np.isnan(tree.end_be_coord[parent_ind]).any():
            parent_coords.append(tree.end_be_coord[parent_ind][math.floor(len(tree.end_be_coord[parent_ind])/2)])
            
            
        parent_coords = get_parent_nodes(tree, start_ind=parent_ind, num_parents=num_parents - 1, parent_coords=parent_coords)


        return parent_coords



""" Convert list into tree in pandas dataframe """
def get_tree_from_im_list(root, input_im, width_tmp, height_tmp, depth_tmp, all_coords_root):
    columns = {'coords', 'parent', 'child', 'depth', 'start_be_coord', 'end_be_coord', 'cur_idx', 'visited'}
    tree_df = pd.DataFrame(columns=columns)
    

    """ add the root to an empty image and find it's skeleton + branchpoints + edges ==> maybe needs a bit of smoothing as well??? """
    empty_root_im = np.zeros(np.shape(input_im))
    for idx_trace in range(len(root)):
          empty_root_im[root[idx_trace][0], root[idx_trace][1], root[idx_trace][2]] = 1        
          if idx_trace == int(len(root)/2):
               centroid = [root[idx_trace][0], root[idx_trace][1], root[idx_trace][2]]    
               
    """ Crop for speed skeletonization """    
    x = int(centroid[0]); y = int(centroid[1]); z = int(centroid[2])
    crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max = crop_around_centroid(empty_root_im, y, x, z, 
                                                                                crop_size=250, z_size=100, height=height_tmp, width=width_tmp, depth=depth_tmp)
     
    """ add endpoints to a new list of points to visit (seed_idx) """
    degrees_small, coordinates = bw_skel_and_analyze(crop)
    degrees_full_size = np.zeros(np.shape(input_im))
    degrees_full_size[box_x_min:box_x_max, box_y_min:box_y_max, box_z_min:box_z_max] = degrees_small
    degrees = degrees_full_size
    
    end_points = np.copy(degrees); end_points[end_points != 1] = 0
    branch_points = np.copy(degrees); branch_points[branch_points <= 2] = 0  ### ANYTHING ABOVE 2 is a branchpoint!
                                ### number of pixel value in degrees is connectivity!!!
                            
    coords_end_points = np.transpose(np.nonzero(end_points))



    """ HACK: find point closest to interactive scroller """
    # degrees_small[degrees_small > 0] = 1
    # only_colocalized_mask, overall_coord = GUI_cell_selector(degrees_small, crop_size=100, z_size=30,
    #                                                         height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_tmp, thresh=0)
    
    # overall_coord = scale_coords_of_crop_to_full(np.transpose(np.vstack(overall_coord)), box_x_min, box_y_min, box_z_min)


    """ To find root ==> is side closest to middle of cell...??? """
    dist_to_root = []
    coord_root = [0, 0, 0]
    for check_coord in all_coords_root:
        expanded = expand_coord_to_neighborhood(check_coord, lower=1, upper=2)
        if (np.vstack(expanded)[:, None] == coords_end_points).all(-1).any():            
            coord_root = check_coord
            
    
    """ next find segment tied to each branchpoint by searching the +/- 1 neighborhood for matching indices
    """
    all_neighborhoods, all_hood_first_last, root_neighborhood = get_neighborhoods(degrees, coord_root=coord_root)
    
    
    
    if len(root_neighborhood) == 0:
        return [], []
        
   
    """ Create tree """
    depth = 0                 
    tree_df, all_children = treeify(tree_df, depth, root_neighborhood, all_neighborhoods, all_hood_first_last, cur_idx = 0, parent= -1)                 

    return tree_df, all_children



""" expand coords into a neighborhood """
def expand_coord_to_neighborhood(coords, lower, upper):
    neighborhood_be = []
    for idx in coords:
        for x in range(-lower, upper):
            for y in range(-lower, upper):
                for z in range(-lower, upper):
                    new_idx = [idx[0] + x, idx[1] + y, idx[2] + z]
                    neighborhood_be.append(new_idx)    
    return neighborhood_be


""" Get neighborhoods from an image ==> include scaling??? """
def get_neighborhoods(degrees, coord_root=0, scale=0, box_x_min=0, box_y_min=0, box_z_min=0, width=1000000000, height=100000000, depth=100000000):
      only_segments = np.copy(degrees); only_segments[only_segments != 2] = 0
      only_branch_ends = np.copy(degrees); only_branch_ends[only_branch_ends == 2] = 0; only_branch_ends[only_branch_ends > 0] = 3; 
      
      ### convert branch and endpoints into a list with +/- neihgbourhood values
      labels = measure.label(only_branch_ends)
      cc_be = measure.regionprops(labels)

      all_neighborhoods = []
      root_neighborhood = []
      
      for branch_end in cc_be:
          coords = branch_end['coords']
          
          neighborhood_be = expand_coord_to_neighborhood(coords, lower=1, upper=2)

          if (np.vstack(neighborhood_be)[:, None] == coord_root).all(-1).any():
              root_neighborhood.append(np.vstack(neighborhood_be))
              print(neighborhood_be)
          else:
              neighborhood_be = np.vstack(neighborhood_be)
              if scale:
                  neighborhood_be = scale_coords_of_crop_to_full(neighborhood_be, box_x_min, box_y_min, box_z_min)
                  
              all_neighborhoods.append(neighborhood_be)
                      
      ### convert segments into just coords and ALSO get there neighborhoods for their FIRST and LAST indices
      labels = measure.label(only_segments)
      cc_segs = measure.regionprops(labels)
      
      all_hood_first_last = []          
      idx = 0
      for seg in cc_segs:
          coords = np.vstack(coords)
          coords = seg['coords']       
          if scale:
              coords = scale_coords_of_crop_to_full(coords, box_x_min, box_y_min, box_z_min)
              
          all_hood_first_last.append(coords)
          idx += 1
      
      return all_neighborhoods, all_hood_first_last, root_neighborhood




""" Check size limits """
def check_limits(all_neighborhoods, width_tmp, height_tmp, depth_tmp):
    
        """ Make sure nothing exceeds size limits """
        idx = 0; 
        for neighbor_be in all_neighborhoods:
            
            if len(neighbor_be) > 0:
                if np.any(neighbor_be[:, 0] >= width_tmp):
                    all_neighborhoods[idx][np.where(neighbor_be[:, 0] >= width_tmp), 0] = width_tmp - 1
                    
                if np.any(neighbor_be[:, 1] >= height_tmp):
                    all_neighborhoods[idx][np.where(neighbor_be[:, 1] >= height_tmp), 1] = height_tmp - 1
    
                if np.any(neighbor_be[:, 2] >= depth_tmp):
                    all_neighborhoods[idx][np.where(neighbor_be[:, 2] >= depth_tmp), 2] = depth_tmp - 1
            idx += 1
            
        return all_neighborhoods   
    



""" Create tree """
def treeify(tree_df, depth, root_neighborhood, all_neighborhoods, all_hood_first_last, cur_idx = 0, parent= -1, start=0, width_tmp=1000000000, height_tmp=100000000, depth_tmp=100000000):


        """ Make sure nothing exceeds size limits """
        all_neighborhoods = check_limits(all_neighborhoods, width_tmp, height_tmp, depth_tmp)
        # idx = 0; 
        # for neighbor_be in all_neighborhoods:
            
        #     if len(neighbor_be) > 0:
        #         if np.any(neighbor_be[:, 0] >= width_tmp):
        #             all_neighborhoods[idx][np.where(neighbor_be[:, 0] >= width_tmp), 0] = width_tmp - 1
        #         if np.any(neighbor_be[:, 1] >= height_tmp):
        #             all_neighborhoods[idx][np.where(neighbor_be[:, 1] >= height_tmp), 0] = height_tmp - 1
    
        #         if np.any(neighbor_be[:, 2] >= depth_tmp):
        #             all_neighborhoods[idx][np.where(neighbor_be[:, 2] >= depth_tmp), 0] = depth_tmp - 1
        #     idx += 1
    
    
    
        all_hood_first_last = check_limits(all_hood_first_last, width_tmp, height_tmp, depth_tmp)
        # idx = 0; 
        # for h_first_last in all_hood_first_last:
            
        #     if len(h_first_last) > 0:
        #         if np.any(h_first_last[:, 0] >= width_tmp):
        #             all_hood_first_last[idx][np.where(h_first_last[:, 0] >= width_tmp), 0] = width_tmp - 1
        #         if np.any(h_first_last[:, 1] >= height_tmp):
        #             all_hood_first_last[idx][np.where(h_first_last[:, 1] >= height_tmp), 0] = height_tmp - 1
    
        #         if np.any(h_first_last[:, 2] >= depth_tmp):
        #             all_hood_first_last[idx][np.where(h_first_last[:, 2] >= depth_tmp), 0] = depth_tmp - 1
        #     idx += 1    

        # IF ROOT (depth == 0) ==> then use root neighborhood
        if len(tree_df) == 0:
            cur_be = np.vstack(root_neighborhood[0])
        elif start:
            cur_be = np.vstack(root_neighborhood)
                   
            
        else:
            idx_parent_df = np.where(tree_df.cur_idx == parent)
            cur_be = np.vstack(tree_df.end_be_coord[idx_parent_df[0][0]])

        ### find next seg                    
        all_children = [];
        for idx_cur_seg in range(len(all_hood_first_last)):
            if not np.asarray(all_hood_first_last[idx_cur_seg]).any():
                continue   # skip if empty
                
            cur_seg = np.vstack(all_hood_first_last[idx_cur_seg])
            if (cur_seg[:, None] == cur_be).all(-1).any():
                
                if len(tree_df) > 0:
                    cur_idx = np.max(tree_df.cur_idx[:]) + 1;  

                full_seg_coords = np.vstack(cur_seg)
                
                ### ADD NEW NODE TO TREE
                new_node = {'coords': full_seg_coords, 'parent': parent, 'child': [], 'depth': depth, 'cur_idx': cur_idx, 'start_be_coord': cur_be, 'end_be_coord': np.nan, 'visited': np.nan}
                tree_df = tree_df.append(new_node, ignore_index=True)
                
                ### find next be
                next_be = []; all_neighborhoods_tmp = all_neighborhoods
                
                isempty = 1
                for idx_cur_be in range(len(all_neighborhoods)):
                    if not np.asarray(all_neighborhoods[idx_cur_be]).any():
                        continue # skip if empty
                    search_be = np.vstack(all_neighborhoods[idx_cur_be])
                    
                    if (cur_seg[:, None] == search_be).all(-1).any():        
                        next_be.append(search_be)
                        
                        # delete the neighborhood we currently assessed
                        all_neighborhoods_tmp[idx_cur_be] = [];
                        isempty = 0
                        
                print(cur_idx)

                # delete the neighborhood we currently assessed
                all_hood_first_last[idx_cur_seg] = []
                # append children to send to previous call
                all_children.append(cur_idx)
                
                
                if not isempty and np.asarray(np.vstack(next_be)).any():
                    next_be = np.vstack(next_be)
                    idx_parent_df = np.where(tree_df.cur_idx == cur_idx)
                    tree_df.end_be_coord[idx_parent_df[0][0]] = next_be
                                                
                    # recurse
                    tree_df, next_children = treeify(tree_df, depth + 1, 
                                          root_neighborhood, all_neighborhoods_tmp, all_hood_first_last, cur_idx=cur_idx, parent=cur_idx,
                                          width_tmp=width_tmp, height_tmp=height_tmp, depth_tmp=depth_tmp) 
    
                    # add all children from next call
                    tree_df.child[cur_idx].append(next_children)


        ### convert empty lists to "nan" ONLY in the end_be_coord column
        # if parent == -1 or start == 1:
        #     #tree_df = tree_df.mask(tree_df.applymap(str).eq('[]'))
        #     #tree_df = tree_df.mask(tree_df.applymap(str).eq('[[]]'))
        #     a = 1

        return tree_df, all_children
    
    


    
    
""" Plot tree """
def show_tree(tree_df, im):
    all_segments = np.asarray(tree_df.coords[:])
    indices = np.asarray(tree_df.cur_idx)
    start_be_coord = np.asarray(tree_df.start_be_coord[:])

    end_be_coord = np.asarray(tree_df.end_be_coord[:])

    for seg, ind, start_be, end_be in zip(all_segments, indices, start_be_coord, end_be_coord):
        
        im[seg[:, 0], seg[:, 1], seg[:, 2]] = ind + 1 # because dont want non-zero
        
        im[start_be[:, 0], start_be[:, 1], start_be[:, 2]] = 1
        
        if not np.isnan(end_be).any() and np.asarray(end_be).any():
            im[end_be[:, 0], end_be[:, 1], end_be[:, 2]] = 2
    return im