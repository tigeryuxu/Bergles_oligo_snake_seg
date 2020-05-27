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





""" Convert list into tree in pandas dataframe """
def get_tree_from_im_list(root, input_im, width_tmp, height_tmp, depth_tmp):
    columns = {'coords', 'parent', 'child', 'depth', 'start_be_coord', 'end_be_coord', 'cur_idx'}
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
    degrees_full_size[box_x_min -1:box_x_max -1, box_y_min-1:box_y_max-1, box_z_min-1:box_z_max-1] = degrees_small
    degrees = degrees_full_size
    
    end_points = np.copy(degrees); end_points[end_points != 1] = 0
    branch_points = np.copy(degrees); branch_points[branch_points <= 2] = 0  ### ANYTHING ABOVE 2 is a branchpoint!
                                ### number of pixel value in degrees is connectivity!!!
                            
    coords_end_points = np.transpose(np.nonzero(end_points))


    """ scale the pixel indices back to the original size values """
    # new_ep_coords = []
    # for end_point in coords_end_points:
    #       ep_center = scale_coords_of_crop_to_full(end_point, box_x_min, box_y_min, box_z_min)
    #       new_ep_coords.append(ep_center)   
    # coords_end_points = new_ep_coords

    """ HACK: find point closest to interactive scroller """
    only_colocalized_mask, overall_coord = GUI_cell_selector(degrees_small, crop_size=100, z_size=30,
                                                            height_tmp=height_tmp, width_tmp=width_tmp, depth_tmp=depth_tmp)
    
    overall_coord = scale_coords_of_crop_to_full(overall_coord, box_x_min, box_y_min, box_z_min)


    """ To find root ==> is side closest to middle of cell...??? """
    vec_dist = np.abs(overall_coord) - np.abs(coords_end_points)
    all_dist = []
    for vec in vec_dist:
        all_dist.append(np.linalg.norm(vec))
    
    idx_min = all_dist.index(np.min(all_dist))
    coord_root = coords_end_points[idx_min]
    

    """ next find segment tied to each branchpoint by searching the +/- 1 neighborhood for matching indices
    """
    only_segments = np.copy(degrees); only_segments[only_segments != 2] = 0
    only_branch_ends = np.copy(degrees); only_branch_ends[only_branch_ends == 2] = 0; only_branch_ends[only_branch_ends > 0] = 3; 
    
    ### convert branch and endpoints into a list with +/- neihgbourhood values
    labels = measure.label(only_branch_ends)
    cc_be = measure.regionprops(labels)
    
    
    all_neighborhoods = []
    root_neighborhood = []
    
    lower = 1
    upper = 2
    for branch_end in cc_be:
        coords = branch_end['coords']
        neighborhood_be = []
        for idx in coords:
            for x in range(-lower, upper):
                for y in range(-lower, upper):
                    for z in range(-lower, upper):
                        new_idx = [idx[0] + x, idx[1] + y, idx[2] + z]
                        neighborhood_be.append(new_idx)
        if (coords == coord_root).all(1).any():
            root_neighborhood.append(neighborhood_be)
        else:
            all_neighborhoods.append(neighborhood_be)
                    
    ### convert segments into just coords and ALSO get there neighborhoods for their FIRST and LAST indices
    labels = measure.label(only_segments)
    cc_segs = measure.regionprops(labels)
    
    all_hood_first_last = []          
    for seg in cc_segs:
        coords = seg['coords']                      
        all_hood_first_last.append(coords)
        
   
    """ Create tree """
    depth = 0               
        
    tree_df = treeify(tree_df, depth, cc_segs, root_neighborhood, all_neighborhoods, all_hood_first_last, cur_idx = 0, parent= -1, im=input_im)                 

    return tree_df



""" Create tree """
def treeify(tree_df, depth, cc_segs, root_neighborhood, all_neighborhoods, all_hood_first_last, cur_idx = 0, parent= -1, im=0):
        # IF ROOT (depth == 0) ==> then use root neighborhood
        if len(tree_df) == 0:
            cur_be = np.vstack(root_neighborhood[0])
        else:
            idx_parent_df = np.where(tree_df.cur_idx == parent)
            cur_be = np.vstack(tree_df.end_be_coord[idx_parent_df[0][0]])

       
        ### find next seg                    
        for idx_cur_seg in range(len(all_hood_first_last)):
            #if not all_hood_first_last[idx_cur_seg]:
                
            if not np.asarray(all_hood_first_last[idx_cur_seg]).any():
                
                continue   # skip if empty
            cur_seg = np.vstack(all_hood_first_last[idx_cur_seg])

            all_children = [];
            if (cur_seg[:, None] == cur_be).all(-1).any():
                
                if depth == 1:
                    print('match')
                
                if len(tree_df) > 0:
                    cur_idx = np.max(tree_df.cur_idx[:]) + 1;  


                full_seg_coords = np.vstack(cc_segs[idx_cur_seg]['coords'])

                new_node = {'coords': full_seg_coords, 'parent': parent, 'child': [], 'depth': depth, 'cur_idx': cur_idx, 'start_be_coord': cur_be, 'end_be_coord': []}
                tree_df = tree_df.append(new_node, ignore_index=True)
                
                ### find next be
                next_be = []; all_neighborhoods_tmp = np.copy(all_neighborhoods)
                
                isempty = 1
                for idx_cur_be in range(len(all_neighborhoods)):
                    if not all_neighborhoods[idx_cur_be]:
                        continue # skip if empty
                        
                    search_be = np.vstack(all_neighborhoods[idx_cur_be])
                    
                    if (cur_seg[:, None] == search_be).all(-1).any():
                                             
                        next_be.append(search_be)
                        
                        # delete the neighborhood we currently assessed
                        all_neighborhoods_tmp[idx_cur_be] = [];
                        isempty = 0
                        
                print(cur_idx)

                # delete the neighborhood we currently assessed
                #all_hood_first_last_tmp = list(np.copy(all_hood_first_last))
                all_hood_first_last[idx_cur_seg] = []
                
                if not isempty and np.asarray(np.vstack(next_be)).any():
                    next_be = np.vstack(next_be)
                    idx_parent_df = np.where(tree_df.cur_idx == cur_idx)
                    tree_df.end_be_coord[idx_parent_df[0][0]] = next_be
                                                
                    # recurse
                    tree_df = treeify(tree_df, depth + 1, cc_segs, 
                                          root_neighborhood, all_neighborhoods_tmp, all_hood_first_last, cur_idx=cur_idx, parent=cur_idx) 

                    # append children
                    all_children.append(cur_idx)
                    
                    # add all children
                    tree_df.child[cur_idx].append(all_children)


                
        return tree_df
    
    
""" Plot tree """
def show_tree(tree_df, template_im):
    all_segments = np.asarray(tree_df.coords[:])
    
    indices = np.asarray(tree_df.cur_idx)
    
    start_be_coord = np.asarray(tree_df.start_be_coord[:])
    
    
    end_be_coord = np.asarray(tree_df.end_be_coord[:])
    
    im = np.zeros(np.shape(template_im))
    for seg, ind, start_be, end_be in zip(all_segments, indices, start_be_coord, end_be_coord):
        
        im[seg[:, 0], seg[:, 1], seg[:, 2]] = ind + 1 # because dont want non-zero
        
        im[start_be[:, 0], start_be[:, 1], start_be[:, 2]] = 1
        
        if np.asarray(end_be).any():
            im[end_be[:, 0], end_be[:, 1], end_be[:, 2]] = 2
    return im