# -*- coding: utf-8 -*-
"""
Created on Sunday Dec. 24th
============================================================
@author: Tiger
### TO DO:
    # (4) paranodes      
      
    Different trainings:
        ### ***larger model on COMPUTE CANADA  
        ### ***exclude 1st image   
        ### ***seed every 5
    
"""
import matplotlib
matplotlib.rc('xtick', labelsize=8) 
matplotlib.rc('ytick', labelsize=8) 
matplotlib.use('Qt5Agg')
matplotlib.use('Agg')

""" Libraries to load """
import numpy as np
from functional.plot_functions_CLEANED import *
from functional.data_functions_CLEANED import *
from functional.data_functions_3D import *
from functional.IO_func import *
import glob, os
import datetime
import time
from natsort import natsort_keygen, ns
natsort_key1 = natsort_keygen(key = lambda y: y.lower())      # natural sorting order

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from layers.UNet_pytorch import *
from layers.UNet_pytorch_online import *
from sklearn.model_selection import train_test_split

from matlab_crop_function import *
from off_shoot_functions import *
from tree_functions import *
from skimage.morphology import skeletonize_3d, skeletonize





from skimage.transform import rescale, resize, downscale_local_mean


torch.backends.cudnn.benchmark = True  
torch.backends.cudnn.enabled = True 



""" Define GPU to use """
import torch
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print(device)

""" Decide if use pregenerated seeds or not """
pregenerated = 1
        
"""  Network Begins: """
#check_path = './(48) Checkpoint_nested_unet_SPATIALW_medium_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/'
#check_path = './(49) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/'; dilation = 1; deep_supervision = False;
check_path = './(49) Checkpoint_nested_unet_SPATIALW_COMPLEX_b4_NEW_DATA_SWITCH_NORM_crop_pad_Haussdorf_balance/train with 1e6 after here/'; dilation = 1; deep_supervision = False;

s_path = check_path + 'TEST_inference/'
try:
    # Create target Directory
    os.mkdir(s_path)
    print("Directory " , s_path ,  " Created ") 
except FileExistsError:
    print("Directory " , s_path ,  " already exists")

input_path = '/media/user/storage/Data/(1) snake seg project/Traces files/seed generation large_25px_NEW/'
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
mean_arr = check['mean_arr'];  std_arr = check['std_arr']
""" Set to eval mode for batch norm """
unet.eval();   unet.to(device)

input_size = 80
depth = 32
crop_size = int(input_size/2)
z_size = depth


""" Change to scaling per crop??? """
original_scaling = 0.2076;
target_scale = 0.20;
scale_factor = original_scaling/target_scale;
scaled_crop_size = round(input_size/scale_factor);
scaled_crop_size = math.ceil(scaled_crop_size / 2.) * 2  ### round up to even num

scale_for_animation = 0

for i in range(len(examples)):              

        """ (1) Loads data as sorted list of seeds """
        sorted_list, input_im, width_tmp, height_tmp, depth_tmp, overall_coord, all_seeds, all_seeds_no_50 = load_input_as_seeds(examples, im_num=i, pregenerated=pregenerated, s_path=s_path)   

        input_name = examples[i]['input']
        filename = input_name.split('/')[-1].split('.')[0:-1]
        filename = '.'.join(filename)
        
       
        """ scale input im for animations """
        if scale_for_animation:
             input_im_rescaled = convert_matrix_to_multipage_tiff(input_im)   
             
            
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
        all_trees = []
        for root in sorted_list:
            tree_df, children = get_tree_from_im_list(root, input_im, width_tmp, height_tmp, depth_tmp, all_coords_root)                                          
            
            ### HACK: continue if empty
            if len(tree_df) == 0:
                continue;
            tmp_tree_im = np.zeros(np.shape(input_im))
            #im = show_tree(tree_df, tmp_tree_im)
            #plot_max(im, ax=-1)                
                        
            ### set "visited" to correct value
            for idx, node in tree_df.iterrows():
                # if not isListEmpty(node.child):
                #     node.visited = 1
                # else:
                    node.visited = np.nan            
            # append to all trees
            all_trees.append(tree_df)

 


        track_trees = np.zeros(np.shape(input_im))    
        num_tree = 0

        """ Create box in middle to only get seed colocalized with middle after splitting branchpoints """
        center_cube = create_cube_in_im(width=10, input_size=input_size, z_size=z_size)
        small_center_cube = create_cube_in_im(width=10, input_size=input_size, z_size=z_size)
                
        for tree in all_trees:
             """ Keep looping until everything has been visited """  
             iterator = 0    
             while np.asarray(tree.visited.isnull()).any():   
                

                  """ Get coords at node """                  
                  unvisited_indices = np.where(tree.visited.isnull() == True)[0]
                  node_idx = unvisited_indices[-1]
                  cur_coords, cur_be_start, cur_be_end, centroid, parent_coords = get_next_coords(tree, node_idx, num_parents=4)
                                      
                  """ Order coords """
                  ### SKIP IF TOO SHORT for mini-steps
                  if len(cur_coords) == 1:
                     tree.visited[node_idx] = 1; iterator += 1; continue;
                  else:
                    cur_coords = order_coords(cur_coords)   ### ***order the points into line coordinates

                  """ Split into mini-steps """
                  ### Step size:
                  step_size = 10; step_size_first = step_size          
                  if len(cur_coords) <= step_size:  ### KEEP GOING IF ONLY SMALL SEGMENT
                      step_size_first = 0
                
                
                  for step in range(step_size_first, len(cur_coords), step_size):    
                      x = int(cur_coords[step, 0]); y = int(cur_coords[step, 1]); z = int(cur_coords[step, 2])
                      
                      cur_seg_im = np.zeros(np.shape(input_im))   # maybe speed up here by not creating the image every time???
                      cur_seg_im[cur_coords[0:step, 0], cur_coords[0:step, 1], cur_coords[0:step, 2]] = 1
                      cur_seg_im[x, y, z] = 1    # add the centroid as well
                      
                      # add the parent
                      if len(parent_coords) > 0:
                          cur_seg_im[parent_coords[:, 0], parent_coords[:, 1], parent_coords[:, 2]] = 1
    
                      """ use centroid of object to make seed crop """
                      crop, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max, boundaries_crop = crop_around_centroid_with_pads(input_im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                
                      #cur_seg_im[x,y,z] = 2   ### IF WANT TO SEE WHAT THE CROP IS CENTERING ON
                      crop_seed, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max, boundaries_crop = crop_around_centroid_with_pads(cur_seg_im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                      
                      """ Dilate the seed by sphere 1 to mimic training data """
                      crop_seed = dilate_by_ball_to_binary(crop_seed, radius=dilation)
        
                      """ Check nothing hanging off edges in seed  """
                      crop_seed = check_resized(crop_seed, depth, width_max=input_size, height_max=input_size)


                      """ Send to segmentor for INFERENCE """
                      crop_seed[crop_seed > 0] = 255  
                      output_PYTORCH = UNet_inference_PYTORCH(unet,np.asarray(crop, np.float32), crop_seed, mean_arr, std_arr, device=device, deep_supervision=deep_supervision)
            
                      """ Since it's centered around crop, ensure doesn't go overboard """
                      output_PYTORCH[boundaries_crop == 0] = 0
                
                      """ SAVE max projections"""
                      plot_save_max_project(fig_num=5, im=crop_seed, max_proj_axis=-1, title='crop seed dilated', 
                                            name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(2)_seed.png', pause_time=0.001)
                      plot_save_max_project(fig_num=2, im=output_PYTORCH, max_proj_axis=-1, title='segmentation', 
                                            name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '(3)_segmentation.png', pause_time=0.001)
                      plot_save_max_project(fig_num=3, im=crop, max_proj_axis=-1, title='input', 
                                            name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) + '_step_' + str(step) +  '_(1)_input_im.png', pause_time=0.001)



                      """ Things to fix still:
                               ***circle instead of cube subtraction??? ==> b/c creating bad cut-offs right now
                          """
    
                      """ REMOVE EDGE """
                      dist_xy = 0; dist_z = 0
                      edge = np.zeros(np.shape(output_PYTORCH)).astype(np.int64)
                      edge[dist_xy:crop_size * 2-dist_xy, dist_xy:crop_size * 2-dist_xy, dist_z:z_size-dist_z] = 1
                      edge = np.where((edge==0)|(edge==1), edge^1, edge)
                      output_PYTORCH[edge == 1] = 0
    
                      """ ***FIND anything that has previously been identified
                          ***EXCLUDING CURRENT CROP_SEED
                      """
                      im = show_tree(tree, track_trees)
                                        
                      ### or IN ALL PREVIOUS TREES??? *** can move this do beginning of loop
                      for cur_tree in all_trees:
                          im += show_tree(cur_tree, track_trees)
                      
                      im[im > 0] = 1
                      crop_prev, box_x_min, box_x_max, box_y_min, box_y_max, box_z_min, box_z_max, boundaries_crop = crop_around_centroid_with_pads(im, y, x, z, crop_size, z_size, height_tmp, width_tmp, depth_tmp)                                                      
                      crop_prev = skeletonize_3d(crop_prev)
                                        
                      ### EXCLUDE current crop seed
                      im_sub = subtract_im_no_sub_zero(crop_prev, crop_seed)
                      im_dil = dilate_by_ball_to_binary(im_sub, radius=3)
                      
                      ### but add back in current crop seed (so now without dilation)
                      im_dil = im_dil + crop_seed
                      im_dil[im_dil > 0] = 1
                      
    
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
                      output_PYTORCH, output_non_bin = bridge_end_points(output_PYTORCH, bridge_radius=2)
                      plot_save_max_project(fig_num=3, im=output_non_bin, max_proj_axis=-1, title='output_be', 
                                            name=s_path + filename + '_Crop_' + str(num_tree) + '_' + str(iterator) +  '_step_' + str(step) +'_(4)_output_be.png', pause_time=0.001)                                              
                          
                      # (1) use old start_coords to find only nearby segments           
                      # ***or just use center cube
                      coloc_with_center = output_PYTORCH + center_cube
                      only_coloc = find_overlap_by_max_intensity(bw=output_PYTORCH, intensity_map=coloc_with_center) 
                          
                      """ moved here: subtract out past identified regions LAST to not prevent propagation """
                      sub_seed = subtract_im_no_sub_zero(only_coloc, crop_seed)
                      sub_seed = subtract_im_no_sub_zero(sub_seed, im_dil)
    
                      """ skip if everything was subtracted out last time: """
                      if np.count_nonzero(sub_seed) < 5:
                              tree.visited[node_idx] = 1; print('Finished')                     
                              plot_save_max_project(fig_num=10, im=np.zeros(np.shape(only_coloc)), max_proj_axis=-1, title='_final_added', 
                                      name=s_path + filename + '_Crop_'   + str(num_tree) + '_' + str(iterator) +  '_step_' + str(step) + '_(6)_zfinal_added.png', pause_time=0.001) 
                              iterator += 1
                              continue                      
    
                      else:
    
                          # (2) skeletonize the output to create "all_neighborhoods" and "all_hood_first_last"                                    
                          degrees, coordinates = bw_skel_and_analyze(only_coloc)
                          """ insert the current be neighborhood at center and set to correct index in "degrees"
                                  or add in the endpoint like I just said, but subtract it out to keep the crop_seed separate???                              
                          """
                          cur_end = np.copy(cur_be_end)
                          cur_end = scale_coords_of_crop_to_full(cur_end, -box_x_min, -box_y_min, -box_z_min)
                          
                          ### check limits to ensure doesnt go out of frame
                          cur_end = check_limits([cur_end], crop_size * 2, crop_size * 2, depth)[0]
                          
                          ### HACK: fix how so end points cant leave frame
                          """ MIGHT GET TOO LARGE b/c of building up previous end points, so need to ensure crop """
                          cur_end[np.where(cur_end[:, 0] >= crop_size * 2), 0] = crop_size * 2 - 1
                          cur_end[np.where(cur_end[:, 1] >= crop_size * 2), 1] = crop_size * 2 - 1
                          cur_end[np.where(cur_end[:, 2] >= depth), 2] = depth - 1
                          
                          ### Then set degrees
                          degrees[cur_end[:, 0], cur_end[:, 1], cur_end[:, 2]] = 4
                          
                          ###remove all the others that match this first one???
                          plot_save_max_project(fig_num=9, im=degrees, max_proj_axis=-1, title='segmentation_deleted', 
                                      name=s_path + filename + '_Crop_' + str(num_tree) + '_' + str(iterator)  + '_step_' + str(step) + '_(5)_segmentation_deleted.png', pause_time=0.001) 
                          
                          all_neighborhoods, all_hood_first_last, root_neighborhood = get_neighborhoods(degrees, coord_root=0, scale=1, box_x_min=box_x_min, box_y_min=box_y_min, box_z_min=box_z_min, order=1)
    

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
                                 cur = scale_coords_of_crop_to_full(cur, -box_x_min, -box_y_min, -box_z_min)
                                 
                                 ### check limits to ensure doesnt go out of frame
                                 cur = check_limits([cur], crop_size * 2, crop_size * 2, depth)[0]
                                 
                                 check_debug[cur[:, 0], cur[:, 1], cur[:, 2]] = 1
                           
                             idx += 1

                          ### HACK: if can't find root_neighborhood, then skip
                          if len(root_neighborhood) == 0:
                               tree.visited[node_idx] = 1;
                               print('NO ROOT')
                               iterator += 1
                              
                               continue;      

                             
                          # delete cur_segs *** NOT NEEDED
                          idx = 0
                          for cur_seg in all_hood_first_last:
                                if (cur_seg[:, None] == tree.coords[node_idx]).all(-1).any():
                                      all_hood_first_last[idx] = []
                                else:
                                   cur = np.copy(cur_seg)
                                   cur = scale_coords_of_crop_to_full(cur, -box_x_min, -box_y_min, -box_z_min)
                                   
                                   ### check limits to ensure doesnt go out of frame
                                   cur = check_limits([cur], crop_size * 2, crop_size * 2, depth)[0]
                                   check_debug[cur[:, 0], cur[:, 1], cur[:, 2]] = 1                 
                                
                                idx += 1
                         
                          plot_save_max_project(fig_num=10, im=check_debug, max_proj_axis=-1, title='_final_added', 
                                      name=s_path + filename + '_Crop_'  + str(num_tree) + '_' + str(iterator)  + '_step_' + str(step) + '_(6)_zfinal_added.png', pause_time=0.001) 
                          
    
                          """ IF is empty (no following part) """
                          if len(all_neighborhoods) == 0:
                              tree.visited[node_idx] = 1;
                              print('Finished'); iterator += 1; continue
                          
                          else:
                              """ else add to tree """
                              cur_idx = tree.cur_idx[node_idx]
                              depth_tree = tree.depth[node_idx] + 1
                              tree = tree
                              #parent =  tree.parent[node_idx]
                              #root_neighborhood = all_neighborhoods[0]
                              
                              tree, cur_childs = treeify(tree, depth_tree, root_neighborhood, all_neighborhoods, all_hood_first_last, cur_idx = cur_idx, parent= cur_idx,
                                                    start=1, width_tmp=width_tmp, height_tmp=height_tmp, depth_tmp=depth_tmp)
                              
                              tree.child[cur_idx].append(cur_childs)
                              
                              
                              
                              ### set "visited" to correct value
                              # for idx, node in tree.iterrows():
                              #       if not isListEmpty(node.child):
                              #           node.visited = 1
                              #       elif not node.visited:
                              #           node.visited1 = np.nan    
                              
                              
                              """ set parent is visited to true """
                              tree.visited[node_idx] = 1;
                              
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
                                            
                
               
             """ Add expanded tree back into all_trees """
             all_trees[num_tree] = tree
               
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
           
        all_trees_copy = all_trees.copy()           
        all_starting_indices = [];
        idx = 0;
        for tree in all_trees:
        
            """ first clean up parent/child associations """
            for index, vertex in tree.iterrows():
                 cur_idx = vertex.cur_idx
                 children = np.where(tree.parent == cur_idx)
                 
                 vertex.child = children[0]
                                  
            if idx == 0:
                all_trees_appended = all_trees[0]
                all_starting_indices.append(0)
                idx += 1
                continue
                 
            tree.child = tree.child + len(all_trees_appended) 
            tree.parent = tree.parent + len(all_trees_appended) 
            tree.cur_idx = tree.cur_idx + len(all_trees_appended) 
            
            all_trees_appended = all_trees_appended.append(tree, ignore_index=True)
            
            all_starting_indices.append(len(all_trees_appended))
            
            idx += 1       

        """ Saves tree as swc file """ 
        save_tree_to_swc(all_trees_appended, s_path, filename = filename + 'output.swc', scale_xy=0.20756792660398113, scale_z=1)

        ### also save each individual tree???
        for counter, tree in enumerate(all_trees):
            save_tree_to_swc(tree, s_path, filename = filename + '_' + str(counter) + '_output.swc', scale_xy=0.20756792660398113, scale_z=1)
        
        """ Save tree as obj file """
        save_tree_to_obj(all_trees, s_path, filename = filename + 'output.obj')

        ### also save individual trees
        for counter, tree in enumerate(all_trees):
            save_tree_to_obj([tree], s_path, filename = filename + '_' + str(counter) + '_output.obj')


        """ *** PRIOR to this final step, also make sure to combine all non-branched together to make less vertices!!!         
                /media/user/storage/Data/(1) snake seg project/Traces files/swc files                
                ***also combine all .swc into one single file?
        
        """

        zzz
        
        
        
        
        
        
        
    