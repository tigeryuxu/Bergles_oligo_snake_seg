# -*- coding: utf-8 -*-
"""
Created on Sun Dec 31 16:16:39 2017

@author: Tiger
"""


import tensorflow as tf
import math
import pylab as mpl
import numpy as np
import time
import matplotlib.pyplot as plt
from PIL import Image
import random
#from skimage import measure



""" Plots generic training outputs """
def plot_trainer_3D(softMaxed, feed_dict_TRAIN, feed_dict_CROSSVAL, input_im, input_im_val, truth_im, truth_im_val,
                 weighted_labels, weighted_labels_val, s_path, epochs, plot_depth=0, multiclass=0):
       """ Plot for debug """
       feed_dict = feed_dict_TRAIN
       output_train = softMaxed.eval(feed_dict=feed_dict)
       seg_train = np.argmax(output_train, axis = -1)[-1]              
       
    
       feed_dict = feed_dict_CROSSVAL
       output_val = softMaxed.eval(feed_dict=feed_dict)
       seg_val = np.argmax(output_val, axis = -1)[-1]                  
       
       raw_truth = np.copy(truth_im)
       raw_truth_val = np.copy(truth_im_val)
       
       
       truth_im = np.argmax(truth_im, axis = -1)
       truth_im_val = np.argmax(truth_im_val, axis = -1)
       
       fig = plt.figure(num=3, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(4,2,1); plt.imshow(input_im[plot_depth, :, :, 0]); plt.title('Input image train');
       plt.subplot(4,2,3); plt.imshow(input_im[plot_depth, :, :, 1]); plt.title('Input image train seed');
       plt.subplot(4,2,5); plt.imshow(truth_im[plot_depth, :, :]); plt.title('Truth Train');
       plt.subplot(4,2,7); plt.imshow(seg_train[plot_depth, :, :]); plt.title('Output Train');
       
       plt.subplot(4,2,2); plt.imshow(input_im_val[plot_depth, :, :, 0]); plt.title('Input image val');
       plt.subplot(4,2,4); plt.imshow(input_im_val[plot_depth, :, :, 1]); plt.title('Input image val seed');
       plt.subplot(4,2,6); plt.imshow(truth_im_val[plot_depth, :, :]); plt.title('Truth val');
       plt.subplot(4,2,8); plt.imshow(seg_val[plot_depth, :, :]); plt.title('Output val');   
       
       plt.savefig(s_path + '_' + str(epochs) + '_output.png')
       
       
       """ Then plot all class info """
       fig = plt.figure(num=4, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       plt.subplot(4,2,3); plt.imshow(raw_truth[plot_depth, :, :, 1]); plt.title('Truth class 1: train');
       plt.subplot(4,2,5); plt.imshow(raw_truth[plot_depth, :, :, 0]); plt.title('Truth background');  
     
       plt.subplot(4,2,4); plt.imshow(raw_truth_val[plot_depth, :, :, 1]); plt.title('Truth class 1: val');
       plt.subplot(4,2,6); plt.imshow(raw_truth_val[plot_depth, :, :, 0]); plt.title('Truth background');  
 
       if multiclass:
            plt.subplot(4,2,1); plt.imshow(raw_truth[plot_depth, :, :, 2]); plt.title('Truth class 2: train');    #plt.pause(0.005)
            plt.subplot(4,2,2); plt.imshow(raw_truth_val[plot_depth, :, :, 2]); plt.title('Truth class 2: train');    #plt.pause(0.005)

      
       weighted_labels = np.amax(weighted_labels, axis = 0)
       weighted_labels_val = np.amax(weighted_labels_val, axis = 0)
       plt.subplot(4,2,7); plt.imshow(weighted_labels[:, :, 1]); plt.title('S_weight 1: train');
       plt.subplot(4,2,8); plt.imshow(weighted_labels_val[:, :, 1]); plt.title('S_weight 2: val');         
          
          
          
          
       plt.savefig(s_path + '_' + str(epochs) + '_output_class.png')
       
       
       
       
       """ Plot for max project evaluate """
       truth_im = np.amax(truth_im, axis= 0)
       truth_im_val = np.amax(truth_im_val, axis = 0)
       seg_train = np.amax(seg_train, axis = 0)
       seg_val = np.amax(seg_val, axis = 0)
       
       

       raw_truth = np.amax(raw_truth, axis = 0)
       raw_truth_val = np.amax(raw_truth_val, axis = 0)
       input_im = np.amax(input_im, axis = 0)
       input_im_val = np.amax(input_im_val, axis = 0)                                          
       


       fig = plt.figure(num=5, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       
       plt.subplot(4,2,1); plt.imshow(input_im[:, :, 0]); plt.title('Input image train');
       plt.subplot(4,2,3); plt.imshow(input_im[:, :, 1]); plt.title('Input image train seed');
       plt.subplot(4,2,5); plt.imshow(truth_im[:, :]); plt.title('Truth Train');
       plt.subplot(4,2,7); plt.imshow(seg_train[:, :]); plt.title('Output Train');
       
       plt.subplot(4,2,2); plt.imshow(input_im_val[:,  :, 0]); plt.title('Input image val');
       plt.subplot(4,2,4); plt.imshow(input_im_val[:, :, 1]); plt.title('Input image val seed');
       plt.subplot(4,2,6); plt.imshow(truth_im_val[:, :]); plt.title('Truth val');
       plt.subplot(4,2,8); plt.imshow(seg_val[:, :]); plt.title('Output val');   
       
       plt.savefig(s_path + '_' + str(epochs) + '_max_project_output.png')
       
       
       """ Then plot all class info """
       fig = plt.figure(num=6, figsize=(10, 10), dpi=80, facecolor='w', edgecolor='k')
       fig.subplots_adjust(hspace=.3)
       plt.subplot(4,2,1); plt.imshow(weighted_labels[:, :, 1]); plt.title('weighted train');    #plt.pause(0.005)
       plt.subplot(4,2,3); plt.imshow(raw_truth[:, :, 1]); plt.title('Truth class 1: train');
     
       plt.subplot(4,2,2); plt.imshow(weighted_labels_val[:, :, 1]); plt.title('weighted val');    #plt.pause(0.005)
       plt.subplot(4,2,4); plt.imshow(raw_truth_val[:, :, 1]); plt.title('Truth class 1: val');


       if multiclass:     
            plt.subplot(4,2,5); plt.imshow(raw_truth[:, :, 2]); plt.title('Truth class 2: train');    #plt.pause(0.005)       
            plt.subplot(4,2,6); plt.imshow(raw_truth_val[:, :, 2]); plt.title('Truth class 2: train');    #plt.pause(0.005)
       
       
       

""" ADDS TEXT TO IMAGE and saves the image """
def add_text_to_image(all_fibers, filename='default.png', resolution=800):
    #fiber_img = Image.fromarray((all_fibers *255).astype(np.uint16)) # ORIGINAL, for 8GB CPU
    fiber_img = (all_fibers*255).astype(np.uint16) 
    plt.figure(80, figsize=(12,10)); plt.clf(); plt.imshow(fiber_img)
    plt.axis('off')
    # PRINT TEXT ONTO IMAGE
    binary_all_fibers = all_fibers > 0
    labelled = measure.label(binary_all_fibers)
    cc_overlap = measure.regionprops(labelled, intensity_image=all_fibers)
    
    # Make a list of random colors corresponding to all the cells
    list_fibers = []
    for Q in range(int(np.max(all_fibers) + 1)):
        color = [random.randint(0,255)/256, random.randint(0,255)/256, random.randint(0,255)/256]
        list_fibers.append(color)
        
    for Q in range(len(cc_overlap)):
        overlap_coords = cc_overlap[Q]['coords']
        new_num = cc_overlap[Q]['MinIntensity']
        
        #if cell_num != new_num:
            #color = [random.randint(0,255)/256, random.randint(0,255)/256, random.randint(0,255)/256]
            #cell_num = new_num
        color = list_fibers[int(new_num)]
        plt.text(overlap_coords[0][1], overlap_coords[0][0], str(int(new_num)), fontsize= 2, color=color)    
    plt.savefig(filename, dpi = resolution)

"""
    Scales the normalized images to be within [0, 1], thus allowing it to be displayed
"""
def show_norm(im):
    m,M = im.min(),im.max()
    plt.imshow((im - m) / (M - m))
    plt.show()


""" Originally from Intro_to_deep_learning workshop
"""

def plotOutput(layer,feed_dict,fieldShape=None,channel=None,figOffset=1,cmap=None):
	# Output summary
	W = layer
	wp = W.eval(feed_dict=feed_dict);
	if len(np.shape(wp)) < 4:		# Fully connected layer, has no shape
		temp = np.zeros(np.product(fieldShape)); temp[0:np.shape(wp.ravel())[0]] = wp.ravel()
		fields = np.reshape(temp,[1]+fieldShape)
	else:			# Convolutional layer already has shape
		wp = np.rollaxis(wp,3,0)
		features, channels, iy,ix = np.shape(wp)   # where "features" is the number of "filters"
		if channel is not None:
			fields = wp[:,channel,:,:]
		else:
			fields = np.reshape(wp,[features*channels,iy,ix])    # all to remove "channels" axis

	perRow = int(math.floor(math.sqrt(fields.shape[0])))
	perColumn = int(math.ceil(fields.shape[0]/float(perRow)))
	fields2 = np.vstack([fields,np.zeros([perRow*perColumn-fields.shape[0]] + list(fields.shape[1:]))])    # adds more zero filters...
	tiled = []
	for i in range(0,perColumn*perRow,perColumn):
		tiled.append(np.hstack(fields2[i:i+perColumn]))    # stacks horizontally together ALL the filters

	tiled = np.vstack(tiled)    # then stacks itself on itself
	if figOffset is not None:
		mpl.figure(figOffset); mpl.clf(); 

	mpl.imshow(tiled,cmap=cmap); mpl.title('%s Output' % layer.name); mpl.colorbar();
    
    
""" Plot layers
"""
def plotLayers(feed_dict, L1, L2, L3, L4, L5, L6, L8, L9, L10):
      plt.figure('Down_Layers');
      plt.clf()
      plt.subplot(221); plotOutput(L1,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(222); plotOutput(L2,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(233); plotOutput(L3,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(234); plotOutput(L5,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(223); plotOutput(L4,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.pause(0.05)
      
      plt.figure('Up_Layers');
      plt.clf()
      plt.subplot(221); plotOutput(L6,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(222); plotOutput(L8,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(223); plotOutput(L9,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.subplot(224); plotOutput(L10,feed_dict=feed_dict,cmap='inferno',figOffset=None);
      plt.pause(0.05); 
      
  
""" Plots global and detailed cost functions
""" 
  
def plot_cost_fun(plot_cost, plot_cost_val, mov_avg_loss=None, plot_cost_val_NO=None):
      """ Graph global loss
      """      
      avg_window_size = 40
 
      if len(plot_cost) <= avg_window_size:
          mov_avg_loss = plot_cost
          mov_avg_loss_val = plot_cost_val
      else:
          mov_avg_loss = plot_cost[0:avg_window_size]    
          mov_avg_loss_val = plot_cost_val[0:avg_window_size]
     
          avg_loss = moving_average(plot_cost, n=avg_window_size).tolist()
          avg_loss_val = moving_average(plot_cost_val, n=avg_window_size).tolist()
          
          mov_avg_loss = mov_avg_loss + avg_loss
          mov_avg_loss_val = mov_avg_loss_val + avg_loss_val


      plt.figure(18); plt.clf();
      plt.plot(plot_cost, alpha=0.3, label='Training'); plt.title('Global Loss')
      if mov_avg_loss is not None:
           plt.plot(mov_avg_loss, color='tab:blue');
      plt.ylabel('Loss'); plt.xlabel('Iterations'); #plt.pause(0.05)
      plt.yscale('log')
      
      # cross-validation
      plt.figure(18); plt.plot(plot_cost_val, alpha=0.3, label='Validation'); #plt.pause(0.05)
      if mov_avg_loss_val is not None:
           plt.plot(mov_avg_loss_val, color='tab:orange');
      plt.legend(loc='upper left');    
      plt.yscale('log')


      plt.figure(25); plt.clf();
      plt.plot(plot_cost, alpha=0.3, label='Training'); plt.title('Global Loss')
      if mov_avg_loss is not None:
           plt.plot(mov_avg_loss, color='tab:blue');
      plt.ylabel('Loss'); plt.xlabel('Iterations'); #plt.pause(0.05)
      plt.xscale('log')
      plt.yscale('log')
      
      # cross-validation
      plt.figure(25); plt.plot(plot_cost_val, alpha=0.3, label='Validation'); #plt.pause(0.05)
      if mov_avg_loss_val is not None:
           plt.plot(mov_avg_loss_val, color='tab:orange');
      plt.legend(loc='upper left');    
      plt.xscale('log')
      plt.yscale('log')
      
      
      """ Graph detailed plot
      """
      last_loss = len(plot_cost)
      start = 0
      if last_loss < 50:
          start = 0
      elif last_loss < 200:
          start = last_loss - 50
          
      elif last_loss < 500:
          start = last_loss - 200
          
      elif last_loss < 1500:
          start = last_loss - 500
          
      elif last_loss < 10000:
          start = last_loss - 1500 
      else:
          start = last_loss - 8000
      plt.close(19);
      x_idx = list(range(start, last_loss))
      plt.figure(19); plt.plot(x_idx,plot_cost[start:last_loss], alpha=0.3, label='Training'); plt.title("Detailed Loss"); 
      plt.figure(19); plt.plot(x_idx,plot_cost_val[start:last_loss], alpha=0.3, label='Validation');
      plt.legend(loc='upper left');             
      plt.ylabel('Loss'); plt.xlabel('Iterations'); #plt.pause(0.05) 
      #plt.xscale('log')
      plt.yscale('log')

      
      if plot_cost_val_NO is not None:
            plt.figure(18); plt.plot(plot_cost_val_NO, label='Cross_validation_NO'); #plt.pause(0.05)                                      
            plt.figure(19); plt.plot(x_idx, plot_cost_val_NO[start:last_loss], label='Validation_NO');   #plt.pause(0.05)    
      
        
""" Plots global and detailed cost functions
""" 
  
def plot_jaccard_fun(plot_jaccard, plot_jaccard_val=False, class_name=''):
      
     avg_window_size = 40
 
     if len(plot_jaccard) <= avg_window_size:
          mov_avg_jacc = plot_jaccard
          mov_avg_jacc_val = plot_jaccard_val
     else:
          mov_avg_jacc = plot_jaccard[0:avg_window_size]    
          mov_avg_jacc_val = plot_jaccard_val[0:avg_window_size]
     
          avg_jacc = moving_average(plot_jaccard, n=avg_window_size).tolist()
          avg_jacc_val = moving_average(plot_jaccard_val, n=avg_window_size).tolist()
          
          mov_avg_jacc = mov_avg_jacc + avg_jacc
          mov_avg_jacc_val = mov_avg_jacc_val + avg_jacc_val
                

     """ Graph global jaccard
     """      
     plt.figure(21); plt.clf();
     plt.plot(plot_jaccard, alpha=0.3, label='Jaccard' + class_name); plt.title('Jaccard')  
     plt.plot(mov_avg_jacc, color='tab:blue');

     if plot_jaccard_val:
         plt.plot(plot_jaccard_val, alpha=0.3, label='Validation Jaccard' + ' ' + class_name);
         plt.plot(mov_avg_jacc_val, color='tab:orange');

     plt.ylabel('Jaccard'); plt.xlabel('Iterations');            
     plt.legend(loc='upper left');    #plt.pause(0.05)
      
      


""" Easier moving average calculation """
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n
  
      
def plot_overlay(plot_cost, plot_cost_val, plot_jaccard, plot_cost_val_NO=None):
      """ Graph global loss
      """      
      plt.figure(18); 
      
      #plt.clf();
      plt.plot(plot_cost, label='Training_NO_W'); plt.title('Global Loss')
      plt.ylabel('Loss'); plt.xlabel('Iterations'); plt.pause(0.05)
      
      # cross-validation
      plt.figure(18); plt.plot(plot_cost_val, label='Cross_validation_NO_W'); plt.pause(0.05)
      plt.legend(loc='upper left');    
      
      """ Graph detailed plot
      """
      last_loss = len(plot_cost)
      start = 0
      if last_loss < 50:
          start = 0
      elif last_loss < 200:
          start = last_loss - 50
          
      elif last_loss < 500:
          start = last_loss - 200
          
      elif last_loss < 1500:
          start = last_loss - 500
          
      else:
          start = last_loss - 1500
      
      #plt.close(19);
      x_idx = list(range(start, last_loss))
      plt.figure(19); plt.plot(x_idx,plot_cost[start:last_loss], label='Training_NO_W'); plt.title("Detailed Loss"); 
      plt.figure(19); plt.plot(x_idx,plot_cost_val[start:last_loss],label='Cross_validation_NO_W');
      plt.legend(loc='upper left');             
      plt.ylabel('Loss'); plt.xlabel('Iterations'); plt.pause(0.05)    
      
      if plot_cost_val_NO is not None:
            plt.figure(18); plt.plot(plot_cost_val_NO, label='Cross_validation_NO'); plt.pause(0.05)                                      
            plt.figure(19); plt.plot(x_idx, plot_cost_val_NO[start:last_loss], label='Cross_validation_NO');   plt.pause(0.05)    
      
    
      plt.figure(21); 
      
      #plt.clf();
      plt.plot(plot_jaccard, label='Jaccard_NO_W'); plt.title('Jaccard')
      plt.ylabel('Jaccard'); plt.xlabel('Iterations'); 
      plt.legend(loc='upper left');    plt.pause(0.05)




    
    
""" Plots the moving average that is much smoother than the overall curve"""
    
def calc_moving_avg(plot_data, num_pts = 20, dist_points=100):
    
    new_plot = []
    for T in range(0, len(plot_data)):
        
        avg_points = []
        for i in range(-dist_points, dist_points):
            
            if T + i < 0:
                continue;
            elif T + i >= len(plot_data):
                break;
            else:
                avg_points.append(plot_data[T+i])
                
        mean_val = sum(avg_points)/len(avg_points)
        new_plot.append(mean_val)
        
    return new_plot
            
        
    
def change_scale_plot():

    multiply = 1000
    font_size = 11
    legend_size = 11
    plt.rcParams.update({'font.size': 9})    
    """Getting back the objects"""
    plot_cost = load_pkl(s_path, 'loss_global.pkl')
    plot_cost_val = load_pkl(s_path, 'loss_global_val.pkl')
    plot_jaccard = load_pkl(s_path, 'jaccard.pkl')


    x_idx = list(range(0, len(plot_cost) * multiply, multiply));   
    plt.figure(19); plt.plot(x_idx,plot_cost, label='Training_weighted'); 
    #plt.title("Detailed Loss"); 
    plt.figure(19); plt.plot(x_idx,plot_cost_val,label='Validation_weighted');
    plt.legend(loc='upper right');             
    plt.ylabel('Loss', fontsize = font_size); plt.xlabel('Iterations', fontsize = font_size); plt.pause(0.05) 
    
    x_idx = list(range(0, len(plot_jaccard) * multiply, multiply));   
    plt.figure(20); plt.plot(x_idx,plot_jaccard, label='Validation_weighted'); 
    #plt.title("Detailed Loss");     
    plt.ylabel('Jaccard', fontsize = font_size); plt.xlabel('Iterations', fontsize = font_size); plt.pause(0.05) 
    plt.legend(loc='upper left');             
    
    """Getting back the objects"""
    plot_cost_noW = load_pkl(s_path, 'loss_global_no_W.pkl')
    plot_cost_val_noW = load_pkl(s_path, 'loss_global_val_no_W.pkl')
    plot_jaccard_noW = load_pkl(s_path, 'jaccard_no_W.pkl')
    

    x_idx = list(range(0, len(plot_cost_noW) * multiply, multiply));   
    plt.figure(19); plt.plot(x_idx,plot_cost_noW, label='Training_no_weight'); 
    #plt.title("Loss"); 
    plt.figure(19); plt.plot(x_idx,plot_cost_val_noW,label='Validation_no_weight');
    plt.legend(loc='upper right', prop={'size': legend_size});             

    
    x_idx = list(range(0, len(plot_jaccard_noW) * multiply, multiply));   
    plt.figure(20); plt.plot(x_idx,plot_jaccard_noW, label='Validation_no_weight'); 
    #plt.title("Jaccard");     
    plt.legend(loc='upper left', prop={'size': legend_size});      
    
    """ Calculate early stopping beyond 180,000 """
    plot_short = plot_cost_val[30000:-1]
    hist_loss = plot_short
    patience_cnt = 0    
    for epoch in range(len(plot_short)):
        # ... 
        # early stopping

        patience = 100
        min_delta = 0.02
        if epoch > 0 and hist_loss[epoch-1] - hist_loss[epoch] > min_delta:
            patience_cnt = 0
        else:
            patience_cnt += 1
     
        if patience_cnt > patience:
            print("early stopping...")
            print(epoch * 5 + 30000 * 5)
            break
    """ 204680 """
    
    """ MOVING AVERAGE """
    num_pts = 10
    dist_points = 20
    mov_cost = calc_moving_avg(plot_cost, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val = calc_moving_avg(plot_cost_val, num_pts=num_pts, dist_points=dist_points)
    mov_jaccard = calc_moving_avg(plot_jaccard, num_pts=num_pts, dist_points=dist_points)

    
    font_size = 11
    plt.rcParams.update({'font.size': 10})    

    x_idx = list(range(0, len(mov_cost) * multiply, multiply));   
    plt.figure(21); plt.plot(x_idx,mov_cost, label='Training_weighted'); plt.title("Detailed Loss"); 
    plt.figure(21); plt.plot(x_idx,mov_cost_val,label='Validation_weighted');
    plt.legend(loc='upper left');             
    plt.ylabel('Loss', fontsize = font_size); plt.xlabel('Iterations', fontsize = font_size); plt.pause(0.05) 
    
    x_idx = list(range(0, len(mov_jaccard) * multiply, multiply));   
    plt.figure(22); plt.plot(x_idx,mov_jaccard, label='Validation_weighted'); plt.title("Detailed Jaccard");     
    plt.ylabel('Jaccard', fontsize = font_size); plt.xlabel('Iterations', fontsize = font_size); plt.pause(0.05) 
    plt.legend(loc='upper left');             
    
    """Getting back the objects"""
    num_pts = 10
    dist_points = 400
    mov_cost_noW = calc_moving_avg(plot_cost_noW, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val_noW = calc_moving_avg(plot_cost_val_noW, num_pts=num_pts, dist_points=dist_points)
    mov_jaccard_noW = calc_moving_avg(plot_jaccard_noW, num_pts=num_pts, dist_points=dist_points)


    x_idx = list(range(0, len(mov_cost_noW) * multiply, multiply));   
    plt.figure(21); plt.plot(x_idx,mov_cost_noW, label='Training_no_weight'); plt.title("Loss"); 
    plt.figure(21); plt.plot(x_idx,mov_cost_val_noW,label='Validation_no_weight');
    plt.legend(loc='upper left');             

    
    x_idx = list(range(0, len(mov_jaccard_noW) * multiply, multiply));   
    plt.figure(22); plt.plot(x_idx,mov_jaccard_noW, label='Validation_no_weight'); plt.title("Jaccard");     
    plt.legend(loc='upper left');      
    
    

""" Plot the average for the NEWEST MyQz11 + ClassW + No_W"""

def change_scale_plot2():

    #s_path = 'C:/Users/Tiger/Anaconda3/AI stuff/MyelinUNet_new/Checkpoints/ALL_FOR_PLOT/'
    
    s_path = 'D:/Tiger/AI stuff/MyelinUNet/Checkpoints/ALL_FOR_PLOT/' 

    num_pts = 10
    multiply = 10
    font_size = 11
    legend_size = 11
    plt.rcParams.update({'font.size': 9})    
    
    """Getting back the objects"""
    #plot_cost = load_pkl(s_path, 'loss_global.pkl')
    plot_cost_val_noW = load_pkl(s_path, 'loss_global_sW_1_rotated.pkl')
    plot_jaccard_noW = load_pkl(s_path, 'jaccard_sW_1_rotated.pkl')

    """Getting back the objects"""
    #plot_cost_noW = load_pkl(s_path, 'loss_global_no_W.pkl')
    plot_cost_val = load_pkl(s_path, 'loss_global_MyQ_2_not_rotated.pkl')
    plot_jaccard = load_pkl(s_path, 'jaccard_MyQ_2_not_rotated.pkl')


    """Getting back the objects"""
    ##plot_cost_noW = load_pkl(s_path, 'loss_global_no_W.pkl')
    #plot_cost_val_sW = load_pkl(s_path, 'loss_global_MyQz11_sW_batch2.pkl')
    #plot_jaccard_sW = load_pkl(s_path, 'jaccard_MyQz11_sW_batch2.pkl')


    font_size = 11
    plt.rcParams.update({'font.size': 10})  
    
    """ no-weight """
    dist_points_loss = 3
    dist_points_jacc = 25
    multiply = 1000
    #mov_cost_noW = calc_moving_avg(plot_cost_noW, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val_noW = calc_moving_avg(plot_cost_val_noW, num_pts=num_pts, dist_points=dist_points_loss)
    mov_jaccard_noW = calc_moving_avg(plot_jaccard_noW, num_pts=num_pts, dist_points=dist_points_jacc)
       
    plot_single_cost(mov_cost_val_noW, multiply, 'Validation rotated', 'Loss')    
    plot_single_jacc(mov_jaccard_noW, multiply, 'Validation rotated', 'Jaccard')

    
    """ class weight """
    multiply = 1000
    #mov_cost = calc_moving_avg(plot_cost, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val = calc_moving_avg(plot_cost_val, num_pts=num_pts, dist_points=dist_points_loss)
    mov_jaccard = calc_moving_avg(plot_jaccard, num_pts=num_pts, dist_points=dist_points_jacc) 
           
    plot_single_cost(mov_cost_val[0:400], multiply, 'Validation no rotate', 'Loss')    
    plot_single_jacc(mov_jaccard[0:400], multiply, 'Validation no rotate', 'Jaccard')
    

    """ spatial W """
    multiply = 1000
    #mov_cost_noW = calc_moving_avg(plot_cost_noW, num_pts=num_pts, dist_points=dist_points)
    mov_cost_val_noW = calc_moving_avg(plot_cost_val_sW, num_pts=num_pts, dist_points=dist_points_loss)
    mov_jaccard_noW = calc_moving_avg(plot_jaccard_sW, num_pts=num_pts, dist_points=dist_points_jacc)
       
    plot_single_cost(mov_cost_val_noW, multiply, 'Validation spatial weight', 'Loss')    
    plot_single_jacc(mov_jaccard_noW, multiply, 'Validation spatial weight', 'Jaccard')




def plot_single_cost(data, multiply, label, title):
    x_idx = list(range(0, len(data) * multiply, multiply));   
    plt.figure(21); plt.plot(x_idx,data, label=label); plt.title(title);     
    plt.legend(loc='upper left');     
    
def plot_single_jacc(data, multiply, label, title):
    x_idx = list(range(0, len(data) * multiply, multiply));   
    plt.figure(22); plt.plot(x_idx,data, label=label); plt.title(title);     
    plt.legend(loc='upper right');     
    