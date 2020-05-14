# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 22:53:34 2018

Initializes cost function and weights

@author: Tiger
"""

import tensorflow as tf
from matplotlib import *
import numpy as np
import scipy
import math

from plot_functions import *
from data_functions import *
#from post_process_functions import *
from UNet import *


""" Perceptual loss training: defined as MS-SSIM + L1 norm (MAE) """
def costOptm_MSSSIM_MAE(y, y_b, logits, train_rate=1e-5, epsilon = 1e-8, optimizer='adam', loss_function='MSSIM-MAE'):
     
    """ MAE loss """
    mae = tf.keras.losses.MeanAbsoluteError()
    loss_mae = mae(y_b, y)
    
    """ MS-SSIM loss *** NOTE: filter_size must be < 8 in order to work for 64 x 64 images 
       ***also get single value across all batches???
       ***MUST BE * -1 in order for correct MSSIM loss to be calculated
    """
    loss_ms_ssim_across_batch = tf.image.ssim_multiscale(y_b, y, max_val = 1, filter_size = 4)
    loss_ms_ssim = tf.reduce_mean(loss_ms_ssim_across_batch * -1)
    
    loss = [];
    if loss_function == 'MAE':
         loss = loss_mae
    elif loss_function == 'MSSIM':  
         loss = loss_ms_ssim
    elif loss_function == 'MSSIM-MAE':
         loss = loss_mae + loss_ms_ssim
                        

    """ Define optimizer """
    if optimizer == 'adam':
         train_step = tf.train.AdamOptimizer(learning_rate=train_rate, epsilon=epsilon).minimize(loss)  # train_step uses Adam Optimizer
    elif optimizer == 'SGD':
         train_step = tf.train.GradientDescentOptimizer(learning_rate=train_rate).minimize(loss)  # train_step uses Gradient Descent Optimizer
         
    return loss_mae, loss_ms_ssim, train_step, loss
    
    
    
    
""" Spatial Weighting. Weights the training loss such that it decays 
    exponentially with distance from the margins of the ensheathed segments 
    of interest by computing the Chebyshev distance transform of the inverted 
    ground-truth mask 
"""
def spatial_weight(y_,edgeFalloff=10,background=0.01,approximate=True):
    if approximate:   # does chebyshev
        dist1 = scipy.ndimage.distance_transform_cdt(y_)
        dist2 = scipy.ndimage.distance_transform_cdt(numpy.where(y_>0,0,1))    # sets everything in the middle of the OBJECT to be 0
                
    else:   # does euclidean
        dist1 = scipy.ndimage.distance_transform_edt(y_, sampling=[1,1,1])
        dist2 = scipy.ndimage.distance_transform_edt(numpy.where(y_>0,0,1), sampling=[1,1,1])
        
    """ DO CLASS WEIGHTING instead of spatial weighting WITHIN the object """
    dist1[dist1 > 0] = 0.5

    dist = dist1+dist2
    attention = math.e**(1-dist/edgeFalloff) + background   # adds background so no loses go to zero
    attention /= numpy.average(attention)
    return numpy.reshape(attention,y_.shape)



""" Applies class weighting by multiplying all pixels of ensheathments by 10
"""
def class_weight(y_, loss, weight=10.0):
     
    weight_mat = np.zeros(np.shape(y_))
    weight_mat[weight_mat == 0] = weight 
    weighted_loss = np.multiply(y_,weight_mat)          # multiply by label weights

    return weighted_loss 


""" Initializes cost function, accuracy, and jaccard index 

Notes on epsilon:
- mostly meant to just prevent weights from reaching zero
- relation is that the weight updates = training rate / epsilon, so that means:
  (a) SMALLER epislon ==> trains faster ==> but weights may become unstable
  (b) thus, some even recommend epislon as big as 1 or 0.1
  (c) For float32 ==> the default epsilon 1e-8 does NOT work, so I changed it to 1e-4, but now training super slow???

- trying now, learning rate == 1e-3, epsilon 1e-4

- and also after, should try, learning rate == 1e-4, epsilon 1e-6

"""
def costOptm(y, y_b, logits, weighted_labels, train_rate=1e-5, epsilon = 1e-8, weight_mat=True, optimizer='adam', multiclass=0):
    # Choose fitness/cost function. Many options:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss
    
    #class_w_loss = class_weight(y_b, loss, class_1=1.0, class_2=10.0);   # PERFORMS CLASS WEIGHTING
    original = loss
    if weight_mat:
        w_reduced = tf.reduce_mean(weighted_labels, axis=-1)
        loss = tf.multiply(loss, w_reduced)
        
    cross_entropy = tf.reduce_mean(loss)         # single loss value
        
    if optimizer == 'adam':
         train_step = tf.train.AdamOptimizer(learning_rate=train_rate, epsilon=epsilon).minimize(cross_entropy)  # train_step uses Adam Optimizer
    elif optimizer == 'SGD':
         train_step = tf.train.GradientDescentOptimizer(learning_rate=train_rate).minimize(cross_entropy)  # train_step uses Gradient Descent Optimizer
 
     
    """ Accuracy """ 
    #correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
    accuracy = []
    
    """ Jaccard  """
    if not multiclass:
         output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
         truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
         intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
         union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
         jaccard = tf.reduce_mean(intersection / union)   
        
    else:
         jaccard_all = []
         shape_y = y.get_shape().as_list()
         for class_idx in range(1, shape_y[-1]):
              background = tf.expand_dims(y[:, :, :, :, 0], axis=-1)
              class_y = tf.expand_dims(y[:, :, :, :, class_idx], axis=-1)
              tmp_output = tf.concat([background, class_y], axis=-1)
              output = tf.cast(tf.argmax(tmp_output,axis=-1), dtype=tf.float32)


              background = tf.expand_dims(y_b[:, :, :, :, 0], axis=-1)
              class_y = tf.expand_dims(y_b[:, :, :, :, class_idx], axis=-1)
              tmp_truth = tf.concat([background, class_y], axis=-1)
              truth = tf.cast(tf.argmax(tmp_truth,axis=-1), dtype=tf.float32)
              
              intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
              union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
              jaccard = tf.reduce_mean(intersection / union)     
              
              jaccard_all.append(jaccard)
              
         jaccard = jaccard_all
        
    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original



""" Added clipping to try to prevent explosive loss """

def costOptm_CLIP(y, y_b, logits, weighted_labels, train_rate=1e-5, epsilon = 1e-8, weight_mat=True):
    # Choose fitness/cost function. Many options:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss
    
    #class_w_loss = class_weight(y_b, loss, class_1=1.0, class_2=10.0);   # PERFORMS CLASS WEIGHTING
    original = loss
    if weight_mat:
        w_reduced = tf.reduce_mean(weighted_labels, axis=-1)
        loss = tf.multiply(loss, w_reduced)
        
    cross_entropy = tf.reduce_mean(loss)         # single loss value
    
    optimizer = tf.train.AdamOptimizer(learning_rate=train_rate, epsilon=epsilon)
    gvs = optimizer.compute_gradients(cross_entropy)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    train_step = optimizer.apply_gradients(capped_gvs)
    
    #train_step = tf.train.AdamOptimizer(learning_rate=train_rate, epsilon=epsilon).minimize(cross_entropy)  # train_step uses Adam Optimizer
 
    """ Accuracy """ 
    #correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
    accuracy = []
    
    """ Jaccard  """
    output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
    truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
    intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
    union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
    jaccard = tf.reduce_mean(intersection / union)   
        
    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original


""" Optimizer with loss scaling to adapt to float 16 """
def costOptm_loss_scaling(y, y_b, logits, weighted_labels, train_rate=1e-5, epsilon = 1e-8, weight_mat=True):
    # Choose fitness/cost function. Many options:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss
    
    #class_w_loss = class_weight(y_b, loss, class_1=1.0, class_2=10.0);   # PERFORMS CLASS WEIGHTING
    original = loss
    if weight_mat:
        w_reduced = tf.reduce_mean(weighted_labels, axis=-1)
        loss = tf.multiply(loss, w_reduced)
        
    cross_entropy = tf.reduce_mean(loss)         # single loss value
    opt = tf.train.AdamOptimizer(learning_rate=train_rate, epsilon=epsilon)


    loss_scale_manager = tf.contrib.mixed_precision.FixedLossScaleManager(5000)
    loss_scale_optimizer = tf.contrib.mixed_precision.LossScaleOptimizer(opt, loss_scale_manager)
     
    train_step = loss_scale_optimizer.minimize(loss)

 
    """ Accuracy """ 
    #correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
    #accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
    accuracy = []
    
    """ Jaccard  """
    output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
    truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
    intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
    union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
    jaccard = tf.reduce_mean(intersection / union)   
        
    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, original



""" Initialized cost function with CLASS WEIGHTING """
def costOptm_CLASSW(y, y_b, logits):
    # Choose fitness/cost function. Many options:
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss

    class_1 = 1.0
    class_2 = 10.0    
    shape = np.concatenate(([1], y_b.get_shape().as_list()[1:3], [1]), axis=0)
    first_c = tf.constant(class_1, shape=shape)
    second_c = tf.constant(class_2, shape=shape)
    weights = tf.concat([first_c, second_c], axis=-1)  
    multiplied = tf.multiply(y_b, weights)
    w_reduced = tf.reduce_mean(multiplied, axis=-1)
 
    weighted_loss = tf.multiply(loss, w_reduced)          # multiply by label weights
        
    cross_entropy = tf.reduce_mean(weighted_loss)         # single loss value
    
    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)  # train_step uses Adam Optimizer
 
    """ Accuracy """ 
    correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
    
    """ Jaccard """
    output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
    truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
    intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
    union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
    jaccard = tf.reduce_mean(intersection / union)   
    
    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, weighted_loss

#
#""" Initializes cost function with no Weight """
#def costOptm_noW(y, y_b, logits):
#    # Choose fitness/cost function. Many options:
#    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss
#    cross_entropy = tf.reduce_mean(loss)         # single loss value
#    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)  # train_step uses Adam Optimizer
# 
#    """ Accuracy 
#    """ 
#    correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
#    
#    """ Jaccard
#    """
#    output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
#    truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
#    intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
#    union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
#    jaccard = tf.reduce_mean(intersection / union)   
#    
#    
#    weighted_loss = 0
#    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, weighted_loss
#
#
#""" Initialized cost function with both spatial AND class weighting """
#def costOptm_BOTH(y, y_b, logits, weighted_labels, weight_mat=True):
#    # Choose fitness/cost function. Many options:
#    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y_b, logits=logits) # calculate loss
#    if weight_mat:
#        w_reduced = tf.reduce_mean(weighted_labels, axis=-1)
#        loss = tf.multiply(loss, w_reduced)    
#    #loss = tf.cast(loss, tf.float64)
#    
#    
#    class_1 = np.float32(1.0)
#    class_2 = np.float32(10.0)   
#    shape = np.concatenate(([1], y_b.get_shape().as_list()[1:3], [1]), axis=0) 
#    first_c = tf.constant(class_1, shape=shape)
#    second_c = tf.constant(class_2, shape=shape)
#    weights = tf.concat([first_c, second_c], axis=-1)  
#    multiplied = tf.multiply(y_b, weights)
#    w_reduced = tf.reduce_mean(multiplied, axis=-1)
# 
#    weighted_loss = tf.multiply(loss, w_reduced)          # multiply by label weights
#        
#    cross_entropy = tf.reduce_mean(weighted_loss)         # single loss value
#    
#    train_step = tf.train.AdamOptimizer(1e-5).minimize(cross_entropy)  # train_step uses Adam Optimizer
# 
#    """ Accuracy 
#    """ 
#    correct_prediction = tf.equal(tf.argmax(y, 3), tf.argmax(y_b, 3))  # accuracy prediction
#    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))    
#    
#    """ Jaccard
#    """
#    output = tf.cast(tf.argmax(y,axis=-1), dtype=tf.float32)
#    truth = tf.cast(tf.argmax(y_b,axis=-1), dtype=tf.float32)
#    intersection = tf.reduce_sum(tf.reduce_sum(tf.multiply(output, truth), axis=-1),axis=-1)
#    union = tf.reduce_sum(tf.reduce_sum(tf.cast(tf.add(output, truth)>= 1, dtype=tf.float32), axis=-1),axis=-1) + 0.0000001
#    jaccard = tf.reduce_mean(intersection / union)   
#        
#    return accuracy, jaccard, train_step, cross_entropy, loss, cross_entropy, weighted_loss


""" Creates network architecture for UNet """
def create_network(x, y_b, training, trainable=True):
    # Building Convolutional layers
    siz_f = 5 # or try 5 x 5
    #training = True

    L1 = tf.layers.conv2d(inputs=x, filters=10, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_D', trainable=trainable)
    L2 = tf.layers.conv2d(inputs=L1, filters=20, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_D', trainable=trainable)
    L3 = tf.layers.conv2d(inputs=L2, filters=30, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_D', trainable=trainable)
    L4 = tf.layers.conv2d(inputs=L3, filters=40, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_D', trainable=trainable)
    L5 = tf.layers.conv2d(inputs=L4, filters=50, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_D', trainable=trainable)

    # up 1
    L6 = tf.layers.conv2d_transpose(inputs=L5, filters=50, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_D', trainable=trainable)
    L6_conv = tf.concat([L6, L4], axis=3)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv2d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_D', trainable=trainable)
    L7_conv = tf.concat([L7, L3], axis=3)

    L8 = tf.layers.conv2d_transpose(inputs=L7_conv, filters=30, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_D', trainable=trainable)
    L8_conv = tf.concat([L8, L2], axis=3)
     
    L9 = tf.layers.conv2d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_D', trainable=trainable)
    L9_conv = tf.concat([L9, L1], axis=3)

    L10 = tf.layers.conv2d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_D', trainable=trainable)
    L10_conv = tf.concat([L10, x], axis=3)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv2d(inputs=L10_conv, filters=2, kernel_size=[siz_f, siz_f], strides=1, padding='same', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D', trainable=trainable)
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed



""" Smaller network architecture """
def create_network_SMALL(x, y_b, training):
    # Building Convolutional layers
    siz_f = 5 # or try 5 x 5
    #training = True

    L1 = tf.layers.conv2d(inputs=x, filters=10, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_D')
    L2 = tf.layers.conv2d(inputs=L1, filters=20, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_D')
    L3 = tf.layers.conv2d(inputs=L2, filters=30, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_D')
#    L4 = tf.layers.conv2d(inputs=L3, filters=40, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_D')
#    L5 = tf.layers.conv2d(inputs=L4, filters=50, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_D')

    # up 1
#    L6 = tf.layers.conv2d_transpose(inputs=L5, filters=50, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_D')
#    L6_conv = tf.concat([L6, L4], axis=3)  # add earlier layers, then convolve together
    
#    L7 = tf.layers.conv2d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_D')
#    L7_conv = tf.concat([L7, L3], axis=3)
    L4 = []; L5 = []; L6 = []; L7 = [];

    L8 = tf.layers.conv2d_transpose(inputs=L3, filters=30, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_D')
    L8_conv = tf.concat([L8, L2], axis=3)
     
    L9 = tf.layers.conv2d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_D')
    L9_conv = tf.concat([L9, L1], axis=3)

    L10 = tf.layers.conv2d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_D')
    L10_conv = tf.concat([L10, x], axis=3)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv2d(inputs=L10_conv, filters=2, kernel_size=[siz_f, siz_f], strides=1, padding='same', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed





""" Smaller network architecture """
def create_network_3D(x, y_b, training):
    # Building Convolutional layers
    siz_f = 5 # or try 5 x 5
    siz_f_z = 2
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=10, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=20, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=30, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L4 = tf.layers.conv3d(inputs=L3, filters=40, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L5 = tf.layers.conv3d(inputs=L4, filters=50, kernel_size=[siz_f, siz_f, siz_f_z], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5, filters=50, kernel_size=[siz_f, siz_f, siz_f_z], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=30, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=4, kernel_size=[siz_f, siz_f, siz_f_z], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed