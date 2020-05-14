# -*- coding: utf-8 -*-
"""
Created on Sat Jun 15 14:23:25 2019

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



""" Create small network for PERCEPTUAL loss training """
def create_network_3D_PERCEPTUAL_LOSS_256(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=64, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=128, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=256, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    #L4 = tf.layers.conv3d(inputs=L3, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    #L5 = tf.layers.conv3d(inputs=L4, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    #L6 = tf.layers.conv3d_transpose(inputs=L5, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    #L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    #L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    #L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L3, filters=256, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=128, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=64, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    y = L11
    
    return y, y_b, L1, L2, L3, L8, L9,L9_conv, L10, L11, logits, softMaxed



""" Create small network for PERCEPTUAL loss training """
def create_network_3D_PERCEPTUAL_LOSS_10(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    #L4 = tf.layers.conv3d(inputs=L3, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    #L5 = tf.layers.conv3d(inputs=L4, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    #L6 = tf.layers.conv3d_transpose(inputs=L5, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    #L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    #L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    #L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L3, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    y = L11
    
    return y, y_b, L1, L2, L3, L8, L9,L9_conv, L10, L11, logits, softMaxed






""" Create small network for PERCEPTUAL loss training """
def create_network_3D_PERCEPTUAL_LOSS_128(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=32, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=64, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=128, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    #L4 = tf.layers.conv3d(inputs=L3, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    #L5 = tf.layers.conv3d(inputs=L4, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    #L6 = tf.layers.conv3d_transpose(inputs=L5, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    #L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    #L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    #L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L3, filters=128, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=64, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=32, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    y = L11
    
    return y, y_b, L1, L2, L3, L8, L9,L9_conv, L10, L11, logits, softMaxed






""" Create small network for PERCEPTUAL loss training """
def create_network_3D_PERCEPTUAL_LOSS_DEEP(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=32, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=64, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=128, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 1, 1], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L4 = tf.layers.conv3d(inputs=L3, filters=512, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L5 = tf.layers.conv3d(inputs=L4, filters=1024, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5, filters=1024, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=512, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=128, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 1, 1], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=64, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=32, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    y = L11
    
    return y, y_b, L1, L2, L3, L8, L9,L9_conv, L10, L11, logits, softMaxed





""" Smaller network architecture """
def create_network_3D(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[3, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L4 = tf.layers.conv3d(inputs=L3, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L5 = tf.layers.conv3d(inputs=L4, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[3, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed



""" Smaller network architecture """
def create_network_3D_for_oligo_seg(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[3, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L4 = tf.layers.conv3d(inputs=L3, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L5 = tf.layers.conv3d(inputs=L4, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[3, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed



""" Smaller network architecture """
def create_network_3D_smaller_NODROP(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True
    #x = tf.layers.dropout(inputs=x, rate=0.5, training=training)
    
    L1 = tf.layers.conv3d(inputs=x, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    #L4 = tf.layers.conv3d(inputs=L3, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    #L5 = tf.layers.conv3d(inputs=L4, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    #L6 = tf.layers.conv3d_transpose(inputs=L5, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    #L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    #L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    #L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L3, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    L4 = []; L5 = []; L6 = []; L7 = [];
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9, L9_conv, L10, L11, logits, softMaxed



""" Smaller network architecture """
def create_network_3D_DROPOUT(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True
    x = tf.layers.dropout(inputs=x, rate=0.8, training=training)

    L1 = tf.layers.conv3d(inputs=x, filters=5, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L4 = tf.layers.conv3d(inputs=L3, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L5 = tf.layers.conv3d(inputs=L4, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=5, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed


""" Smaller network architecture 

default data_format: channels_last (default) corresponds to inputs with shape (batch, depth, height, width, channels)

"""
def create_network_3D_smaller(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[10, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[5, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=30, kernel_size=[2, siz_f, siz_f], strides=[1, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    #L4 = tf.layers.conv3d(inputs=L3, filters=40, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    #L5 = tf.layers.conv3d(inputs=L4, filters=50, kernel_size=[siz_f, siz_f, siz_f_z], strides=[1, 2, 2], padding='same', 
    #                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    #L6 = tf.layers.conv3d_transpose(inputs=L5, filters=50, kernel_size=[siz_f, siz_f, siz_f_z], strides=[1, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    #L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    #L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f, siz_f, siz_f_z], strides=[2, 2, 2], padding='same', 
    #                                activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    #L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L3, filters=30, kernel_size=[2, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[5, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[10, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3,  L8, L9,L9_conv, L10, L11, logits, softMaxed





""" Generates hybrid 2D/3D layers by copying middle slice weights from 2D to 3D conv layer """
def generate_hybrid_layer(sess, inputs, filters, kernel_size, strides, padding, activation, kernel_initializer, name, deconvolve = 0):

    siz_z = kernel_size[0]

    if not deconvolve:
        temp_layer = tf.layers.conv3d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      activation=activation, kernel_initializer=kernel_initializer, name=name + '_new_3D_tmp')
    else:
        temp_layer = tf.layers.conv3d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      activation=activation, kernel_initializer=kernel_initializer, name=name + '_new_3D_tmp')
    w1 = tf.get_default_graph().get_tensor_by_name(name + "/kernel:0")
    w2 = tf.get_default_graph().get_tensor_by_name(name + "_new_3D_tmp/kernel:0")

    tf.global_variables_initializer().run(); tf.local_variables_initializer().run()   # HAVE TO HAVE THESE in order to initialize the newly created layers
    
    w1_r = sess.run(w1);  w2_r = sess.run(w2)
    print("w1_r and w2_r should be different: %.5f" %(np.sum(w1_r - w2_r))) # checks that w1_r and w2_r are different
    
    middle_slice = math.ceil(siz_z / 2);  w2_r[middle_slice, :, :, :, :] = w1_r   # Transfers weights to middle slice
    print("w2_r and w1_r should now be same: %.5f" %(np.sum(w2_r[middle_slice, :, :, :, :] - w1_r))) # checks that w1_r and w2_r are different
    
    new_w2 = tf.constant_initializer(w2_r)    # converts weight matrix into a NEW useable kernel so that can initialize the next step!!!
    if not deconvolve:
        hybrid_layer = tf.layers.conv3d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      activation=activation, kernel_initializer=new_w2, name=name + '_new_3D')
    else:
        hybrid_layer = tf.layers.conv3d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      activation=activation, kernel_initializer=new_w2, name=name + '_new_3D')
        
    
    """ Debug/check that weights are now correctly set"""
    w3 = tf.get_default_graph().get_tensor_by_name(name + "_new_3D/kernel:0")
    
    tf.global_variables_initializer().run(); tf.local_variables_initializer().run()   # HAVE TO HAVE THESE in order to initialize the newly created layers
    
    # Check to see if the newly created layer has the correct new spliced weights
    w3_r = sess.run(w3)
    print("w2_r and w1_r should now be same: %.5f" %(np.sum(w2_r[middle_slice, :, :, :, :] - w1_r))) # checks that w1_r and w2_r are different
    print("w3_r and w2_r should now be same: %.5f" %(np.sum(w3_r  - w2_r))) # checks that w1_r and w2_r are different  
    
    return hybrid_layer


""" Smaller network architecture """
def create_network_3D_COMPLEX(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L4 = tf.layers.conv3d(inputs=L3, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L5 = tf.layers.conv3d(inputs=L4, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=40, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=30, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=20, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed




""" Smaller network architecture """
def create_network_3D_COMPLEX_REAL(x, y_b, kernel_size, training, num_classes):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    #training = True

    L1 = tf.layers.conv3d(inputs=x, filters=25, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=100, kernel_size=[siz_f_z, siz_f, siz_f], strides=[3, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L4 = tf.layers.conv3d(inputs=L3, filters=150, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L5 = tf.layers.conv3d(inputs=L4, filters=200, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5, filters=200, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=150, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=100, kernel_size=[siz_f_z, siz_f, siz_f], strides=[3, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=25, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed





""" Smaller network architecture """
def create_network_3D_COMPLEX_REAL_pool_first(x, y_b, kernel_size, training, num_classes, dropout=None):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    
    if dropout:
         x = tf.layers.dropout(inputs=x, rate=0.8, training=training)


    L1 = tf.layers.conv3d(inputs=x, filters=25, kernel_size=[siz_f_z, siz_f, siz_f], strides=[3, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=100, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L4 = tf.layers.conv3d(inputs=L3, filters=150, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L5 = tf.layers.conv3d(inputs=L4, filters=200, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5, filters=200, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=150, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=100, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=25, kernel_size=[siz_f_z, siz_f, siz_f], strides=[3, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed





""" Smaller network architecture """
def create_network_3D_COMPLEX_REAL_NO_POOL_Z(x, y_b, kernel_size, training, num_classes, dropout=None):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    
    if dropout:
         x = tf.layers.dropout(inputs=x, rate=0.7, training=training)


    L1 = tf.layers.conv3d(inputs=x, filters=25, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L2 = tf.layers.conv3d(inputs=L1, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L3 = tf.layers.conv3d(inputs=L2, filters=100, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L4 = tf.layers.conv3d(inputs=L3, filters=150, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L5 = tf.layers.conv3d(inputs=L4, filters=200, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')

    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5, filters=200, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=150, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=100, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=25, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed
    
    
    
    
    
""" MAX POOLING included """
def create_network_3D_COMPLEX_MAXPOOL(x, y_b, kernel_size, training, num_classes, dropout=None):
    # Building Convolutional layers
    siz_f = kernel_size[1]
    siz_f = kernel_size[2]
    siz_f_z = kernel_size[0]
    
    if dropout:
         x = tf.layers.dropout(inputs=x, rate=0.7, training=training)


    L1 = tf.layers.conv3d(inputs=x, filters=25, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_3D')
    L1_max = tf.nn.max_pool3d(L1, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', data_format=None, name='MaxPool_1')
    
    L2 = tf.layers.conv3d(inputs=L1_max, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_3D')
    L2_max = tf.nn.max_pool3d(L2, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', data_format=None, name='MaxPool_2')
   
    L3 = tf.layers.conv3d(inputs=L2_max, filters=100, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same',
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_3D')
    L3_max = tf.nn.max_pool3d(L3, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', data_format=None, name='MaxPool_3')

    L4 = tf.layers.conv3d(inputs=L3_max, filters=150, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_3D')
    L4_max = tf.nn.max_pool3d(L4, ksize=[1,2,2,2,1], strides=[1,2,2,2,1], padding='SAME', data_format=None, name='MaxPool_4')

    L5 = tf.layers.conv3d(inputs=L4_max, filters=200, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_3D')
    L5_max = tf.nn.max_pool3d(L5, ksize=[1,1,2,2,1], strides=[1,1,2,2,1], padding='SAME', data_format=None, name='MaxPool_4')


    # up 1
    L6 = tf.layers.conv3d_transpose(inputs=L5_max, filters=200, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv1_3D')
    L6_conv = tf.concat([L6, L4_max], axis=-1)  # add earlier layers, then convolve together
    
    L7 = tf.layers.conv3d_transpose(inputs=L6_conv, filters=150, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv2_3D')
    L7_conv = tf.concat([L7, L3_max], axis=-1)

    L8 = tf.layers.conv3d_transpose(inputs=L7_conv, filters=100, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv3_3D')
    L8_conv = tf.concat([L8, L2_max], axis=-1)
     
    L9 = tf.layers.conv3d_transpose(inputs=L8_conv, filters=50, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv4_3D')
    L9_conv = tf.concat([L9, L1_max], axis=-1)

    L10 = tf.layers.conv3d_transpose(inputs=L9_conv, filters=25, kernel_size=[siz_f_z, siz_f, siz_f], strides=[2, 2, 2], padding='same', 
                                     activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_3D')
    L10_conv = tf.concat([L10, x], axis=-1)
          
    # 1 x 1 convolution (NO upsampling) 
    L11 = tf.layers.conv3d(inputs=L10_conv, filters=num_classes, kernel_size=[siz_f_z, siz_f, siz_f], strides=[1, 1, 1], padding='same', activation=None, 
                          kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
   
    softMaxed = tf.nn.softmax(L11, name='Softmaxed')   # for the output, but NOT the logits

    # Set outputs 
    y = softMaxed
    logits = L11
    
    return y, y_b, L1, L2, L3, L4, L5, L6, L7, L8, L9,L9_conv, L10, L11, logits, softMaxed
    
    
        
""" Copies weights of 2D pretrained network to new 2D layer """
def generate_pretrained_2D_layer(sess, inputs, filters, kernel_size, strides, padding, activation, kernel_initializer, name, deconvolve = 0):
    
    siz_f = kernel_size[-1]
    
    if not deconvolve:
        temp_layer = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      activation=activation, kernel_initializer=kernel_initializer, name=name + '_new_3D_tmp')
    else:
        temp_layer = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      activation=activation, kernel_initializer=kernel_initializer, name=name + '_new_3D_tmp')
    w1 = tf.get_default_graph().get_tensor_by_name(name + "/kernel:0")
    w2 = tf.get_default_graph().get_tensor_by_name(name + "_new_3D_tmp/kernel:0")

    tf.global_variables_initializer().run(); tf.local_variables_initializer().run()   # HAVE TO HAVE THESE in order to initialize the newly created layers
    
    w1_r = sess.run(w1);  w2_r = sess.run(w2)
    print("w1_r and w2_r should be different: %.5f" %(np.sum(w1_r - w2_r))) # checks that w1_r and w2_r are different
    
    middle_slice = math.ceil(siz_f / 2);  w2_r[:, :, :, :] = w1_r   # Transfers weights to middle slice
    print("w2_r and w1_r should now be same: %.5f" %(np.sum(w2_r[ :, :, :, :] - w1_r))) # checks that w1_r and w2_r are different
    
    new_w2 = tf.constant_initializer(w2_r)    # converts weight matrix into a NEW useable kernel so that can initialize the next step!!!
    if not deconvolve:
        pretrained = tf.layers.conv2d(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      activation=activation, kernel_initializer=new_w2, name=name + '_new_3D')
    else:
        pretrained = tf.layers.conv2d_transpose(inputs=inputs, filters=filters, kernel_size=kernel_size, strides=strides, padding=padding,
                                      activation=activation, kernel_initializer=new_w2, name=name + '_new_3D')
        
    
    """ Debug/check that weights are now correctly set"""
    w3 = tf.get_default_graph().get_tensor_by_name(name + "_new_3D/kernel:0")
    
    tf.global_variables_initializer().run(); tf.local_variables_initializer().run()   # HAVE TO HAVE THESE in order to initialize the newly created layers
    
    # Check to see if the newly created layer has the correct new spliced weights
    w3_r = sess.run(w3)
    print("w2_r and w1_r should now be same: %.5f" %(np.sum(w2_r[:, :, :, :] - w1_r))) # checks that w1_r and w2_r are different
    print("w3_r and w2_r should now be same: %.5f" %(np.sum(w3_r  - w2_r))) # checks that w1_r and w2_r are different  
    
    return pretrained  
    


""" Set up new neural network for training     
        ***(A) For down-sampling branch ==> must change weight-initialization to be from the original 2D conv
                    ==> keep going until size == 1 for depth ==> then delete that dimension???
           (B) For the up-sampling branch ==> only concatenate the middle slice of the down-sampling 3D array
           
           (C) the last 1 convolutional layers are also turned into 3D convolutions
           
           (D) classifier convolutional layer at end is a NEWLY DEFINED LAYER (not using previous weights)
           
           default data_format: channels_last (default) corresponds to inputs with shape (batch, depth, height, width, channels)

"""
def create_network_hybrid(sess, x_3D, y_3D_, kernel_size, depth, num_truth_class):

    siz_x = kernel_size[1]
    siz_y = kernel_size[2]
    siz_z = kernel_size[0]
    
    
    y_b_3D = y_3D_
    siz_f = 5 # or try 5 x 5
    NEW_LAYER_1_3D = generate_hybrid_layer(sess, inputs=x_3D, filters=10, kernel_size=[siz_z, siz_x, siz_y], strides=[5, 2, 2], padding='same', 
                                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_D')
    NEW_LAYER_2_3D = generate_hybrid_layer(sess, inputs=NEW_LAYER_1_3D, filters=20, kernel_size=[siz_z, siz_x, siz_y], strides=[5, 2, 2], padding='same', 
                                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_D')
    NEW_LAYER_3_3D = generate_hybrid_layer(sess, inputs=NEW_LAYER_2_3D, filters=30, kernel_size=[siz_z, siz_x, siz_y], strides=[5, 2, 2], padding='same', 
                                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_D')
    
    layer_3_2D_squeezed = tf.squeeze(NEW_LAYER_3_3D, axis=[1])   # SQUEEZES to remove the depth dimension which has a value of 1
    
    #L1 = tf.layers.conv2d(inputs=x, filters=10, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='new_Conv1_D')
    #L2 = tf.layers.conv2d(inputs=L1, filters=20, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='new_Conv2_D')
    #L3 = tf.layers.conv2d(inputs=L2, filters=30, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='new_Conv3_D')
    L4_new = generate_pretrained_2D_layer(sess, inputs=layer_3_2D_squeezed, filters=40, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
                              activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_D')
    L5_new = generate_pretrained_2D_layer(sess, inputs=L4_new, filters=50, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
                              activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_D')
    
    # up 1
    L6_new = generate_pretrained_2D_layer(sess, inputs=L5_new, filters=50, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
                                        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                        name='DeConv1_D', deconvolve = 1)
    L6_conv_new = tf.concat([L6_new, L4_new], axis=3)  # add earlier layers, then convolve together
     
    L7_new = generate_pretrained_2D_layer(sess, inputs=L6_conv_new, filters=40, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
                                        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                        name='DeConv2_D', deconvolve = 1)
    L7_conv_new = tf.concat([L7_new, layer_3_2D_squeezed], axis=3)
    
    L8_new = generate_pretrained_2D_layer(sess, inputs=L7_conv_new, filters=30, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
                                        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                        name='DeConv3_D', deconvolve = 1)
    L8_conv_new = tf.concat([L8_new, NEW_LAYER_2_3D[:, math.ceil(int(NEW_LAYER_2_3D.shape[1]) / 2), :, :, :]], axis=3)
     
    L9_new = generate_pretrained_2D_layer(sess, inputs=L8_conv_new, filters=20, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
                                        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                        name='DeConv4_D', deconvolve = 1)
    
    
    L9_conv_new = tf.concat([L9_new, NEW_LAYER_1_3D[:, math.ceil(int(NEW_LAYER_1_3D.shape[1]) / 2), :, :, :]], axis=3)
    
    #L10 = tf.layers.conv2d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='new_DeConv5_D')
    
    """ newly added expand_dims and turn last 2 layers also into hybrid 3D layers"""
    L9_expanded = tf.expand_dims((L9_conv_new), 1)
    L10_new_3D = generate_hybrid_layer(sess, inputs=L9_expanded, filters=10, kernel_size=[siz_z, siz_x, siz_y], strides=[depth, 2, 2], padding='same', 
                                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_D', 
                                      deconvolve = 1)
    L10_conv_new_3D = tf.concat([L10_new_3D, x_3D], axis=-1)
    
    # 1 x 1 convolution (NO upsampling) 
    #L11 = tf.layers.conv2d(inputs=L10_conv, filters=2, kernel_size=[siz_f, siz_f], strides=1, padding='same', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='new_Conv1x1_D')
    #L11_new_3D = generate_hybrid_layer(inputs=L10_conv_new_3D, filters=num_truth_class, kernel_size=[siz_f, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
    #                                  activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
    L11_new_3D = tf.layers.conv3d(inputs=L10_conv_new_3D, filters=num_truth_class, kernel_size=[siz_x, siz_y, siz_z], strides=[1, 1, 1], padding='same',
                                  activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='new_Conv1x1_D')
    
    
    
    softMaxed_new = tf.nn.softmax(L11_new_3D, name='new_Softmaxed')   # for the output, but NOT the logits
    
    # Set outputs 
    y_3D = softMaxed_new
    logits_3D = L11_new_3D
    
    return y_3D, y_b_3D, NEW_LAYER_1_3D, NEW_LAYER_2_3D, NEW_LAYER_3_3D, L4_new, L5_new, L6_new, L7_new, L8_new, L9_new, L9_conv_new, L10_new_3D, L11_new_3D, logits_3D, softMaxed_new




""" Set up new neural network for training     
        ***(A) For down-sampling branch ==> must change weight-initialization to be from the original 2D conv
                    ==> keep going until size == 1 for depth ==> then delete that dimension???
           (B) For the up-sampling branch ==> only concatenate the middle slice of the down-sampling 3D array
           
           (C) the last 1 convolutional layers are also turned into 3D convolutions
           
           (D) classifier convolutional layer at end is a NEWLY DEFINED LAYER (not using previous weights)
           
           
           default data_format: channels_last (default) corresponds to inputs with shape (batch, depth, height, width, channels)

           
"""
def create_network_hybrid_small(sess, x_3D, y_3D_, kernel_size, depth, num_truth_class):

    siz_x = kernel_size[1]
    siz_y = kernel_size[2]
    siz_z = kernel_size[0]
    
    
    y_b_3D = y_3D_
    siz_f = 5 # or try 5 x 5
    NEW_LAYER_1_3D = generate_hybrid_layer(sess, inputs=x_3D, filters=10, kernel_size=[siz_z, siz_x, siz_y], strides=[5, 2, 2], padding='same', 
                                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1_D')
    NEW_LAYER_2_3D = generate_hybrid_layer(sess, inputs=NEW_LAYER_1_3D, filters=20, kernel_size=[siz_z, siz_x, siz_y], strides=[5, 2, 2], padding='same', 
                                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv2_D')
    NEW_LAYER_3_3D = generate_hybrid_layer(sess, inputs=NEW_LAYER_2_3D, filters=30, kernel_size=[siz_z, siz_x, siz_y], strides=[5, 2, 2], padding='same', 
                                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv3_D')
    
    layer_3_2D_squeezed = tf.squeeze(NEW_LAYER_3_3D, axis=[1])   # SQUEEZES to remove the depth dimension which has a value of 1
    

    #L4_new = generate_pretrained_2D_layer(sess, inputs=layer_3_2D_squeezed, filters=40, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
    #                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv4_D')
    #L5_new = generate_pretrained_2D_layer(sess, inputs=L4_new, filters=50, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
    #                          activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv5_D')
    
    # up 1
    #L6_new = generate_pretrained_2D_layer(sess, inputs=L5_new, filters=50, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
    #                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
    #                                    name='DeConv1_D', deconvolve = 1)
    #L6_conv_new = tf.concat([L6_new, L4_new], axis=3)  # add earlier layers, then convolve together
     
    #L7_new = generate_pretrained_2D_layer(sess, inputs=L6_conv_new, filters=40, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
    #                                    activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
    #                                    name='DeConv2_D', deconvolve = 1)
    #L7_conv_new = tf.concat([L7_new, layer_3_2D_squeezed], axis=3)
    
#    size = layer_3_2D_squeezed.shape
#    z_size = size[-1]
#    x_size = size[1]
#    y_size = size[2]
#    temp_array = tf.constant(0, shape=[None, x_size, y_size, z_size])
#    
    zeros_30 = tf.fill(tf.shape(layer_3_2D_squeezed), 0.0)
    zeros_10 = tf.fill(tf.shape(layer_3_2D_squeezed), 0.0)
    zeros_40 = tf.concat([zeros_30, zeros_10[:, :, :, 0:10]], axis=-1)
    
    L3_new_concat = tf.concat([layer_3_2D_squeezed, zeros_40],  axis=3)
    
    
    
    L8_new = generate_pretrained_2D_layer(sess, inputs=L3_new_concat, filters=30, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
                                        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                        name='DeConv3_D', deconvolve = 1)
    L8_conv_new = tf.concat([L8_new, NEW_LAYER_2_3D[:, math.ceil(int(NEW_LAYER_2_3D.shape[1]) / 2), :, :, :]], axis=3)
     
    L9_new = generate_pretrained_2D_layer(sess, inputs=L8_conv_new, filters=20, kernel_size=[siz_x, siz_y], strides=2, padding='same', 
                                        activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), 
                                        name='DeConv4_D', deconvolve = 1)
    
    
    L9_conv_new = tf.concat([L9_new, NEW_LAYER_1_3D[:, math.ceil(int(NEW_LAYER_1_3D.shape[1]) / 2), :, :, :]], axis=3)
    
    #L10 = tf.layers.conv2d_transpose(inputs=L9_conv, filters=10, kernel_size=[siz_f, siz_f], strides=2, padding='same', activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='new_DeConv5_D')
    
    """ newly added expand_dims and turn last 2 layers also into hybrid 3D layers"""
    L9_expanded = tf.expand_dims((L9_conv_new), 1)
    L10_new_3D = generate_hybrid_layer(sess, inputs=L9_expanded, filters=10, kernel_size=[siz_z, siz_x, siz_y], strides=[depth, 2, 2], padding='same', 
                                      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='DeConv5_D', 
                                      deconvolve = 1)
    L10_conv_new_3D = tf.concat([L10_new_3D, x_3D], axis=-1)
    
    # 1 x 1 convolution (NO upsampling) 
    #L11 = tf.layers.conv2d(inputs=L10_conv, filters=2, kernel_size=[siz_f, siz_f], strides=1, padding='same', activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='new_Conv1x1_D')
    #L11_new_3D = generate_hybrid_layer(inputs=L10_conv_new_3D, filters=num_truth_class, kernel_size=[siz_f, siz_f, siz_f], strides=[1, 1, 1], padding='same', 
    #                                  activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='Conv1x1_D')
    L11_new_3D = tf.layers.conv3d(inputs=L10_conv_new_3D, filters=num_truth_class, kernel_size=[siz_x, siz_y, siz_z], strides=[1, 1, 1], padding='same',
                                  activation=None, kernel_initializer=tf.contrib.layers.xavier_initializer(), name='new_Conv1x1_D')
    
    
    
    softMaxed_new = tf.nn.softmax(L11_new_3D, name='new_Softmaxed')   # for the output, but NOT the logits
    
    # Set outputs 
    y_3D = softMaxed_new
    logits_3D = L11_new_3D
    
    return y_3D, y_b_3D, NEW_LAYER_1_3D, NEW_LAYER_2_3D, NEW_LAYER_3_3D, L8_new, L9_new, L9_conv_new, L10_new_3D, L11_new_3D, logits_3D, softMaxed_new