# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 23:18:44 2019

@author: tiger
"""

from csbdeep import data
import numpy as np
import matplotlib.pyplot as plt
import datetime

from plot_functions import *
from data_functions import *
from data_functions_3D import *
from natsort import natsort_keygen, ns





def plot_max(im):
     ma = np.amax(im, axis=0)
     plt.figure(); plt.imshow(ma)

#cd ... make sure in correct folder!
#%load_ext tensorboard
#%tensorboard --logdir==training:./my_model_with_highSNR-tensorboard/ --host=127.0.0.1
#%tensorboard --logdir=./my_model_with_highSNR-tensorboard/
#netstat -ano | findstr :6006
#taskkill /PID 15488 /F



# image size == 2048 x 2048

#patch_size = (8, 128, 128)     # ALL MUST BE DIVISIBLE BY 4
#n_patches_per_image = 500    # ideally should be 15,000
##""" Load and generate training data """
##raw_data = data.RawData.from_folder(basepath='/lustre04/scratch/yxu233/Training FULL data/', source_dirs=['High SNR - Train','Medium SNR - Train','Low SNR - Train'], 
##                                    target_dir='High SNR - Truth', axes='CZYX', pattern='*.tif*')
##X, Y, XY_axes = data.create_patches(raw_data, patch_size=patch_size, n_patches_per_image=n_patches_per_image)
##
##
##from csbdeep import io
##from csbdeep.io import save_training_data
##save_training_data('training_data_FULL', X, Y, XY_axes)
#
#
#""" Required to allow correct GPU usage ==> or else crashes """
#import tensorflow as tf
#config = tf.ConfigProto()
#config.gpu_options.allow_growth = True
#tf.keras.backend.set_session(tf.Session(config=config))
#
#
#
## about 8 mins per 5 epochs
## so 1 day (27 hours) ==> 1000 epochs
#
#""" Start training """
#from csbdeep.io import load_training_data
#from csbdeep.models import Config, CARE
#(X,Y), (X_val,Y_val), axes = load_training_data('training_data_FULL.npz', validation_split=0.1)
#config = Config(axes, n_dim=3, train_batch_size=1, train_epochs=1000, train_steps_per_epoch=400,
#                train_learning_rate=0.0004, train_tensorboard=True)
#model = CARE(config, 'my_model_FULL_DATASET')
##model.train(X,Y, validation_data=(X_val,Y_val), callbacks=tensorboard_callback)
#model.train(X,Y, validation_data=(X_val,Y_val))
#
#model.export_TF()



patch_size = (16, 64, 64)     # ALL MUST BE DIVISIBLE BY 4
#n_patches_per_image = 15000    # first time this worked on 20x airy, did with 15,000 on single image!!!

n_patches_per_image = 2500    # try with 63x images



#raw_data = data.RawData.from_folder(basepath='/lustre04/scratch/yxu233/Training data 63x - Huganir - shortened/', source_dirs=['Train-medium', 'Train-bad'], target_dir='Truth', axes='CZYX', pattern='*.tif*')


raw_data = data.RawData.from_folder(basepath='/scratch/yxu233/Training_data_LIVE_63x_GOOD_HUGANIR/', source_dirs=['Train-medium_to_downsample', 'Train-bad_to_downsample'], target_dir='Truth', axes='CZYX', pattern='*.tif*')


X, Y, XY_axes = data.create_patches(raw_data, patch_size=patch_size, n_patches_per_image=n_patches_per_image,
                                    shuffle=True)
                                        # CAN TURN BACK ON PATCH_FILTER ==> to reduce background patches


from csbdeep import io
from csbdeep.io import save_training_data
#save_training_data('/lustre04/scratch/yxu233/Training data - Huganir/training_data_HUGANIR_LESS', X, Y, XY_axes)
#save_training_data('/lustre04/scratch/yxu233/Training_data_LIVE_63x_GOOD_HUGANIR/training_data_LIVE_63x_GOOD_HUGANIR.npz', X, Y, XY_axes)

save_training_data('/scratch/yxu233/Training_data_LIVE_63x_GOOD_HUGANIR/training_data_LIVE_63x_GOOD_HUGANIR_downsampled.npz', X, Y, XY_axes)


""" Required to allow correct GPU usage ==> or else crashes """
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


""" Start training """
from csbdeep.io import load_training_data
from csbdeep.models import Config, CARE
#(X,Y), (X_val,Y_val), axes = load_training_data('/lustre04/scratch/yxu233/Training data - Huganir/training_data_HUGANIR_LESS.npz', validation_split=0.1)
(X,Y), (X_val,Y_val), axes = load_training_data('/scratch/yxu233/Training_data_LIVE_63x_GOOD_HUGANIR/training_data_LIVE_63x_GOOD_HUGANIR_downsampled.npz', validation_split=0.1)

""" ORIGINAL TRAINING CONFIG """
#config = Config(axes, n_dim=3, train_batch_size=16, train_epochs=10000, train_steps_per_epoch=400,
#                train_learning_rate=0.0004, train_tensorboard=True,  unet_kern_size=5, unet_n_depth=2)

""" Fooling around with optimization """
config = Config(axes, n_dim=3, train_batch_size=8, train_epochs=10000, train_steps_per_epoch=200,
                 train_learning_rate=0.0004, train_tensorboard=True, unet_kern_size=5, unet_n_depth=2)
                
#model = CARE(config, 'my_model_HUGANIR_LESS')
model = CARE(config, 'my_model_HUGANIR_63x_LIVE_GOOD_REGISTERED_downsampled')
#model.train(X,Y, validation_data=(X_val,Y_val), callbacks=tensorboard_callback)
model.train(X,Y, validation_data=(X_val,Y_val))

model.export_TF()

