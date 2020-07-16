#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:39:11 2020

@author: user
"""


class train_tracker():
    
    def __init__(kernel_size):
        """ Get metrics per batch """
        self.train_loss_per_batch = [] 
        self.train_jacc_per_batch = []
        self.val_loss_per_batch = []; self.val_jacc_per_batch = []
        """ Get metrics per epoch"""
        self.train_loss_per_epoch = []; self.train_jacc_per_epoch = []
        self.val_loss_per_eval = []; self.val_jacc_per_eval = []
        self. plot_sens = []; self.plot_sens_val = [];
        self.plot_prec = []; self.plot_prec_val = [];
        self.lr_plot = [];
        self.iterations = 0;
        
        
        """ Netwrok params """
        self.kernel_size = kernel_size
        
        
        
    def load():
        
        
        
    def save():
    