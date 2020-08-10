#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 28 14:39:11 2020

@author: user
"""


class train_tracker():
    
    def __init__(self):
        """ Get metrics per batch """
        self.train_loss_per_batch = [] 
        self.train_jacc_per_batch = []
        self.val_loss_per_batch = []; self.val_jacc_per_batch = []
        
        
        self.train_ce_pb = []; self.train_hd_pb = []; self.train_dc_pb = [];
        
        """ Get metrics per epoch"""
        self.train_loss_per_epoch = []; self.train_jacc_per_epoch = []
        self.val_loss_per_eval = []; self.val_jacc_per_eval = []
        self. plot_sens = []; self.plot_sens_val = [];
        self.plot_prec = []; self.plot_prec_val = [];
        self.lr_plot = [];
        self.iterations = 0;


        # plot_sens = check['plot_sens']
        # plot_sens_val = check['plot_sens_val']
        # plot_prec = check['plot_prec']
        # plot_prec_val = check['plot_prec_val']
        
        
        """ Netwrok params """
        # self.kernel_size = kernel_size
        # self.loss_function = loss_function        
        # self.optimizer = optimizer        
        # self.scheduler = scheduler
        
        """ Bools """
        # self.switch_norm = switch_norm
        # self.deep_supervision = deep_supervision

        
        
        
    def load():
        
        return
        
    def save():
         
         return