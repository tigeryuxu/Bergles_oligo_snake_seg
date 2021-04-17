#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 17:03:45 2021

@author: user
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from cIDice_soft_skeleton import soft_skel


class soft_cldice(nn.Module):
    def __init__(self, iter_=3, smooth = 1.):
        #super().__init__()   ### Tiger added this
        super(soft_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth

    def forward(self, y_true, y_pred):
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,0,:,:,:])+self.smooth)/(torch.sum(skel_pred[:,0,:,:,:])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,0,:,:,:])+self.smooth)/(torch.sum(skel_true[:,0,:,:,:])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return cl_dice


def soft_dice(y_true, y_pred):
    """[function to compute dice loss]
    Args:
        y_true ([float32]): [ground truth image]
        y_pred ([float32]): [predicted image]
    Returns:
        [float32]: [loss value]
    """
    y_true = y_true.float()
    smooth = 1e5
    intersection = torch.sum((y_true * y_pred))
    
    y_sum = torch.sum(y_true * y_true)
    z_sum = torch.sum(y_pred * y_pred)
    
    coeff = (2. *  intersection + smooth) /  (z_sum + y_sum + smooth)
    return (1. - coeff)


class soft_dice_cldice(nn.Module):
    def __init__(self, iter_=3, alpha=0.5, smooth = 1.):
        #super().__init__()    ### Tiger added this
        super(soft_dice_cldice, self).__init__()
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha

    def forward(self, y_true, y_pred):
        
        """ First get normal DICE loss """
        dice = soft_dice(y_true, y_pred)
        skel_pred = soft_skel(y_pred, self.iter)
        skel_true = soft_skel(y_true, self.iter)
        
        
        """ Then get skeleton DICE loss """
        tprec = (torch.sum(torch.multiply(skel_pred, y_true)[:,0,:,:,:])+self.smooth)/(torch.sum(skel_pred[:,0,:,:,:])+self.smooth)    
        tsens = (torch.sum(torch.multiply(skel_true, y_pred)[:,0,:,:,:])+self.smooth)/(torch.sum(skel_true[:,0,:,:,:])+self.smooth)    
        cl_dice = 1.- 2.0*(tprec*tsens)/(tprec+tsens)
        return (1.0-self.alpha)*dice+self.alpha*cl_dice