#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 22:24:21 2020

@author: user
"""

import os
import sys
#from tqdm import tqdm
#from tensorboardX import SummaryWriter
#import shutil
#import argparse
#import logging
#import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
#import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

#from networks.vnet import VNet
#from dataloaders.livertumor import LiverTumor, RandomCrop, CenterCrop, RandomRotFlip, ToTensor, TwoStreamBatchSampler

from scipy.ndimage import distance_transform_edt as distance


""" Steps alpha after each epoch """
def alpha_step(ce, dc, hd, iter_cur_epoch):
     mean_ce = ce/iter_cur_epoch
     mean_dc = dc/iter_cur_epoch
     mean_combined = (mean_ce + mean_dc)/2
    
     mean_hd = hd/iter_cur_epoch
    
     alpha = mean_hd/(mean_combined)
     
     return alpha

""" computes composite (DICE + CE) + alpha * HD loss """
def compute_HD_loss(output, labels, alpha, tracker, ce, dc, hd, val_bool=0):
    loss_ce = F.cross_entropy(output, labels)
    outputs_soft = F.softmax(output, dim=1)
    loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels == 1)
    # compute distance maps and hd loss
    with torch.no_grad():
        # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
        gt_dtm_npy = compute_dtm(labels.cpu().numpy(), outputs_soft.shape)
        gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
        seg_dtm_npy = compute_dtm(outputs_soft[:, 1, :, :, :].cpu().numpy()>0.5, outputs_soft.shape)
        seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)

    loss_hd = hd_loss(outputs_soft, labels, seg_dtm, gt_dtm)
    
    loss = alpha*(loss_ce+loss_seg_dice) + loss_hd

    
    if not val_bool:   ### append to training trackers if not validation
        tracker.train_ce_pb.append(loss_ce.cpu().data.numpy())
        tracker.train_dc_pb.append(loss_seg_dice.cpu().data.numpy())
        tracker.train_hd_pb.append(loss_hd.cpu().data.numpy())

    else:
        tracker.val_ce_pb.append(loss_ce.cpu().data.numpy())
        tracker.val_dc_pb.append(loss_seg_dice.cpu().data.numpy())
        tracker.val_hd_pb.append(loss_hd.cpu().data.numpy())        

    
    ce += loss_ce.cpu().data.numpy()
    dc += loss_seg_dice.cpu().data.numpy()
    hd += loss_hd.cpu().data.numpy()    
    
    return loss, tracker, ce, dc, hd



def dice_loss(score, target):
    target = target.float()
    smooth = 1e-5
    intersect = torch.sum(score * target)
    y_sum = torch.sum(target * target)
    z_sum = torch.sum(score * score)
    loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
    loss = 1 - loss
    return loss

def compute_dtm01(img_gt, out_shape):
    """
    compute the normalized distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) shape=out_shape
    sdf(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
             0; x out of segmentation
    normalize sdf to [0, 1]
    """

    normalized_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
            # ignore background
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                normalized_dtm[b][c] = posdis/np.max(posdis)

    return normalized_dtm

def compute_dtm(img_gt, out_shape):
    """
    compute the distance transform map of foreground in binary mask
    input: segmentation, shape = (batch_size, x, y, z)
    output: the foreground Distance Map (SDM) 
    dtm(x) = 0; x in segmentation boundary
             inf|x-y|; x in segmentation
    """

    fg_dtm = np.zeros(out_shape)

    for b in range(out_shape[0]): # batch size
        for c in range(1, out_shape[1]):
            posmask = img_gt[b].astype(np.bool)
            if posmask.any():
                posdis = distance(posmask)
                fg_dtm[b][c] = posdis

    return fg_dtm

def hd_loss(seg_soft, gt, seg_dtm, gt_dtm):
    """
    compute huasdorff distance loss for binary segmentation
    input: seg_soft: softmax results,  shape=(b,2,x,y,z)
           gt: ground truth, shape=(b,x,y,z)
           seg_dtm: segmentation distance transform map; shape=(b,2,x,y,z)
           gt_dtm: ground truth distance transform map; shape=(b,2,x,y,z)
    output: boundary_loss; sclar
    """

    delta_s = (seg_soft[:,1,...] - gt.float()) ** 2
    s_dtm = seg_dtm[:,1,...] ** 2
    g_dtm = gt_dtm[:,1,...] ** 2
    dtm = s_dtm + g_dtm
    multipled = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    hd_loss = multipled.mean()

    return hd_loss



""" DEFAULT ALPHA == 1.0 

        alpha -= 0.001
        if alpha <= 0.001:
            alpha = 0.001
            
            
            
        *** in paper, did as ratio???
            "We  choose λ such  that  equal  weights  are  given  to  the HD-based  and  DSC  loss  terms.  
            Specifically,  after  eachtraining epoch, we compute the HD-based and DSC lossterms 
            on the training data and setλ(for the next epoch)as  the  ratio  of  the  mean  of  
            the  HD-based  loss  term  tothe  mean  of  the  DSC  loss  term.""

"""

"""
            loss_ce = F.cross_entropy(outputs, label_batch)
            outputs_soft = F.softmax(outputs, dim=1)
            loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], label_batch == 1)
            # compute distance maps and hd loss
            with torch.no_grad():
                # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
                gt_dtm_npy = compute_dtm(label_batch.cpu().numpy(), outputs_soft.shape)
                gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
                seg_dtm_npy = compute_dtm(outputs_soft[:, 1, :, :, :].cpu().numpy()>0.5, outputs_soft.shape)
                seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)

            loss_hd = hd_loss(outputs_soft, label_batch, seg_dtm, gt_dtm)
            loss = alpha*(loss_ce+loss_seg_dice) + (1 - alpha) * loss_hd

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            alpha -= 0.001
            if alpha <= 0.001:
                alpha = 0.001

"""