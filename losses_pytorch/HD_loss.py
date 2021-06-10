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




import cv2 as cv
import numpy as np

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt
from scipy.ndimage import convolve

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
"""
class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0, **kwargs):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)
        pred = pred.cpu().numpy()
        target = target.cpu().numpy()
    
        pred_dt = torch.from_numpy(self.distance_field(pred)).float()
        target_dt = torch.from_numpy(self.distance_field(target)).float()

        pred_error = (pred - target) ** 2
        distance = pred_dt ** self.alpha + target_dt ** self.alpha

        dt_field = torch.from_numpy(pred_error) * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    pred_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    pred_dt.cpu().numpy()[0, 0],
                    target_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return loss


class HausdorffERLoss(nn.Module):
    """Binary Hausdorff loss based on morphological erosion"""

    def __init__(self, alpha=2.0, erosions=10, **kwargs):
        super(HausdorffERLoss, self).__init__()
        self.alpha = alpha
        self.erosions = erosions
        self.prepare_kernels()

    def prepare_kernels(self):
        cross = np.array([cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))])
        bound = np.array([[[0, 0, 0], [0, 1, 0], [0, 0, 0]]])

        self.kernel2D = cross * 0.2
        self.kernel3D = np.array([bound, cross, bound]) * (1 / 7)

    @torch.no_grad()
    def perform_erosion(
        self, pred: np.ndarray, target: np.ndarray, debug
    ) -> np.ndarray:
        bound = (pred - target) ** 2

        if bound.ndim == 5:
            kernel = self.kernel3D
        elif bound.ndim == 4:
            kernel = self.kernel2D
        else:
            raise ValueError(f"Dimension {bound.ndim} is nor supported.")

        eroted = np.zeros_like(bound)
        erosions = []

        for batch in range(len(bound)):

            # debug
            erosions.append(np.copy(bound[batch][0]))

            for k in range(self.erosions):

                # compute convolution with kernel
                dilation = convolve(bound[batch], kernel, mode="constant", cval=0.0)

                # apply soft thresholding at 0.5 and normalize
                erosion = dilation - 0.5
                erosion[erosion < 0] = 0

                if erosion.ptp() != 0:
                    erosion = (erosion - erosion.min()) / erosion.ptp()

                # save erosion and add to loss
                bound[batch] = erosion
                eroted[batch] += erosion * (k + 1) ** self.alpha

                if debug:
                    erosions.append(np.copy(erosion[0]))

        # image visualization in debug mode
        if debug:
            return eroted, erosions
        else:
            return eroted

    def forward(
        self, pred: torch.Tensor, target: torch.Tensor, debug=False
    ) -> torch.Tensor:
        """
        Uses one binary channel: 1 - fg, 0 - bg
        pred: (b, 1, x, y, z) or (b, 1, x, y)
        target: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert pred.dim() == 4 or pred.dim() == 5, "Only 2D and 3D supported"
        assert (
            pred.dim() == target.dim()
        ), "Prediction and target need to be of same dimension"

        # pred = torch.sigmoid(pred)

        if debug:
            eroted, erosions = self.perform_erosion(
                pred.cpu().numpy(), target.cpu().numpy(), debug
            )
            return eroted.mean(), erosions

        else:
            eroted = torch.from_numpy(
                self.perform_erosion(pred.cpu().numpy(), target.cpu().numpy(), debug)
            ).float()

            loss = eroted.mean()

            return loss





# """ Steps alpha after each epoch """
# def alpha_step(ce, dc, hd, iter_cur_epoch):
     
#      mean_dc = dc/iter_cur_epoch
#      mean_combined = mean_dc
     
#      ### IF WANT TO ADD LOSS_CE
#      mean_ce = ce/iter_cur_epoch
#      mean_combined = (mean_ce + mean_dc)/2
    
#      mean_hd = hd/iter_cur_epoch
    
#      alpha = mean_hd/(mean_combined)
     
#      return alpha

# """ computes composite (DICE + CE) + alpha * HD loss """
# def compute_HD_loss(output, labels, alpha, tracker, ce, dc, hd, val_bool=0, spatial_weight=0, weight_arr=[]):
    
#     """ April 16th 2021 ----- Tiger removed CE loss because probably not helping... """
#     loss_ce = F.cross_entropy(output, labels, reduction='none')
#     #ce = 0
    
#     """ Make spatial weight matrix that is just exponential decay from middle of image """
#     # if spatial_weight:
#     #     weighted = torch.multiply(loss_ce, weight_arr)
#     #     loss_ce = weighted
        
#     loss_ce = torch.mean(loss_ce)
        
    
#     outputs_soft = F.softmax(output, dim=1)
#     loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels == 1, spatial_weight=spatial_weight, weight_arr=weight_arr)
#     # compute distance maps and hd loss
#     # with torch.no_grad():
#     #     # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
#     #     gt_dtm_npy = compute_dtm(labels.cpu().numpy(), outputs_soft.shape)
#     #     gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
#     #     seg_dtm_npy = compute_dtm(outputs_soft[:, 1, :, :, :].cpu().numpy()>0.5, outputs_soft.shape)
#     #     seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)


#     """ debug """
#     #plot_max(inputs[0, 0].cpu().numpy())
#     #plot_max(labels[0].cpu().numpy())


    
    
#     #loss_hd = hd_loss(outputs_argm, labels, seg_dtm, gt_dtm, spatial_weight=spatial_weight, weight_arr=weight_arr)
    
#     """ updated HAUSSDORF LOSS: 
        
#                 can choose either with DT or with ER
#         """
#     outputs_argm = torch.argmax(output, dim=1)
#     hd_loss = HausdorffDTLoss()  
    
#     #hd_loss = HausdorffERLoss()
    
#     loss_hd = hd_loss.forward(pred=outputs_argm.unsqueeze(1), target=labels.unsqueeze(1))
#     loss = alpha*(loss_seg_dice) + loss_hd
    
    
    
#     ### if want to add loss_ce
#     loss = alpha*(loss_ce+loss_seg_dice) + loss_hd

    
#     if not val_bool:   ### append to training trackers if not validation
#         #tracker.train_ce_pb.append(loss_ce.cpu().data.numpy())
#         tracker.train_dc_pb.append(loss_seg_dice.cpu().data.numpy())
#         tracker.train_hd_pb.append(loss_hd.cpu().data.numpy())

#     else:
#         #tracker.val_ce_pb.append(loss_ce.cpu().data.numpy())
#         tracker.val_dc_pb.append(loss_seg_dice.cpu().data.numpy())
#         tracker.val_hd_pb.append(loss_hd.cpu().data.numpy())        

    
#     ce += loss_ce.cpu().data.numpy()
#     dc += loss_seg_dice.cpu().data.numpy()
#     hd += loss_hd.cpu().data.numpy()    
    
#     return loss, tracker, ce, dc, hd



""" Steps alpha after each epoch """
def alpha_step(ce, dc, hd, iter_cur_epoch):
      #mean_ce = ce/iter_cur_epoch
      mean_dc = dc/iter_cur_epoch
      mean_combined = mean_dc
      #mean_combined = (mean_ce + mean_dc)/2
    
      mean_hd = hd/iter_cur_epoch
    
      alpha = mean_hd/(mean_combined)
     
      return alpha

""" computes composite (DICE + CE) + alpha * HD loss """
def compute_HD_loss(output, labels, alpha, tracker, ce, dc, hd, val_bool=0, spatial_weight=0, weight_arr=[]):
    
    """ April 16th 2021 ----- Tiger removed CE loss because probably not helping... """
    #loss_ce = F.cross_entropy(output, labels, reduction='none')
    ce = 0
        
    outputs_soft = F.softmax(output, dim=1)
    loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels == 1, spatial_weight=spatial_weight, weight_arr=weight_arr)

    
    """ updated HAUSSDORF LOSS: 
        
                can choose either with DT or with ER
        """
    outputs_argm = torch.argmax(output, dim=1)
    hd_loss = HausdorffDTLoss()  
    
    #hd_loss = HausdorffERLoss()
    
    loss_hd = hd_loss.forward(pred=outputs_argm.unsqueeze(1), target=labels.unsqueeze(1))
    loss = alpha*(loss_seg_dice) + loss_hd
    
    
    
    #loss = alpha*(loss_ce+loss_seg_dice) + loss_hd

    
    if not val_bool:   ### append to training trackers if not validation
        #tracker.train_ce_pb.append(loss_ce.cpu().data.numpy())
        tracker.train_dc_pb.append(loss_seg_dice.cpu().data.numpy())
        tracker.train_hd_pb.append(loss_hd.cpu().data.numpy())

    else:
        #tracker.val_ce_pb.append(loss_ce.cpu().data.numpy())
        tracker.val_dc_pb.append(loss_seg_dice.cpu().data.numpy())
        tracker.val_hd_pb.append(loss_hd.cpu().data.numpy())        

    
    #ce += loss_ce.cpu().data.numpy()
    dc += loss_seg_dice.cpu().data.numpy()
    hd += loss_hd.cpu().data.numpy()    
    
    return loss, tracker, ce, dc, hd



""" computes composite (DICE + CE) loss """
def compute_HD_loss_OLD(output, labels, alpha, tracker, ce, dc, hd, val_bool=0):
    loss_ce = F.cross_entropy(output, labels)
    
    outputs_soft = F.softmax(output, dim=1)
    loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels == 1)
    #compute distance maps and hd loss
    with torch.no_grad():
        # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
        gt_dtm_npy = compute_dtm(labels.cpu().numpy(), outputs_soft.shape)
        gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
        seg_dtm_npy = compute_dtm(outputs_soft[:, 1, :, :, :].cpu().numpy()>0.5, outputs_soft.shape)
        seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)

    loss_hd = hd_loss(outputs_soft, labels, seg_dtm, gt_dtm)
    
    loss = alpha*(loss_ce+loss_seg_dice) + loss_hd

    
    if not val_bool:   ### append to training trackers if not validation
        #tracker.train_ce_pb.append(loss_ce.cpu().data.numpy())
        tracker.train_dc_pb.append(loss_seg_dice.cpu().data.numpy())
        tracker.train_hd_pb.append(loss_hd.cpu().data.numpy())

    else:
        #tracker.val_ce_pb.append(loss_ce.cpu().data.numpy())
        tracker.val_dc_pb.append(loss_seg_dice.cpu().data.numpy())
        tracker.val_hd_pb.append(loss_hd.cpu().data.numpy())  

    
    #ce += loss_ce.cpu().data.numpy()
    dc += loss_seg_dice.cpu().data.numpy()
    hd += loss_hd.cpu().data.numpy()    
    

    return loss, tracker, ce, dc, hd

""" computes composite (DICE + CE) loss """
def compute_DICE_CE_loss(output, labels, alpha, tracker, ce, dc, hd, val_bool=0):
    loss_ce = F.cross_entropy(output, labels)
    
    outputs_soft = F.softmax(output, dim=1)
    loss_seg_dice = dice_loss(outputs_soft[:, 1, :, :, :], labels == 1)
    # compute distance maps and hd loss
    # with torch.no_grad():
    #     # defalut using compute_dtm; however, compute_dtm01 is also worth to try;
    #     gt_dtm_npy = compute_dtm(labels.cpu().numpy(), outputs_soft.shape)
    #     gt_dtm = torch.from_numpy(gt_dtm_npy).float().cuda(outputs_soft.device.index)
    #     seg_dtm_npy = compute_dtm(outputs_soft[:, 1, :, :, :].cpu().numpy()>0.5, outputs_soft.shape)
    #     seg_dtm = torch.from_numpy(seg_dtm_npy).float().cuda(outputs_soft.device.index)

    # loss_hd = hd_loss(outputs_soft, labels, seg_dtm, gt_dtm)
    
    loss = (loss_ce+loss_seg_dice) 

    
    if not val_bool:   ### append to training trackers if not validation
        tracker.train_ce_pb.append(loss_ce.cpu().data.numpy())
        tracker.train_dc_pb.append(loss_seg_dice.cpu().data.numpy())

    else:
        tracker.val_ce_pb.append(loss_ce.cpu().data.numpy())
        tracker.val_dc_pb.append(loss_seg_dice.cpu().data.numpy())

    
    ce += loss_ce.cpu().data.numpy()
    dc += loss_seg_dice.cpu().data.numpy()

    return loss, tracker, ce, dc


def dice_loss(score, target, spatial_weight=0, weight_arr=[]):
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



def hd_loss(seg_soft, gt, seg_dtm, gt_dtm, spatial_weight=0, weight_arr=[]):
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
    multiplied = torch.einsum('bxyz, bxyz->bxyz', delta_s, dtm)
    
    """ Make spatial weight matrix that is just exponential decay from middle of image """
    if spatial_weight:
        weighted = torch.multiply(multiplied, weight_arr)
        multiplied = weighted
    
    
    hd_loss = multiplied.mean()

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