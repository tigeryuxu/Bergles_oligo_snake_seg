# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 20:05:09 2020

@author: tiger
"""

import torch
from torch import nn
import torch.nn.functional as F


class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size, pad):
        block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                )
        return block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size, pad):
            block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=2, stride=2, output_padding=0)
                    )
            return  block
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size, pad):
            block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv3d(kernel_size=1, in_channels=mid_channel, out_channels=out_channels),
                    #torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                    )
            return  block
    
    def __init__(self, in_channel, out_channel, kernel_size, pad):
        super(UNet, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=10, kernel_size=kernel_size, pad=pad)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(10, 20, kernel_size=kernel_size, pad=pad)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(20, 30, kernel_size=kernel_size, pad=pad)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(30, 40, kernel_size=kernel_size, pad=pad)
        self.conv_maxpool4 = torch.nn.MaxPool3d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=40, out_channels=50, padding=pad),
                            torch.nn.ReLU(inplace=True),
                            #torch.nn.BatchNorm2d(512),
                            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=50, out_channels=50, padding=pad),
                            torch.nn.ReLU(inplace=True),
                            #torch.nn.BatchNorm2d(512),
                            torch.nn.ConvTranspose3d(in_channels=50, out_channels=40, kernel_size=2, stride=2, output_padding=0)
                            )
        # Decode
        self.conv_decode4 = self.expansive_block(80, 40, 30, kernel_size=kernel_size, pad=pad)
        self.conv_decode3 = self.expansive_block(60, 30, 20, kernel_size=kernel_size, pad=pad)   # ***must take into account concatenation!
        self.conv_decode2 = self.expansive_block(40, 20, 10, kernel_size=kernel_size, pad=pad)
        self.final_layer  =     self.final_block(20, 10, out_channel, kernel_size=kernel_size, pad=pad)
        
    def crop_and_concat(self, upsampled, bypass, kernel_size, pad, crop=False):
        if crop:
            #c = (bypass.size()[2] - upsampled.size()[2]) // 2
            #bypass = F.pad(bypass, (-c, -c, -c, -c))
        
            """ Tiger crop """  
            upsampled = F.pad(upsampled, (-pad, -pad, -pad, -pad, -pad, -pad))
          
        return torch.cat((upsampled, bypass), 1)
   
     
    def center_crop(self, layer, target_size, bridge):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[1]) // 2
        diff_x = (layer_width - target_size[2]) // 2
        diff_z = (layer_depth - target_size[0]) // 2
        layer[
            :, :, diff_z: (diff_z + target_size[0]), diff_y : (diff_y + target_size[1]), diff_x : (diff_x + target_size[2])
        ]
        
        return torch.cat([bridge, layer], 1)
    
    def forward(self, x, pad):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)
        
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool4)
        # Decode
        decode_block4 = self.center_crop(bottleneck1, encode_block4.shape[2:], encode_block4)
        cat_layer4 = self.conv_decode4(decode_block4)
        decode_block3 = self.center_crop(cat_layer4, encode_block3.shape[2:], encode_block3)
        cat_layer3 = self.conv_decode3(decode_block3)
        decode_block2 = self.center_crop(cat_layer3, encode_block2.shape[2:], encode_block2)
        cat_layer2 = self.conv_decode2(decode_block2)
        decode_block1 = self.center_crop(cat_layer2, encode_block1.shape[2:], encode_block1)        
        
        final_layer = self.final_layer(decode_block1)
        return  final_layer
   
     
   
     
""" With dilated network """
class UNet_dilated_keep_maxpool(nn.Module):
   
    def contracting_block(self, in_channels, out_channels, kernel_size, pad, dilation):
        block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=pad, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=pad, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                )
        return block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size, pad, dilation):
            block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=pad, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=pad, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=2, stride=2, output_padding=0)
                    )
            return  block
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size, pad, dilation):
            block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=pad, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=pad, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv3d(kernel_size=1, in_channels=mid_channel, out_channels=out_channels),
                    #torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                    )
            return  block
    
    def __init__(self, in_channel, out_channel, kernel_size, pad):
        super(UNet_dilated_keep_maxpool, self).__init__()
        #Encode
        dilation = 2
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=10, kernel_size=kernel_size, pad=pad, dilation=dilation)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(10, 20, kernel_size=kernel_size, pad=pad, dilation=dilation)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(20, 30, kernel_size=kernel_size, pad=pad, dilation=dilation)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(30, 40, kernel_size=kernel_size, pad=pad, dilation=dilation)
        self.conv_maxpool4 = torch.nn.MaxPool3d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=40, out_channels=50, padding=pad, dilation=dilation),
                            torch.nn.ReLU(inplace=True),
                            #torch.nn.BatchNorm2d(512),
                            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=50, out_channels=50, padding=pad, dilation=dilation),
                            torch.nn.ReLU(inplace=True),
                            #torch.nn.BatchNorm2d(512),
                            torch.nn.ConvTranspose3d(in_channels=50, out_channels=40, kernel_size=2, stride=2, output_padding=0)
                            )
        # Decode
        self.conv_decode4 = self.expansive_block(80, 40, 30, kernel_size=kernel_size, pad=pad, dilation=dilation)
        self.conv_decode3 = self.expansive_block(60, 30, 20, kernel_size=kernel_size, pad=pad, dilation=dilation)   # ***must take into account concatenation!
        self.conv_decode2 = self.expansive_block(40, 20, 10, kernel_size=kernel_size, pad=pad, dilation=dilation)
        self.final_layer  =     self.final_block(20, 10, out_channel, kernel_size=kernel_size, pad=pad, dilation=dilation)
        
    def crop_and_concat(self, upsampled, bypass, kernel_size, pad, crop=False):
        if crop:
            #c = (bypass.size()[2] - upsampled.size()[2]) // 2
            #bypass = F.pad(bypass, (-c, -c, -c, -c))
        
            """ Tiger crop """  
            upsampled = F.pad(upsampled, (-pad, -pad, -pad, -pad, -pad, -pad))
          
        return torch.cat((upsampled, bypass), 1)
   
     
    def center_crop(self, layer, target_size, bridge):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[1]) // 2
        diff_x = (layer_width - target_size[2]) // 2
        diff_z = (layer_depth - target_size[0]) // 2
        layer[
            :, :, diff_z: (diff_z + target_size[0]), diff_y : (diff_y + target_size[1]), diff_x : (diff_x + target_size[2])
        ]
        
        return torch.cat([bridge, layer], 1)
    
    def forward(self, x, pad):
        # Encode
        encode_block1 = self.conv_encode1(x)
        encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_pool1)
        encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_pool2)
        encode_pool3 = self.conv_maxpool3(encode_block3)
        
        encode_block4 = self.conv_encode4(encode_pool3)
        encode_pool4 = self.conv_maxpool4(encode_block4)
        
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_pool4)
        # Decode
        decode_block4 = self.center_crop(bottleneck1, encode_block4.shape[2:], encode_block4)
        cat_layer4 = self.conv_decode4(decode_block4)
        decode_block3 = self.center_crop(cat_layer4, encode_block3.shape[2:], encode_block3)
        cat_layer3 = self.conv_decode3(decode_block3)
        decode_block2 = self.center_crop(cat_layer3, encode_block2.shape[2:], encode_block2)
        cat_layer2 = self.conv_decode2(decode_block2)
        decode_block1 = self.center_crop(cat_layer2, encode_block1.shape[2:], encode_block1)        
        
        final_layer = self.final_layer(decode_block1)
        return  final_layer
   
   
   
   
   
""" With dilated network """
class UNet_dilated_NO_maxpool(nn.Module):

    def contracting_block_first(self, in_channels, out_channels, kernel_size, pad, dilation):
        block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=pad, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                )
        return block     
   
    def contracting_block(self, in_channels, out_channels, kernel_size, pad, dilation):
        block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=2, stride=2, dilation=dilation),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                )
        return block
    
    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size, pad):
            block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=2, stride=2, output_padding=0)
                    )
            return  block
    
    def final_block(self, in_channels, mid_channel, out_channels, kernel_size, pad):
            block = torch.nn.Sequential(
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=pad),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(mid_channel),
                    torch.nn.Conv3d(kernel_size=1, in_channels=mid_channel, out_channels=out_channels),
                    torch.nn.ReLU(inplace=True),
                    #torch.nn.BatchNorm2d(out_channels),
                    )
            return  block
    
    def __init__(self, in_channel, out_channel, kernel_size, pad):
        super(UNet_dilated_NO_maxpool, self).__init__()
        #Encode
        self.conv_encode1 = self.contracting_block_first(in_channels=in_channel, out_channels=10, kernel_size=kernel_size, pad=pad, dilation=1)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(10, 20, kernel_size=kernel_size, pad=pad, dilation=2)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(20, 30, kernel_size=kernel_size, pad=pad, dilation=2)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode4 = self.contracting_block(30, 40, kernel_size=kernel_size, pad=pad, dilation=2)
        self.conv_maxpool4 = torch.nn.MaxPool3d(kernel_size=2)
        # Bottleneck
        self.bottleneck = torch.nn.Sequential(
                            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=40, out_channels=50, padding=pad, dilation=2),
                            torch.nn.ReLU(inplace=True),
                            #torch.nn.BatchNorm2d(512),
                            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=50, out_channels=50, padding=pad),
                            torch.nn.ReLU(inplace=True),
                            #torch.nn.BatchNorm2d(512),
                            torch.nn.ConvTranspose3d(in_channels=50, out_channels=40, kernel_size=2, stride=2, output_padding=0)
                            )
        # Decode
        self.conv_decode4 = self.expansive_block(80, 40, 30, kernel_size=kernel_size, pad=pad)
        self.conv_decode3 = self.expansive_block(60, 30, 20, kernel_size=kernel_size, pad=pad)   # ***must take into account concatenation!
        self.conv_decode2 = self.expansive_block(40, 20, 10, kernel_size=kernel_size, pad=pad)
        self.final_layer  =     self.final_block(20, 10, out_channel, kernel_size=kernel_size, pad=pad)
        
    def crop_and_concat(self, upsampled, bypass, kernel_size, pad, crop=False):
        if crop:
            #c = (bypass.size()[2] - upsampled.size()[2]) // 2
            #bypass = F.pad(bypass, (-c, -c, -c, -c))
        
            """ Tiger crop """  
            upsampled = F.pad(upsampled, (-pad, -pad, -pad, -pad, -pad, -pad))
          
        return torch.cat((upsampled, bypass), 1)
   
     
    def center_crop(self, layer, target_size, bridge):
        _, _, layer_depth, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[1]) // 2
        diff_x = (layer_width - target_size[2]) // 2
        diff_z = (layer_depth - target_size[0]) // 2
        layer[
            :, :, diff_z: (diff_z + target_size[0]), diff_y : (diff_y + target_size[1]), diff_x : (diff_x + target_size[2])
        ]
        
        return torch.cat([bridge, layer], 1)
    
    def forward(self, x, pad):
        # Encode
        encode_block1 = self.conv_encode1(x)
        #encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_block1)
        #encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_block2)
        #encode_pool3 = self.conv_maxpool3(encode_block3)
        
        encode_block4 = self.conv_encode4(encode_block3)
        #encode_pool4 = self.conv_maxpool4(encode_block4)
        
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_block4)
        # Decode
        decode_block4 = self.center_crop(bottleneck1, encode_block4.shape[2:], encode_block4)
        cat_layer4 = self.conv_decode4(decode_block4)
        decode_block3 = self.center_crop(cat_layer4, encode_block3.shape[2:], encode_block3)
        cat_layer3 = self.conv_decode3(decode_block3)
        decode_block2 = self.center_crop(cat_layer3, encode_block2.shape[2:], encode_block2)
        cat_layer2 = self.conv_decode2(decode_block2)
        decode_block1 = self.center_crop(cat_layer2, encode_block1.shape[2:], encode_block1)        
        
        final_layer = self.final_layer(decode_block1)
        return  final_layer
   
   
   