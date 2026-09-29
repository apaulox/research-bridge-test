#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
import os
import sys

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))
    if(Relu):
        model.append(nn.ReLU())
    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

class VisualNet(nn.Module):
    def __init__(self, backbone='dinov3_vitb16', dinov3_repo='',
                 dinov3_weights='', pretrained=True):
        super(VisualNet, self).__init__()
        self.backbone = backbone
        if backbone == 'dinov2_vitb14_reg':
            self.patch_size = 14
            self.feature_extraction = torch.hub.load(
                'facebookresearch/dinov2', backbone, pretrained=pretrained)
        elif backbone == 'dinov3_vitb16':
            if not dinov3_repo:
                raise ValueError('--dinov3_repo is required for the DINOv3 backbone')
            if pretrained and not dinov3_weights:
                raise ValueError(
                    '--dinov3_weights must point to the official DINOv3 ViT-B/16 '
                    'LVD-1689M .pth checkpoint')
            if pretrained and not (dinov3_weights.startswith('http://') or
                                   dinov3_weights.startswith('https://')):
                dinov3_weights = os.path.abspath(os.path.expanduser(dinov3_weights))
                if not os.path.isfile(dinov3_weights):
                    raise FileNotFoundError(
                        'DINOv3 weights not found: %s' % dinov3_weights)
            dinov3_repo = os.path.abspath(os.path.expanduser(dinov3_repo))
            if dinov3_repo not in sys.path:
                sys.path.insert(0, dinov3_repo)
            from dinov3.hub.backbones import dinov3_vitb16
            self.patch_size = 16
            load_kwargs = {'pretrained': pretrained}
            if pretrained:
                load_kwargs['weights'] = dinov3_weights
            self.feature_extraction = dinov3_vitb16(**load_kwargs)
        else:
            raise ValueError('Unsupported visual backbone: %s' % backbone)
        for param in self.feature_extraction.parameters():
            param.requires_grad = False
            
        # One 384-channel map feeds both reconstruction and contrastive learning.
        self.projection = nn.Sequential(
            nn.Conv2d(768, 384, 1, bias=False), nn.BatchNorm2d(384), nn.ReLU(inplace=True),
            nn.Conv2d(384, 384, 1, bias=False), nn.BatchNorm2d(384))

    def forward(self, x):
        # Preserve the original DenseSSL 16x32 visual interface. DINOv3-B/16
        # produces a 14x28 map for the same 224x448 input, so only the adapter
        # resamples its patch map to retain the decoder's spatial dimensions.
        target_h, target_w = x.shape[-2] // 14, x.shape[-1] // 14
        features = self.feature_extraction.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        B, N, C = patch_tokens.shape
        H, W = x.shape[-2] // self.patch_size, x.shape[-1] // self.patch_size
        if N != H * W or C != 768:
            raise RuntimeError(
                '%s returned patch tokens with shape %s; expected [B, %d, 768]'
                % (self.backbone, tuple(patch_tokens.shape), H * W))
        x = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        if (H, W) != (target_h, target_w):
            x = F.interpolate(
                x, size=(target_h, target_w), mode='bilinear', align_corners=False)
        
        return self.projection(x)

class AudioNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioNet, self).__init__()
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        
        self.visual_proj = nn.Conv2d(4096, 512, kernel_size=1)
        self.audionet_upconvlayer1 = unet_upconv(1024, ngf * 8) # 1024 = 512 (projected visual) + 512 (audio feature)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask
        self.conv1x1 = create_conv(384, 8, 1, 0) #reduce dimension of extracted spatial visual features
        
        self.projection = nn.Sequential(
            nn.Conv2d(ngf * 8, ngf * 8, 1, bias=False), nn.BatchNorm2d(ngf * 8), nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 8, 384, 1, bias=False), nn.BatchNorm2d(384))

    def forward(self, mono, visual_feat):
        # Mono encoding is shared by the decoder and contrastive projection.
        audio_conv1feature = self.audionet_convlayer1(mono)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        # Use the shared 384-channel visual map without contrastive normalization or pooling.
        visual_feat = self.conv1x1(visual_feat) # [B, 8, 16, 32]
        visual_feat = visual_feat.reshape(visual_feat.shape[0], -1, 1, 1) # flatten to [B, 4096, 1, 1]
        # visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) # tile to [B, 4096, H_a, W_a]
        visual_feat = self.visual_proj(visual_feat) # [B, 512, 1, 1]
        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[-2], audio_conv5feature.shape[-1]) # tile to [B, 512, H_a, W_a]
        
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)
        
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1)) * 2 - 1

        return mask_prediction, self.projection(audio_conv5feature)
