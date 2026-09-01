#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from . import networks,criterion


class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt
        #initialize model
        self.net_visual, self.net_audio = nets

    def forward(self, input):
        visual_input = input['frame']
        audio_diff = input['audio_diff_spec']
        audio_mix = input['audio_mix_spec']
        audio_gt = audio_diff[:,:,:-1,:].detach()

        input_spectrogram = audio_mix
        heads_proj, visual_feature_org = self.net_visual(visual_input)
        
        # heads_proj is [B, 2, 384, H, W], output from projection layers.
        visual_feature_semantic = heads_proj[:, 0, :, :, :]
        visual_feature_spatial = heads_proj[:, 1, :, :, :]
        
        # Compute l and r from mix and diff
        audio_l = 0.5 * (audio_mix + audio_diff)
        audio_r = 0.5 * (audio_mix - audio_diff)
        audio_spatial = torch.cat((audio_l, audio_r), dim=1)
        
        # Pass the original un-projected 768-ch visual feature map [B, 768, H, W] to AudioNet for U-Net decoding
        mask_prediction, spatial_feature, semantic_audio_feat, spatial_audio_feat = self.net_audio(audio_mix, audio_spatial, visual_feature_org)

        #complex masking to obtain the predicted spectrogram
        spectrogram_diff_real = input_spectrogram[:,0,:-1,:] * mask_prediction[:,0,:,:] - input_spectrogram[:,1,:-1,:] * mask_prediction[:,1,:,:]
        spectrogram_diff_img = input_spectrogram[:,0,:-1,:] * mask_prediction[:,1,:,:] + input_spectrogram[:,1,:-1,:] * mask_prediction[:,0,:,:]
        binaural_spectrogram = torch.cat((spectrogram_diff_real.unsqueeze(1), spectrogram_diff_img.unsqueeze(1)), 1)

        output =  {
            'mask_prediction': mask_prediction, 
            'binaural_spectrogram': binaural_spectrogram, 
            'audio_gt': audio_gt, 
            'spatial_feature': spatial_feature,
            'semantic_audio_feat': semantic_audio_feat,
            'spatial_audio_feat': spatial_audio_feat,
            'semantic_visual_feat': visual_feature_semantic,
            'spatial_visual_feat': visual_feature_spatial
        }
        return output
