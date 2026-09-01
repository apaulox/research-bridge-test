#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torchvision
from torchvision.models import ResNet18_Weights
from .networks import VisualNet, AudioNet, weights_init

class ModelBuilder():
    def build_visual(self, weights=''):
        pretrained = True
        net = VisualNet()

        if len(weights) > 0:
            print('Loading weights for visual stream')
            state_dict = torch.load(weights)
            # Check if this is an older checkpoint without shared_proj
            has_shared_proj = any('shared_proj' in k for k in state_dict.keys())
            
            if has_shared_proj:
                net.load_state_dict(state_dict)
            else:
                print('Old checkpoint detected (missing shared_proj). Adapting architecture...')
                net.load_state_dict(state_dict, strict=False)
                # Replace the randomly initialized shared_proj with Identity to mimic old behavior
                net.shared_proj = torch.nn.Sequential(torch.nn.Identity())
                
        return net

    #builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights=''):
        #AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net
