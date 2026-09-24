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
    # builder for visual stream
    def build_visual(self, weights='', backbone='dinov2_vitb14_reg',
                     dinov3_repo='', dinov3_weights='', head_layout='multi'):
        net = VisualNet(
            backbone=backbone,
            dinov3_repo=dinov3_repo,
            dinov3_weights=dinov3_weights,
            pretrained=(len(weights) == 0), head_layout=head_layout)

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights))
        return net

    #builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights='', head_layout='multi'):
        #AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc, head_layout=head_layout)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights))
        return net
