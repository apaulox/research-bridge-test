#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from .networks import VisualNet, AudioNet, weights_init

class ModelBuilder():
    # builder for visual stream
    def build_visual(self, weights='', backbone='dinov3_vitb16',
                     dinov3_repo='', dinov3_weights=''):
        net = VisualNet(
            backbone=backbone,
            dinov3_repo=dinov3_repo,
            dinov3_weights=dinov3_weights,
            pretrained=(len(weights) == 0))

        if len(weights) > 0:
            print('Loading weights for visual stream')
            net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        return net

    #builder for audio stream
    def build_audio(self, ngf=64, input_nc=2, output_nc=2, weights=''):
        #AudioNet: 5 layer UNet
        net = AudioNet(ngf, input_nc, output_nc)

        net.apply(weights_init)
        if len(weights) > 0:
            print('Loading weights for audio stream')
            net.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
        return net
