#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import random
import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.model == 'audioVisual':
        from data.audioVisual_dataset import AudioVisualDataset
        dataset = AudioVisualDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        
        training = opt.mode == 'train'
        # Match the historical validation policy: shuffled full batches only.
        full_batches = opt.mode in ('train', 'val')
        if training and opt.batchSize < 2:
            raise ValueError('Contrastive training requires batchSize >= 2')
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset, batch_size=opt.batchSize, shuffle=full_batches,
            drop_last=full_batches, num_workers=int(opt.nThreads))
        if full_batches and len(self.dataloader) == 0:
            raise ValueError('Training/validation split is smaller than one full batch')

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

