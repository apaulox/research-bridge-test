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

class CategoryBatchSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, batch_size, s_samples):
        self.dataset = dataset
        self.batch_size = batch_size
        self.s = s_samples
        
        # Extract clip ids from the dataset (dataset.audios only contains clips for the current split mode)
        self.clip_ids = [os.path.basename(path).strip()[:-4] for path in dataset.audios]
        self.clip_to_idx = {clip: idx for idx, clip in enumerate(self.clip_ids)}
        
        category_json_path = os.path.join(os.path.dirname(__file__), 'category.json')
        with open(category_json_path, 'r') as f:
            categories = json.load(f)
            
        self.clip_to_category = {}
        self.category_to_clips = {}
        for cat_id, clips in categories.items():
            valid_clips = []
            for c in clips:
                c_str = str(c).zfill(6)
                # check if c_str is in train/val/test
                if c_str in self.clip_to_idx:
                    valid_clips.append(c_str)
                    self.clip_to_category[c_str] = cat_id
            self.category_to_clips[cat_id] = valid_clips
            
        self.indices = list(range(len(self.clip_ids)))
        
    def __iter__(self):
        random.shuffle(self.indices)
        
        for i in range(0, len(self.indices), self.batch_size):
            chunk = self.indices[i : i + self.batch_size]
            if len(chunk) < self.batch_size:
                continue
                
            anchor_idx = chunk[0]
            anchor_clip = self.clip_ids[anchor_idx]
            
            if anchor_clip in self.clip_to_category:
                cat_id = self.clip_to_category[anchor_clip]
                candidates = [c for c in self.category_to_clips[cat_id] if c != anchor_clip]
                
                if len(candidates) >= self.s:
                    s_clips = random.sample(candidates, self.s)
                else: 
                    if len(candidates) == 0:
                        s_clips = [anchor_clip] * self.s
                    else:
                        s_clips = random.choices(candidates, k=self.s)
            else:
                s_clips = [anchor_clip] * self.s
                
            s_indices = [self.clip_to_idx[clip] for clip in s_clips]
            yield chunk + s_indices
            
    def __len__(self):
        return len(self.indices) // self.batch_size

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
        
        if opt.model == 'audioVisual' and hasattr(opt, 'spatial_num_samples'):
            sampler = CategoryBatchSampler(self.dataset, opt.batchSize, opt.spatial_num_samples)
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_sampler=sampler,
                num_workers=int(opt.nThreads))
        else:
            self.dataloader = torch.utils.data.DataLoader(
                self.dataset,
                batch_size=opt.batchSize,
                shuffle=True,
                num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data

