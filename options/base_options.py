#!/usr/bin/env python

# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os
from util import util
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--visual_backbone', type=str, default='dinov2_vitb14_reg', choices=['dinov2_vitb14_reg', 'dinov3_vitb16'], help='frozen visual backbone')
		self.parser.add_argument('--dinov3_repo', type=str, default='/home/huskypaul/dinov3', help='local clone of facebookresearch/dinov3')
		self.parser.add_argument('--dinov3_weights', type=str, default='/home/huskypaul/dinov3_weights/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth', help='official DINOv3 ViT-B/16 LVD-1689M .pth checkpoint path or URL')
		self.parser.add_argument('--split_file', type=str, default='', help='path to the JSON split file (e.g. splits/unseen1.json) with train/val/test keys')
		self.parser.add_argument('--audio_dir', type=str, default='/home/huskypaul/FAIR-Play/audios', help='path to the audio directory')
		self.parser.add_argument('--video_dir', type=str, default='/home/huskypaul/FAIR-Play/frames', help='path to the video frames directory')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--name', type=str, default='spatialAudioVisual', help='name of the experiment. It decides where to store models')
		self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')
		self.parser.add_argument('--model', type=str, default='audioVisual', help='chooses how datasets are loaded.')
		self.parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
		self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
		self.parser.add_argument('--audio_sampling_rate', default=16000, type=int, help='audio sampling rate')
		self.parser.add_argument('--audio_length', default=0.63, type=float, help='audio length, default 0.63s')
		self.enable_data_augmentation = True
		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()

		self.opt.mode = self.mode
		self.opt.isTrain = self.isTrain
		self.opt.enable_data_augmentation = self.enable_data_augmentation

		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])


		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		return self.opt
