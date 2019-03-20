from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
lib_path = os.path.abspath(os.path.join('/home/prquan/faster-rcnn.pytorch/lib/'))
sys.path.append(lib_path)

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.utils.config import cfg
import math
import numpy as np
from model.roi_align.modules.roi_align import RoIAlignAvg


ANCHOR_RATIOS = [0.5,1,2]
ANCHOR_SCALES = [4,8,16,32]

def get_anchor_type(anchors, roi_size):
	# anchors: a list of size of anchors

	h, w = roi_size
	idx = 0
	iou = 0
	for i in range(anchors):
		h_hat, w_hat = anchors[i]

		inters = min(h_hat, h)*min(w_hat, w)
		iou_ = inters / (h*w+ h_hat*w_hat - inters)
		if iou_ > iou:
			idx = i
			iou = iou_

	return idx

def get_anchors_size(ratio, scales, stride=16):
	# h x w
	output = []
	for scale in scales:
		area = (scale*16)**2
		size = [((r*area)**0.5, (area/r)**0.5) for r in ratio]
		output.extend(size)
	return output

class gaussian_conv_pooling(nn.Module):
	"""docstring for gaussian_conv_pooling"""
	def __init__(self, channel, stride=16, scale=1, anchor_ratio=ANCHOR_RATIOS, anchor_scale=ANCHOR_SCALES, is_cuda=True):
		super(gaussian_conv_pooling, self).__init__()
		self.anchor_ratio = anchor_ratio
		self.anchor_scale = anchor_scale
		self.stride = stride
		self.filter_size = 7
		self.num_bins = 7
		self.scale = scale

		self.anchors_size = get_anchors_size(anchor_ratio, anchor_scale, stride)

		self.channel = channel
		self.RCNN_roi_align = RoIAlignAvg(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
		
		def gaussian_kernel(roi, scale=1, stride=16, filter_size=5, num_bins=7):
			# roi = (gpu, x1, y1, x2, y2)
			x1, y1, x2, y2 = int(roi[1]/stride)+1, int(roi[2]/stride)+1, \
							math.ceil(roi[3]/stride)-1, math.ceil(roi[4]/stride)-1

			bins = num_bins
			x_sigma, y_sigma = scale*(roi[3]-roi[1])/(bins*stride), \
								scale*(roi[4]-roi[2])/(bins*stride)
			
			kernel = torch.zeros(filter_size, filter_size).float()
			if x_sigma == 0 or y_sigma == 0:
				return kernel

			center = filter_size//2
			x_range = torch.arange(0.0, filter_size)
			y_range = torch.arange(0.0, filter_size).reshape(filter_size,-1)

			x_range = (x_range-center)**2/x_sigma**2
			y_range = (y_range-center)**2/y_sigma**2

			distance = x_range+y_range

			kernel += torch.exp(-distance/2)

			return kernel

		self.kernel = torch.zeros(len(self.anchors_size), channel, 1, self.filter_size, self.filter_size).float()

		#for (h, w) in self.anchors_size:
		for k in range(len(self.anchors_size)):
			h, w = self.anchors_size[k]

			kernel_ = torch.zeros(channel, 1, self.filter_size, self.filter_size).float()
			for i in range(channel):
				kernel_[i][0] += gaussian_kernel((0,0,0,w,h), scale=self.scale, stride=stride, filter_size=self.filter_size, num_bins=7)

			self.kernel[k] += kernel_

		self.kernel = self.kernel.cuda() if is_cuda else self.kernel
		num_anchor_type = len(self.anchors_size)

		hidden_dim = 4096
		input_dim = 256*7*7
		output_dim = len(self.anchors_size)

		self.use_softmax = True

		if self.use_softmax:
			self.conv_layers = nn.Sequential(
											nn.Linear(input_dim, hidden_dim),
											nn.BatchNorm1d(hidden_dim),
											nn.ReLU(inplace=True),

											nn.Linear(hidden_dim, output_dim),
											nn.BatchNorm1d(output_dim)														
											)
								
		else:
			self.conv_layers = nn.ModuleList([nn.Sequential(
															nn.Conv2d(channel, 64, kernel_size=1, padding=0),
															nn.BatchNorm2d(64),
															nn.ReLU(inplace=True),

															nn.Conv2d(64, 256, kernel_size=1, padding=0),
															nn.BatchNorm2d(256),
															)
															for i in range(num_anchor_type)])			


	def forward(self, base_feat, kernel, rois):
		num_kernel = len(kernel)
		# pooled_feat = base_feat.new(rois.shape[0]*rois.shape[1], self.channel, self.num_bins, self.num_bins).zero_().type_as(base_feat)
		num_roi_samples = rois.shape[0]*rois.shape[1]

		if self.use_softmax:
			filtered_feat = [F.conv2d(base_feat, kernel[i], padding=self.filter_size//2, groups=self.channel) for i in range(num_kernel)]
			filtered_feat = torch.stack(filtered_feat, 0)

			pooled_feat = self.RCNN_roi_align(base_feat, rois.view(-1, 5))

			pooled_feat_list = [self.RCNN_roi_align(filtered_feat[i], rois.view(-1, 5)) for i in range(num_kernel)]
			pooled_feat_list = torch.stack(pooled_feat_list, 0)

			pooled_feat_unfold = pooled_feat.view(num_roi_samples, -1)
			
			pre_softmax = self.conv_layers(pooled_feat_unfold)
			# pre_softmax = torch.sum(pre_softmax, (2,3,4), keepdim=True)
			post_softmax = F.softmax(pre_softmax, 1).permute(1, 0)
			post_softmax = post_softmax.reshape(num_kernel, num_roi_samples, 1, 1, 1)

			pooled_feat_list = pooled_feat_list*post_softmax
			pooled_feat = torch.sum(pooled_feat_list, 0)

		else:
			filtered_feat = [F.conv2d(base_feat, kernel[i], padding=self.filter_size//2, groups=self.channel) for i in range(num_kernel)]
			filtered_feat = torch.stack(filtered_feat, 0)

			filtered_feat = [self.conv_layers[i](filtered_feat[i]) for i in range(num_kernel)]
			filtered_feat = torch.stack(filtered_feat, 0)

			pooled_feat_list = [self.RCNN_roi_align(filtered_feat[i], rois.view(-1, 5)) for i in range(num_kernel)]
			pooled_feat_list = torch.stack(pooled_feat_list, 0)

			pooled_feat = torch.sum(pooled_feat_list, 0)
			# for i in range(len(pooled_feat_list)):
			# 	# pdb.set_trace()
			# 	pooled_feat += pooled_feat_list[i]

		return pooled_feat

if __name__ == '__main__':
	import time
	anchors_size = get_anchors_size(ANCHOR_RATIOS, ANCHOR_SCALES, 16)
	GP = gaussian_conv_pooling(channel=256, is_cuda=True)
	GP = GP.cuda()

	input_ = torch.randn(2,256,53,53).cuda()
	roi = torch.tensor([
						[[0.0,0.0,0.0,100.0,200.0],
						[0.0,0.0,0.0,200.0,300.0]],

						[[0.0,0.0,0.0,100.0,200.0],
						[0.0,0.0,0.0,200.0,300.0]]
						]).cuda()
	start = time.time()
	output_ = GP(input_, roi)
	print(time.time()-start)
	pdb.set_trace()

