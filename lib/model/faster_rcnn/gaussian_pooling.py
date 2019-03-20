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


def gaussian_pooling(base_feat, rois, scale=1, stride=None):
	# def __init__(self, scale=1, stride=None):
	# 	self.scale = scale
	# 	self.stride = cfg.FEAT_STRIDE[0] if not stride else stride
	# 	self.num_bins = cfg.POOLING_SIZE

	def gaussian_kernel(size, roi, scale=1, stride=16, num_bins=7):
		# roi = (gpu, x1, y1, x2, y2)
		H, W = size
		x1, y1, x2, y2 = int(roi[1].item()/stride)+1, int(roi[2].item()/stride)+1, \
						math.ceil(roi[3].item()/stride)-1, math.ceil(roi[4].item()/stride)-1

		bins = num_bins
		x_sigma, y_sigma = scale*(roi[3].item()-roi[1].item())/(bins*stride), \
							scale*(roi[4].item()-roi[2].item())/(bins*stride)
		
		kernel = np.zeros((bins, bins, H, W))
		if x_sigma == 0 or y_sigma == 0:
			kernel = torch.from_numpy(kernel).float()
			return kernel
		for i in range(bins):
			for j in range(bins):
				y_mean = (y2-y1)/(2*bins)*(2*i+1) + y1
				x_mean = (x2-x1)/(2*bins)*(2*j+1) + x1

				x_range = np.arange(W)-x_mean
				x_range = x_range**2/(x_sigma**2)

				y_range = np.arange(H)-y_mean
				y_range = (y_range**2/(y_sigma**2)).reshape(H, 1)

				distance = y_range + x_range
				kernel[i][j] = np.exp(-distance/2)

		kernel = torch.from_numpy(kernel).float()
		return kernel

	scale_ = scale
	stride_ = cfg.FEAT_STRIDE[0] if not stride else stride
	num_bins_ = cfg.POOLING_SIZE

	# base_feat: (N x C x H x W)
	batch_size, num_rois = rois.shape[0], rois.shape[1]
	channels = base_feat.shape[1]
	pooled_feat = base_feat.new(batch_size*num_rois, channels, num_bins_, num_bins_).zero_().type_as(base_feat)

	for i in range(batch_size):
		H, W = base_feat[i].shape[1], base_feat[i].shape[2]
		for j in range(num_rois):

			kernel = gaussian_kernel((H, W), rois[i][j], scale_, stride_, num_bins_)
			kernel = kernel.cuda() if base_feat.is_cuda else kernel
			
			for h in range(num_bins_):
				for w in range(num_bins_):
					pooled_feat[i*num_rois+j, :, h, w] += torch.sum(torch.sum(base_feat[i]*kernel[h][w], dim=2), dim=1)
	
	return pooled_feat

def gaussian_conv_pooling(base_feat, roi_align, rois, scale=1, stride=None, filter_size=5):
	# def __init__(self, scale=1, stride=None):
	# 	self.scale = scale
	# 	self.stride = cfg.FEAT_STRIDE[0] if not stride else stride
	# 	self.num_bins = cfg.POOLING_SIZE

	def gaussian_kernel(roi, scale=1, stride=16, filter_size=5, num_bins=7):
		# roi = (gpu, x1, y1, x2, y2)

		x1, y1, x2, y2 = int(roi[1].item()/stride)+1, int(roi[2].item()/stride)+1, \
						math.ceil(roi[3].item()/stride)-1, math.ceil(roi[4].item()/stride)-1

		bins = num_bins
		x_sigma, y_sigma = scale*(roi[3].item()-roi[1].item())/(bins*stride), \
							scale*(roi[4].item()-roi[2].item())/(bins*stride)
		
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

	scale_ = scale
	stride_ = cfg.FEAT_STRIDE[0] if not stride else stride
	num_bins_ = cfg.POOLING_SIZE

	# base_feat: (N x C x H x W)
	batch_size, num_rois = rois.shape[0], rois.shape[1]
	channels = base_feat.shape[1]
	pooled_feat = base_feat.new(batch_size*num_rois, channels, num_bins_, num_bins_).zero_().type_as(base_feat)
	
	H, W = base_feat.shape[2], base_feat.shape[3]
	for i in range(batch_size):
		
		# kernel = base_feat.new(channels, channels, filter_size, filter_size).zero_().type_as(base_feat)
		for j in range(num_rois):
			conv_feat = base_feat.new(channels, H, W).zero_().type_as(base_feat)
			ker_ = gaussian_kernel(rois[i][j], scale_, stride_, filter_size, num_bins_)
			ker_ = ker_.cuda() if base_feat.is_cuda else ker_
			# kernel[j][j] += ker_
			# kernel = kernel.cuda() if base_feat.is_cuda else kernel
			
			# conv_feat = F.conv2d(base_feat[i], kernel, padding=filter_size//2)
			# pooled_feat[i*num_rois+j] += roi_align(conv_feat, rois[i][j].view(-1, 5))
			for k in range(channels):
				feat = base_feat[i][k]
				feat = torch.unsqueeze(torch.unsqueeze(feat, 0), 0)
				feat = F.conv2d(feat, torch.unsqueeze(kernel,0), padding=filter_size//2)
				conv_feat[k] += feat[0][0]
			pooled_feat[i*num_rois+j] += roi_align(conv_feat, rois[i][j].view(-1, 5))

	return pooled_feat

def visualize(kernel, rois):
	import matplotlib
	import matplotlib.pyplot as plt
	import matplotlib.patches as patches

	ke = torch.sum(torch.sum(kernel, dim=0), dim=0)
	fig,ax = plt.subplots(1)
	rect = patches.Rectangle((rois[1].item()//16+1,rois[2].item()//16+1),\
		math.ceil(rois[3].item()/16-rois[1].item()/16)-1, math.ceil(rois[4].item()/16-rois[2].item()/16)-1,linewidth=1,edgecolor='r',facecolor='none')
	ax.add_patch(rect)
	plt.imshow(ke.cpu().numpy(), cmap='hot', interpolation='nearest')
	plt.savefig('./test.png')


if __name__ == '__main__':
	# import matplotlib
	# import matplotlib.pyplot as plt
	# import matplotlib.patches as patches

	# gp = gaussian_pooling(scale=1, stride=16)
	# rois = torch.tensor([0, 100, 200, 500, 600])
	# size = (38, 50)
	# kernel = gp.gaussian_kernel(size, rois)
	# ke = torch.sum(torch.sum(kernel, dim=0), dim=0)
	# fig,ax = plt.subplots(1)
	# rect = patches.Rectangle((rois[1].item()//16+1,rois[2].item()//16+1),\
	# 	math.ceil(rois[3].item()/16-rois[1].item()/16)-1, math.ceil(rois[4].item()/16-rois[2].item()/16)-1,linewidth=1,edgecolor='r',facecolor='none')
	# ax.add_patch(rect)
	# plt.imshow(ke.numpy(), cmap='hot', interpolation='nearest')
	# plt.savefig('./test.png')
	# pdb.set_trace()
	pass
