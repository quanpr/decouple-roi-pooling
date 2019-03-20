# --------------------------------------------------------
# Pytorch multi-GPU Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jiasen Lu, Jianwei Yang, based on code from Ross Girshick
# --------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torch.utils.data.sampler import Sampler

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.utils.net_utils import weights_normal_init, save_net, load_net, \
			adjust_learning_rate, save_checkpoint, clip_gradient

from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from model.faster_rcnn.tiny_resnet import tiny_resnet
from model.faster_rcnn.inception import inception
from model.utils.trainer import trainer

def parse_args():
	"""
	Parse input arguments
	"""
	parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
	parser.add_argument('--dataset', dest='dataset',
											help='training dataset',
											default='pascal_voc', type=str)
	parser.add_argument('--net', dest='net',
										help='vgg16, res101',
										default='vgg16', type=str)
	parser.add_argument('--start_epoch', dest='start_epoch',
											help='starting epoch',
											default=1, type=int)
	parser.add_argument('--epochs', dest='max_epochs',
											help='number of epochs to train',
											default=20, type=int)
	parser.add_argument('--disp_interval', dest='disp_interval',
											help='number of iterations to display',
											default=100, type=int)
	parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
											help='number of iterations to display',
											default=10000, type=int)

	parser.add_argument('--save_dir', dest='save_dir',
											help='directory to save models', default="models",
											type=str)
	parser.add_argument('--nw', dest='num_workers',
											help='number of worker to load data',
											default=0, type=int)
	parser.add_argument('--cuda', dest='cuda',
											help='whether use CUDA',
											action='store_true')
	parser.add_argument('--ls', dest='large_scale',
											help='whether use large imag scale',
											action='store_true')                      
	parser.add_argument('--mGPUs', dest='mGPUs',
											help='whether use multiple GPUs',
											action='store_true')
	parser.add_argument('--bs', dest='batch_size',
											help='batch_size',
											default=1, type=int)
	parser.add_argument('--cag', dest='class_agnostic',
											help='whether perform class_agnostic bbox regression',
											action='store_true')

# config optimization
	parser.add_argument('--o', dest='optimizer',
											help='training optimizer',
											default="sgd", type=str)
	parser.add_argument('--lr', dest='lr',
											help='starting learning rate',
											default=0.001, type=float)
	parser.add_argument('--lr_decay_step', dest='lr_decay_step',
											help='step to do learning rate decay, unit is epoch',
											default=5, type=int)
	parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
											help='learning rate decay ratio',
											default=0.1, type=float)
# set training session
	parser.add_argument('--s', dest='session',
											help='training session',
											default=1, type=int)

# resume trained model
	parser.add_argument('--r', dest='resume',
											help='resume checkpoint or not',
											action='store_true')
	parser.add_argument('--checksession', dest='checksession',
											help='checksession to load model',
											default=1, type=int)
	parser.add_argument('--checkepoch', dest='checkepoch',
											help='checkepoch to load model',
											default=1, type=int)
	parser.add_argument('--checkpoint', dest='checkpoint',
											help='checkpoint to load model',
											default=0, type=int)
# log and diaplay
	parser.add_argument('--use_tfb', dest='use_tfboard',
											help='whether use tensorboard',
											action='store_true')
	parser.add_argument('--name', dest='name',
											help='name of models', default="faster_rcnn_curr.pth",
											type=str)
	parser.add_argument('--mm', dest='mimic',
											help='whether perform mimicking',
											action='store_true')
	parser.add_argument('--layers', dest='layers',
											help='tiny network layers',
											default=101, type=int)
	parser.add_argument('--save_model', dest='save_model',
											help='name to save', default="my_faster_rcnn_curr.pth",
											type=str)
	parser.add_argument('--recall', dest='evl_rec',
											help='whether evaluate recall',
											action='store_true')
	parser.add_argument('--decouple', dest='decouple',
											help='whether to use decouple roi pooling',
											action='store_true')
	parser.add_argument('--scale', dest='scale',
											help='scale of sigma with respect to ROI',
											default=1.0, type=float)
	args = parser.parse_args()
	return args


class sampler(Sampler):
	def __init__(self, train_size, batch_size):
		self.num_data = train_size
		self.num_per_batch = int(train_size / batch_size)
		self.batch_size = batch_size
		self.range = torch.arange(0,batch_size).view(1, batch_size).long()
		self.leftover_flag = False
		if train_size % batch_size:
			self.leftover = torch.arange(self.num_per_batch*batch_size, train_size).long()
			self.leftover_flag = True

	def __iter__(self):
		rand_num = torch.randperm(self.num_per_batch).view(-1,1) * self.batch_size
		self.rand_num = rand_num.expand(self.num_per_batch, self.batch_size) + self.range

		self.rand_num_view = self.rand_num.view(-1)

		if self.leftover_flag:
			self.rand_num_view = torch.cat((self.rand_num_view, self.leftover),0)

		return iter(self.rand_num_view)

	def __len__(self):
		return self.num_data

def main():

	args = parse_args()

	print('Called with args:')
	print(args)

	if args.dataset == "pascal_voc":
			args.imdb_name = "voc_2007_trainval"
			args.imdbval_name = "voc_2007_test"
			args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif args.dataset == "pascal_voc_0712":
			args.imdb_name = "voc_2007_trainval+voc_2012_trainval"
			args.imdbval_name = "voc_2007_test"
			args.set_cfgs = ['ANCHOR_SCALES', '[8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '20']
	elif args.dataset == "coco":
			args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
			args.imdbval_name = "coco_2014_minival"
			args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']
	elif args.dataset == "imagenet":
			args.imdb_name = "imagenet_train"
			args.imdbval_name = "imagenet_val"
			args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '30']
	elif args.dataset == "vg":
			# train sizes: train, smalltrain, minitrain
			# train scale: ['150-50-20', '150-50-50', '500-150-80', '750-250-150', '1750-700-450', '1600-400-20']
			args.imdb_name = "vg_150-50-50_minitrain"
			args.imdbval_name = "vg_150-50-50_minival"
			args.set_cfgs = ['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]', 'MAX_NUM_GT_BOXES', '50']

	args.cfg_file = "cfgs/{}_ls.yml".format(args.net) if args.large_scale else "cfgs/{}.yml".format(args.net)

	if args.cfg_file is not None:
		cfg_from_file(args.cfg_file)
	if args.set_cfgs is not None:
		cfg_from_list(args.set_cfgs)

	print('Using config:')
	pprint.pprint(cfg)
	np.random.seed(cfg.RNG_SEED)

	#torch.backends.cudnn.benchmark = True
	if torch.cuda.is_available() and not args.cuda:
		print("WARNING: You have a CUDA device, so you should probably run with --cuda")

	# train set
	# -- Note: Use validation set and disable the flipped to enable faster loading.
	cfg.TRAIN.USE_FLIPPED = True
	cfg.USE_GPU_NMS = args.cuda
	imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
	train_size = len(roidb)

	print('{:d} roidb entries'.format(len(roidb)))

	output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	sampler_batch = sampler(train_size, args.batch_size)

	# These models are pytorch pretrained with RGB channel
	rgb = True if args.net in ('res18', 'res34', 'res50','inception', 'res101') else False

	dataset = roibatchLoader(roidb, ratio_list, ratio_index, args.batch_size, \
													 imdb.num_classes, training=True, rgb=rgb)

	dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
														sampler=sampler_batch, num_workers=args.num_workers)
	
	if args.cuda:
		cfg.CUDA = True

	# initilize the network here.
	if args.net == 'vgg16':
		fasterRCNN = vgg16(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'inception':
		fasterRCNN = inception(imdb.classes, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res18':
		fasterRCNN = resnet(imdb.classes, 18, pretrained=True, class_agnostic=args.class_agnostic, layer=args.layers, decouple=args.decouple, args=args)
	elif args.net == 'res101':
		fasterRCNN = resnet(imdb.classes, 101, pretrained=True, class_agnostic=args.class_agnostic)
	elif args.net == 'res50':
		fasterRCNN = resnet(imdb.classes, 50, pretrained=True, class_agnostic=args.class_agnostic, layer=args.layers)
	elif args.net == 'res152':
		fasterRCNN = resnet(imdb.classes, 152, pretrained=True, class_agnostic=args.class_agnostic)
	else:
		print("network is not defined")
		pdb.set_trace()

	fasterRCNN.create_architecture()

	lr = cfg.TRAIN.LEARNING_RATE
	lr = args.lr
	#tr_momentum = cfg.TRAIN.MOMENTUM
	#tr_momentum = args.momentum

	params = []
	for key, value in dict(fasterRCNN.named_parameters()).items():
		if value.requires_grad:
			if 'bias' in key:
				params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
								'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
			else:
				params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]

	if args.optimizer == "adam":
		lr = lr * 0.1
		optimizer = torch.optim.Adam(params)

	elif args.optimizer == "sgd":
		optimizer = torch.optim.SGD(params, momentum=cfg.TRAIN.MOMENTUM)

	if args.cuda:
		fasterRCNN.cuda()

	if args.resume:
		load_name = os.path.join(output_dir, args.name)
		print("loading checkpoint %s" % (load_name))
		checkpoint = torch.load(load_name)
		# args.session = checkpoint['session']
		# args.start_epoch = checkpoint['epoch']
		fasterRCNN.load_state_dict(checkpoint['model'])
		# optimizer.load_state_dict(checkpoint['optimizer'])
		# lr = optimizer.param_groups[0]['lr']
		# if 'pooling_mode' in checkpoint.keys():
		# 	cfg.POOLING_MODE = checkpoint['pooling_mode']
		print("loaded checkpoint %s" % (load_name))

	if args.mGPUs:
		fasterRCNN = nn.DataParallel(fasterRCNN)

	if not args.mimic:
		trainer(fasterRCNN, dataloader, optimizer, args)
	else:
		student_net = tiny_resnet(imdb.classes, 101, pretrained=False, class_agnostic=args.class_agnostic)
		student_net.create_architecture()
		if args.cuda:
			student_net.cuda()
		if args.mGPUs:
			student_net = nn.DataParallel(student_net)
		mm_trainer(fasterRCNN, student_model, dataloader, optimizer, args)


if __name__ == '__main__':
	main()