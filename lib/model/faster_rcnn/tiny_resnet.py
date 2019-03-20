from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from model.utils.config import cfg
from model.faster_rcnn.faster_rcnn import _fasterRCNN

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import torch.utils.model_zoo as model_zoo
import pdb

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
			 'resnet152']


model_urls = {
	'resnet18': 'https://s3.amazonaws.com/pytorch/models/resnet18-5c106cde.pth',
	'resnet34': 'https://s3.amazonaws.com/pytorch/models/resnet34-333f7ec4.pth',
	'resnet50': 'https://s3.amazonaws.com/pytorch/models/resnet50-19c8e357.pth',
	'resnet101': 'https://s3.amazonaws.com/pytorch/models/resnet101-5d3b4d8f.pth',
	'resnet152': 'https://s3.amazonaws.com/pytorch/models/resnet152-b121ed2d.pth',
}

def conv3x3(in_planes, out_planes, stride=1):
	"3x3 convolution with padding"
	return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
					 padding=1, bias=False)


class BasicBlock(nn.Module):
	expansion = 1

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(BasicBlock, self).__init__()
		self.conv1 = conv3x3(inplanes, planes, stride)
		self.bn1 = nn.BatchNorm2d(planes)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(planes, planes)
		self.bn2 = nn.BatchNorm2d(planes)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class Bottleneck(nn.Module):
	expansion = 4

	def __init__(self, inplanes, planes, stride=1, downsample=None):
		super(Bottleneck, self).__init__()
		self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False) # change
		self.bn1 = nn.BatchNorm2d(planes)
		self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, # change
								 padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(planes)
		self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
		self.bn3 = nn.BatchNorm2d(planes * 4)
		self.relu = nn.ReLU(inplace=True)
		self.downsample = downsample
		self.stride = stride

	def forward(self, x):
		residual = x

		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.downsample is not None:
			residual = self.downsample(x)

		out += residual
		out = self.relu(out)

		return out


class TinyResNet(nn.Module):
	def __init__(self, block, layers, num_classes=1000, shrink=2):
		self.shrink = shrink
		self.inplanes = 64//self.shrink
		super(TinyResNet, self).__init__()
		self.conv1 = nn.Conv2d(3, 64//self.shrink, kernel_size=7, stride=2, padding=3,
								 bias=False)
		self.bn1 = nn.BatchNorm2d(64//self.shrink)
		self.relu = nn.ReLU(inplace=True)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=0, ceil_mode=True) # change
		self.layer1 = self._make_layer(block, 64//self.shrink, layers[0])
		self.layer2 = self._make_layer(block, 128//self.shrink, layers[1], stride=2)
		self.layer3 = self._make_layer(block, 256//self.shrink, layers[2], stride=2)
		self.layer4 = self._make_layer(block, 512//self.shrink, layers[3], stride=2)
		# it is slightly better whereas slower to set stride = 1
		# self.layer4 = self._make_layer(block, 512, layers[3], stride=1)
		self.avgpool = nn.AvgPool2d(7)
		self.fc = nn.Linear(512 * block.expansion//self.shrink, num_classes)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def _make_layer(self, block, planes, blocks, stride=1):
		downsample = None
		if stride != 1 or self.inplanes != planes * block.expansion:
			downsample = nn.Sequential(
				nn.Conv2d(self.inplanes, planes * block.expansion,
							kernel_size=1, stride=stride, bias=False),
				nn.BatchNorm2d(planes * block.expansion),
			)

		layers = []
		layers.append(block(self.inplanes, planes, stride, downsample))
		self.inplanes = planes * block.expansion
		for i in range(1, blocks):
			layers.append(block(self.inplanes, planes))

		return nn.Sequential(*layers)

	def forward(self, x):
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.maxpool(x)

		x = self.layer1(x)
		x = self.layer2(x)
		x = self.layer3(x)
		x = self.layer4(x)

		x = self.avgpool(x)
		x = x.view(x.size(0), -1)
		x = self.fc(x)

		return x


def tiny_resnet18(pretrained=False):
	"""Constructs a ResNet-18 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = TinyResNet(BasicBlock, [2, 2, 2, 2])
	return model


def tiny_resnet34(pretrained=False):
	"""Constructs a ResNet-34 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = TinyResNet(BasicBlock, [3, 4, 6, 3])
	return model


def tiny_resnet50(pretrained=False):
	"""Constructs a ResNet-50 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = TinyResNet(Bottleneck, [3, 4, 6, 3])
	return model


def tiny_resnet101(pretrained=False):
	"""Constructs a ResNet-101 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = TinyResNet(Bottleneck, [3, 4, 23, 3])
	return model


def tiny_resnet152(pretrained=False):
	"""Constructs a ResNet-152 model.
	Args:
		pretrained (bool): If True, returns a model pre-trained on ImageNet
	"""
	model = TinyResNet(Bottleneck, [3, 8, 36, 3])
	return model

class tiny_resnet(_fasterRCNN):
	def __init__(self, classes, num_layers=101, pretrained=False, class_agnostic=False, shrink=2, mimic=False, layer=101):
		self.layer = layer
		self.dout_base_model = 256//shrink if self.layer in (18, 34) else 1024//shrink
		self.pretrained = pretrained
		self.class_agnostic = class_agnostic

		_fasterRCNN.__init__(self, classes, class_agnostic, shrink, mimic)

	def _init_modules(self):
		expansion = 1 if self.layer in (18, 34) else 4

		print('tiny network backbone: resnet-{}'.format(self.layer))
		if self.layer == 18:
			resnet = tiny_resnet18(pretrained=False)
		elif self.layer == 34:
			resnet = tiny_resnet34(pretrained=False)
		elif self.layer == 50:
			resnet = tiny_resnet50(pretrained=False)
		else:
			resnet = tiny_resnet101(pretrained=False)
		#resnet = resnet50(pretrained=False)

		# Build resnet.
		self.RCNN_base = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu,
			resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3)

		self.RCNN_top = nn.Sequential(resnet.layer4)

		self.RCNN_cls_score = nn.Linear(512*expansion//self.shrink, self.n_classes)
		if self.class_agnostic:
			self.RCNN_bbox_pred = nn.Linear(512*expansion//self.shrink, 4)
		else:
			self.RCNN_bbox_pred = nn.Linear(512*expansion//self.shrink, 4 * self.n_classes)

		in_channels = self.dout_base_model
		self.transf_layer = nn.Sequential(
										nn.Conv2d(in_channels, 256*expansion,
											kernel_size=3, stride=1,padding=1,
											dilation=1, bias=True),
										nn.BatchNorm2d(256*expansion)
										)

		# initialize transform layer
		for m in self.transf_layer:
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()		

	# def rpn_train(self, mode=True):
	# 	# Override train so that the training mode is set as we want
	# 	nn.Module.train(self, mode)
	# 	if mode:
	# 		# Set fixed blocks to be in eval mode
	# 		self.RCNN_cls_score.eval()
	# 		self.RCNN_bbox_pred.eval()

	def _head_to_tail(self, pool5):
		fc7 = self.RCNN_top(pool5).mean(3).mean(2)
		return fc7
