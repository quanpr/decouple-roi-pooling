import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pdb

class mimic_net(nn.Module):
	def __init__(self, tch_net, std_net):
		super(mimic_net, self).__init__()
		self.teacher_net = tch_net
		self.student_net = std_net

	def forward(self, im_data, im_info, gt_boxes, num_boxes):

		std_rois, cls_prob, bbox_pred, \
		rpn_loss_cls, rpn_loss_bbox, \
		RCNN_loss_cls, RCNN_loss_bbox, \
		rois_label, std_base_feat, std_pooled_feat = self.student_net(im_data, im_info, gt_boxes, num_boxes)

		rois, _, _, \
		_, _, \
		_, _, \
		_, base_feat, pooled_feat = self.teacher_net(im_data, im_info, gt_boxes, num_boxes, std_rois)

		base_feat, pooled_feat = base_feat.detach(), pooled_feat.detach()

		std_feat, tch_feat = [], []
		for i in range(im_data.shape[0]):
			std_img_feat, tch_img_feat = [], []
			dy, dx = base_feat.shape[2], base_feat.shape[3]
			for j in range(std_rois.shape[1]):

				stride = 17 if 'inception' in str(type(self.teacher_net)) else 16
				x1, y1, x2, y2 = int(std_rois[i][j][1].item()/stride),\
										 int(std_rois[i][j][2].item()/stride),\
										 int(std_rois[i][j][3].item()/stride),\
										 int(std_rois[i][j][4].item()/stride)

				if x2>dx or x1>dx or y2>dy or y1>dy:
					pdb.set_trace()
						
				if y2>y1 and x2>x1:
					std_img_feat.append(std_base_feat[i][:, y1:y2+1, x1:x2+1])
					tch_img_feat.append(base_feat[i][:, y1:y2+1, x1:x2+1])
						
			std_feat.append(std_img_feat)
			tch_feat.append(tch_img_feat)

		MSELoss = nn.MSELoss()
		mse_loss = torch.empty(1).zero_().type_as(im_data)
		for i in range(len(std_feat)):
			for j in range(len(std_feat[i])):
				if math.isnan(MSELoss(std_feat[i][j], tch_feat[i][j]).item()):
					pdb.set_trace()
				mse_loss += MSELoss(std_feat[i][j], tch_feat[i][j])
		
		mse_loss /= (std_rois.shape[0]*std_rois.shape[1])

		return rois, cls_prob, bbox_pred, rpn_loss_cls, rpn_loss_bbox, RCNN_loss_cls, RCNN_loss_bbox, rois_label, mse_loss