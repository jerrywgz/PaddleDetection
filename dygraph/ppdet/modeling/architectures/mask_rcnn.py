# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved. 
#   
# Licensed under the Apache License, Version 2.0 (the "License");   
# you may not use this file except in compliance with the License.  
# You may obtain a copy of the License at   
#   
#     http://www.apache.org/licenses/LICENSE-2.0    
#   
# Unless required by applicable law or agreed to in writing, software   
# distributed under the License is distributed on an "AS IS" BASIS, 
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  
# See the License for the specific language governing permissions and   
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
from ppdet.core.workspace import register
from .meta_arch import BaseArch
from ppdet.py_op.post_process import bbox_post_process, mask_post_process

__all__ = ['MaskRCNN']


@register
class MaskRCNN(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'anchor',
        'proposal',
        'mask',
        'backbone',
        'neck',
        'rpn_head',
        'bbox_head',
        'mask_head',
        'bbox_post_process',
        'mask_post_process',
    ]

    def __init__(self,
                 anchor,
                 proposal,
                 mask,
                 backbone,
                 rpn_head,
                 bbox_head,
                 mask_head,
                 bbox_post_process,
                 mask_post_process,
                 neck=None):
        super(MaskRCNN, self).__init__()
        self.anchor = anchor
        self.proposal = proposal
        self.mask = mask
        self.backbone = backbone
        self.neck = neck
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head
        self.mask_head = mask_head
        self.bbox_post_process = bbox_post_process
        self.mask_post_process = mask_post_process

    def model_arch(self):
        # Backbone
        #import pickle
        #im = pickle.load(open('image.npy', 'rb'))
        #gt_bbox = pickle.load(open('gt_boxes.npy', 'rb'))
        #gt_segms = pickle.load(open('gt_segms.npy', 'rb'))
        #self.inputs['image'] = paddle.to_tensor(im)
        #self.inputs['gt_poly'] = paddle.to_tensor(gt_segms)
        #self.inputs['gt_bbox'] = paddle.to_tensor(gt_bbox).unsqueeze(0)
        body_feats = self.backbone(self.inputs)
        spatial_scale = 1. / 16

        # Neck
        if self.neck is not None:
            body_feats, spatial_scale = self.neck(body_feats)

        # RPN
        # rpn_head_out contains (rpn_rois_score, rpn_rois_delta)
        self.rpn_rois_score, self.rpn_rois_delta = self.rpn_head(body_feats)

        # Anchor
        # anchor_out returns a list,
        # each element contains (anchor, anchor_var)
        self.anchor_out = self.anchor(body_feats)

        # Proposal RoI 
        # compute targets here when training
        self.rois = self.proposal(self.inputs, self.rpn_rois_score,
                                  self.rpn_rois_delta, self.anchor_out)
        # BBox Head
        bbox_feat, self.bbox_head_out, bbox_head_feat_func = self.bbox_head(
            body_feats, self.rois, spatial_scale)

        mask_index = None
        if self.inputs['mode'] == 'infer':
            bbox_pred, bboxes = self.bbox_head.get_prediction(
                self.bbox_head_out, self.rois)
            # Refine bbox by the output from bbox_head at test stage
            self.bboxes = self.bbox_post_process(bbox_pred, bboxes,
                                                 self.inputs['im_shape'])
        else:
            # Proposal RoI for Mask branch
            # bboxes update at training stage only
            bbox_targets = self.proposal.get_targets()[0]
            self.bboxes, self.mask_label, self.mask_target, mask_index, self.mask_weight = self.mask(
                self.inputs, self.rois, bbox_targets)

        # Mask Head 
        self.mask_head_out = self.mask_head(self.inputs, body_feats,
                                            self.bboxes, bbox_feat, mask_index,
                                            spatial_scale, bbox_head_feat_func)

    def get_loss(self, ):
        loss = {}

        # RPN loss
        rpn_loss_inputs = self.anchor.generate_loss_inputs(
            self.inputs, self.rpn_rois_score, self.rpn_rois_delta,
            self.anchor_out)
        loss_rpn = self.rpn_head.get_loss(rpn_loss_inputs)
        loss.update(loss_rpn)

        # BBox loss
        bbox_targets = self.proposal.get_targets()
        loss_bbox = self.bbox_head.get_loss([self.bbox_head_out], bbox_targets,
                                            self.rois)
        loss.update(loss_bbox)

        # Mask loss
        loss_mask = self.mask_head.get_loss(self.mask_head_out, self.mask_label,
                                            self.mask_target, self.mask_weight)
        loss.update(loss_mask)
        total_loss = paddle.add_n(list(loss.values()))
        loss.update({'loss': total_loss})
        return loss

    def get_pred(self):
        self.bboxes = bbox_post_process(self.bboxes, self.inputs)
        mask = mask_post_process(self.mask_head_out[:, 0, :, :], self.bboxes,
                                 self.inputs)
        bbox, bbox_num = self.bboxes
        label = bbox[:, 0]
        score = bbox[:, 1]
        bbox = bbox[:, 2:]
        output = {
            'bbox': bbox,
            'score': score,
            'label': label,
            'bbox_num': bbox_num,
            'mask': mask,
        }
        return output
