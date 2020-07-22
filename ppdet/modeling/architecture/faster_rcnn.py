from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from paddle import fluid
from ppdet.core.workspace import register
from .meta_arch import BaseArch

__all__ = ['FasterRCNN']


@register
class FasterRCNN(BaseArch):
    __category__ = 'architecture'
    __inject__ = [
        'neck',
        'anchor',
        'proposal',
        'backbone',
        'rpn_head',
        'bbox_head',
    ]

    def __init__(self, anchor, proposal, backbone, rpn_head, bbox_head, *args,
                 **kwargs):
        super(FasterRCNN, self).__init__(*args, **kwargs)
        self.anchor = anchor
        self.proposal = proposal
        self.backbone = backbone
        self.rpn_head = rpn_head
        self.bbox_head = bbox_head

    def model_arch(self, ):
        # Backbone
        body_feats = self.backbone(self.gbd)

        # Neck
        if self.neck is not None:
            body_feats = self.neck(body_feats)

        # RPN
        # rpn_head returns two list: rpn_feat, rpn_head_out 
        # each element in rpn_feats contains 
        # each element in rpn_head_out contains (rpn_rois_score, rpn_rois_delta)
        rpn_feats, self.rpn_head_out = self.rpn_head(bb_out)

        # Anchor
        # anchor_out returns a list,
        # each element contains (anchor, anchor_var)
        self.anchor_out = self.anchor(rpn_feats)

        # Proposal RoI
        rois, rois_num = self.proposal(self.gbd, self.rpn_head_out,
                                       self.anchor_out)

        # BBox Head
        # bboxhead_out returns a list
        # each element contains (score, delta)
        self.bboxhead_out = self.bbox_head(body_feats, rois, rois_num)

    def loss(self, ):
        loss = {}
        rpn_loss_inputs = self.anchor.generate_loss_inputs(
            self.gbd, self.rpn_head_out, self.anchor_out)
        rpn_loss = self.rpn_head.loss(rpn_loss_inputs)
        loss.update(rpn_loss)

        targets = self.proposal.get_targets()
        bbox_loss = self.bbox_head.loss(self.bboxhead_out, targets)
        loss.update(bbox_loss)
        total_loss = fluid.layers.sum(loss.values())
        loss.update({'loss': total_loss})
        return loss

    def infer(self, ):
        proposals = self.proposal.get_proposals()
        bbox_out = self.proposal.post_process(
            self.gbd, self.bboxhead_out, proposals, self.bbox_head.cls_agnostic)
        outs = {
            "bbox": bbox_out['predicted_bbox'].numpy(),
            "bbox_nums": bbox_out['predicted_bbox_nums'].numpy(),
            'im_id': self.gbd['im_id'].numpy(),
            'im_shape': self.gbd['im_shape'].numpy()
        }
        return outs
