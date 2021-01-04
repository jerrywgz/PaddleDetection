import numpy as np
from ppdet.core.workspace import register


@register
class Mask(object):
    __inject__ = ['mask_target_generator']

    def __init__(self, mask_target_generator):
        super(Mask, self).__init__()
        self.mask_target_generator = mask_target_generator

    def __call__(self, inputs, rois, targets):
        mask_rois, mask_rois_num, mask_label, mask_target, mask_index, mask_weight = self.generate_mask_target(
            inputs, rois, targets)
        mask_rois = (mask_rois, mask_rois_num)
        return mask_rois, mask_label, mask_target, mask_index, mask_weight

    def generate_mask_target(self, inputs, rois, targets):
        labels_int32 = targets['labels_int32']
        sampled_gt_inds = targets['sampled_gt_inds']
        proposals, proposals_num = rois
        mask_rois, mask_rois_num, mask_label, mask_target, mask_index, mask_weight = self.mask_target_generator(
            gt_segms=inputs['gt_poly'],
            rois=proposals,
            rois_num=proposals_num,
            labels_int32=labels_int32,
            sampled_gt_inds=sampled_gt_inds)
        return mask_rois, mask_rois_num, mask_label, mask_target, mask_index, mask_weight

    def get_targets(self):
        return self.mask_int32
