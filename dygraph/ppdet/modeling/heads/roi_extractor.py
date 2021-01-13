#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
from ppdet.core.workspace import register
from ppdet.modeling import ops
from ppdet.py_op.bbox import bbox_area


@register
class RoIAlign(object):
    def __init__(self,
                 resolution=14,
                 sampling_ratio=0,
                 canconical_level=4,
                 canonical_size=224,
                 start_level=0,
                 end_level=3,
                 aligned=False):
        super(RoIAlign, self).__init__()
        self.resolution = resolution
        self.sampling_ratio = sampling_ratio
        self.canconical_level = canconical_level
        self.canonical_size = canonical_size
        self.start_level = start_level
        self.end_level = end_level
        self.aligned = aligned

    def assign_boxes_to_levels(self, roi):
        offset = 2
        area = bbox_area(roi)
        box_sizes = paddle.sqrt(area)
        level_assignments = paddle.floor(self.canconical_level + paddle.log2(
            box_sizes / self.canonical_size + 1e-8) - offset)
        level_assignments = paddle.clip(
            level_assignments, min=self.start_level,
            max=self.end_level).cast('int32')
        return level_assignments

    def expand_rois_num(self, rois_num):
        N = rois_num.shape[0]
        rois_batch_id = []
        for i in range(N):
            batch_id = paddle.full([1], i, dtype='int32')
            batch_id = paddle.tile(batch_id, [rois_num[i]])
            rois_batch_id.append(batch_id)
        return paddle.concat(rois_batch_id)

    def __call__(self, feats, rois, spatial_scale):
        roi, rois_num = rois
        roi = paddle.concat(roi) if len(roi) > 1 else roi[0]

        if self.start_level == self.end_level:
            roi = roi[0]
            rois_feat = ops.roi_align(
                feats[self.start_level],
                roi,
                self.resolution,
                spatial_scale,
                rois_num=rois_num,
                aligned=self.aligned)
        else:
            """
            offset = 2

            level_assignments = self.assign_boxes_to_levels(roi)
            #rois_batch_id = self.expand_rois_num(rois_num)
            num_boxes = roi.shape[0]
            num_channels = feats[0].shape[1]
            rois_feat = paddle.zeros(
                (num_boxes, num_channels, self.resolution, self.resolution), dtype='float32')

            for lvl in range(self.start_level, self.end_level + 1):
                mask = level_assignments == lvl
                rois_num_level = mask.cast('int32').sum()
                #mask_split = paddle.tensor.split(mask, rois_num)
                #for bs in mask_split:
                #    rois_num_level.append(mask_split[bs].cast('int32').sum())
                #rois_num_level = paddle.concat(rois_num_level)
                inds = paddle.nonzero(mask)
                if inds.numel() == 0:
                    continue
                roi_level = paddle.gather(roi, inds)
                
                rois_feat_level  = ops.roi_align(feats[lvl], roi_level, self.resolution, 
                             spatial_scale[lvl], sampling_ratio=self.sampling_ratio, 
                             rois_num=rois_num_level, aligned=self.aligned) 
                rois_feat = paddle.scatter(rois_feat, inds.flatten(), rois_feat_level)
           

            """
            offset = 2
            k_min = self.start_level + offset
            k_max = self.end_level + offset
            rois_dist, restore_index, rois_num_dist = ops.distribute_fpn_proposals(
                roi,
                k_min,
                k_max,
                self.canconical_level,
                self.canonical_size,
                rois_num=rois_num)
            rois_feat_list = []
            for lvl in range(self.start_level, self.end_level + 1):
                roi_feat = ops.roi_align(
                    feats[lvl],
                    rois_dist[lvl],
                    self.resolution,
                    spatial_scale[lvl],
                    sampling_ratio=self.sampling_ratio,
                    rois_num=rois_num_dist[lvl],
                    aligned=self.aligned)
                if roi_feat.shape[0] > 0:
                    rois_feat_list.append(roi_feat)
            rois_feat_shuffle = paddle.concat(rois_feat_list)
            rois_feat = paddle.gather(rois_feat_shuffle, restore_index)

        return rois_feat
