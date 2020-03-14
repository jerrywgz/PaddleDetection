# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

from collections import OrderedDict

from paddle import fluid

from ppdet.core.workspace import register
from ppdet.modeling.ops import MultiClassSoftNMS
from ppdet.modeling.architectures.cornernet_squeeze import * 
import numpy as np

__all__ = ['CenterNetSqueeze']

def rescale_center(ct_xs, ct_ys, ratios, borders, im):
    shape = fluid.layers.cast(fluid.layers.shape(im), 'float32')
    zero = fluid.layers.assign(np.array([0], dtype='float32'))
    ct_xs = ct_xs / ratios[:, 1] - borders[:, 2]
    ct_xs = fluid.layers.elementwise_max(ct_xs, zero)
    ct_xs = fluid.layers.elementwise_min(ct_xs, shape[3])

    ct_ys = ct_ys / ratios[:, 0] - borders[:, 0]
    ct_ys = fluid.layers.elementwise_max(ct_ys, zero)
    ct_ys = fluid.layers.elementwise_min(ct_ys, shape[2])
    return ct_xs, ct_ys

def rescale_bboxes(bboxes, ratios, borders, im, score):
    shape = fluid.layers.cast(fluid.layers.shape(im), 'float32')
    x1, y1, x2, y2 = fluid.layers.split(bboxes, 4)
    x1 = x1 / ratios[:, 1] - borders[:, 2]
    zero = fluid.layers.assign(np.array([0], dtype='float32'))
    x1 = fluid.layers.elementwise_max(x1, zero)
    x1 = fluid.layers.elementwise_min(x1, shape[3])
    x2 = x2 / ratios[:, 1] - borders[:, 2]
    x2 = fluid.layers.elementwise_max(x2, zero)
    x2 = fluid.layers.elementwise_min(x2, shape[3])

    y1 = y1 / ratios[:, 0] - borders[:, 0]
    y1 = fluid.layers.elementwise_max(y1, zero)
    y1 = fluid.layers.elementwise_min(y1, shape[2])
    y2 = y2 / ratios[:, 0] - borders[:, 0]
    y2 = fluid.layers.elementwise_max(y2, zero)
    y2 = fluid.layers.elementwise_min(y2, shape[2])

    tx_mask = fluid.layers.cast(x1 <= -5, 'float32')
    bx_mask = fluid.layers.cast(x2 >= shape[3] + 5, 'float32')
    ty_mask = fluid.layers.cast(y1 <= -5, 'float32')
    by_mask = fluid.layers.cast(y2 >= shape[2] + 5, 'float32')
    mask_list = [tx_mask, bx_mask, ty_mask, by_mask]
    for mask in mask_list:
        scores = score * (1 - mask) - mask
    return fluid.layers.concat([x1, y1, x2, y2], axis=2), scores


@register
class CenterNetSqueeze(CornerNetSqueeze):
    """
    """
    __category__ = 'architecture'
    __inject__ = ['backbone', 'center_head', 'nms', 'fpn']
    __shared__ = ['num_classes']

    def __init__(self,
                 backbone,
                 nms=MultiClassSoftNMS().__dict__,
                 center_head='CornerHead',
                 num_classes=80,
                 fpn=None):
        super(CenterNetSqueeze, self).__init__(backbone,
            nms=nms, corner_head=center_head, num_classes=num_classes, fpn=fpn)
        self.center_head = center_head


    def _inputs_def(self, image_shape, output_size, max_tag_len):
        im_shape = [None] + image_shape
        C = self.num_classes
        # yapf: disable
        inputs_def = {
            'image':        {'shape': im_shape,  'dtype': 'float32', 'lod_level': 0},
            'im_id':        {'shape': [None, 1], 'dtype': 'int64',   'lod_level': 0},
            'gt_bbox':       {'shape': [None, 4], 'dtype': 'float32', 'lod_level': 1},
            'gt_class':     {'shape': [None, 1], 'dtype': 'int32',   'lod_level': 1},
            'ratios':       {'shape': [None, 2],  'dtype': 'float32', 'lod_level': 0},
            'borders':      {'shape': [None, 4],  'dtype': 'float32', 'lod_level': 0},
            'tl_heatmaps':  {'shape': [None, C, output_size, output_size],  'dtype': 'float32', 'lod_level': 0},
            'br_heatmaps':  {'shape': [None, C, output_size, output_size],  'dtype': 'float32', 'lod_level': 0},
            'ct_heatmaps':  {'shape': [None, C, output_size, output_size],  'dtype': 'float32', 'lod_level': 0},
            'tl_regrs':     {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'br_regrs':     {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'ct_regrs':     {'shape': [None, max_tag_len, 2], 'dtype': 'float32', 'lod_level': 0},
            'tl_tags':      {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'br_tags':      {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'ct_tags':      {'shape': [None, max_tag_len], 'dtype': 'int64', 'lod_level': 0},
            'tag_masks':     {'shape': [None, max_tag_len], 'dtype': 'int32', 'lod_level': 0},
        }
        # yapf: enable
        return inputs_def

    def build(self, feed_vars, mode='train'):
        im = feed_vars['image']
        body_feats = self.backbone(im)
        if self.fpn is not None:
            body_feats, _ = self.fpn.get_output(body_feats)
            body_feats = [body_feats.values()[-1]]
        if mode == 'train':
            target_vars = [
              'tl_heatmaps', 'br_heatmaps', 'ct_heatmaps', 'tag_masks', 'tl_regrs',
              'br_regrs', 'ct_regrs', 'tl_tags', 'br_tags', 'ct_tags'
            ]
            target = {key: feed_vars[key] for key in target_vars}
            self.center_head.get_output(body_feats)
            loss = self.center_head.get_loss(target)
            return loss

        elif mode == 'test':
            ratios = feed_vars['ratios']
            borders = feed_vars['borders']
            bboxes, scores, tl_scores, br_scores, clses, ct_xs, ct_ys, ct_clses, ct_scores = self.center_head.get_prediction(
                body_feats[-1])
            bboxes, scores = rescale_bboxes(bboxes, ratios, borders, im, scores)
            ct_xs, ct_ys = rescale_center(ct_xs, ct_ys, ratios, borders, im)
            detections = fluid.layers.concat(
                [bboxes, scores, tl_scores, br_scores, clses], axis=2)
            scores = fluid.layers.squeeze(scores, axes=[0, 2])
            ct_xs = fluid.layers.squeeze(ct_xs, axes=[0])
            ct_ys = fluid.layers.squeeze(ct_ys, axes=[0])
            ct_clses = fluid.layers.squeeze(ct_clses, axes=[0])
            ct_scores = fluid.layers.squeeze(ct_scores, axes=[0])
            detections = detections[0]

            keep_inds = fluid.layers.squeeze(
                fluid.layers.where(scores > -1), axes=[-1])
            inds_shape = fluid.layers.shape(keep_inds)
            inds_size = fluid.layers.reduce_prod(inds_shape)
            cond = inds_size < 1 
            valid_res = fluid.layers.create_global_var(
                shape=[1,6],
                value=0.0,
                dtype='float32',
                persistable=False,
                name='valid_res')
            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(cond):
                    fluid.layers.assign(
                        input=np.zeros((1,8)).astype('float32')-1, output=valid_res)
                with switch.default():
                    valid_detections = fluid.layers.gather(detections, keep_inds)
                    fluid.layers.assign(input=valid_detections, output=valid_res)
            box_width = valid_res[:,2] - valid_res[:,0]
            box_height = valid_res[:,3] - valid_res[:,1]
            s_inds = fluid.layers.where(box_width*box_height <= 22500)
            l_inds = fluid.layers.where(box_width*box_height > 22500)

            inds_shape = fluid.layers.shape(s_inds)
            inds_size = fluid.layers.reduce_prod(inds_shape)
            cond = inds_size < 1
            s_detection = fluid.layers.create_global_var(
                shape=[1,8],
                value=0.0,
                dtype='float32',
                persistable=False,
                name='s_detection')
            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(cond):
                    fluid.layers.assign(
                        input=np.zeros((1,8)).astype('float32')-1, output=s_detection)
                with switch.default():
                    valid_detections = fluid.layers.gather(detections, s_inds)
                    fluid.layers.assign(input=valid_detections, output=s_detection)

            inds_shape = fluid.layers.shape(l_inds)
            inds_size = fluid.layers.reduce_prod(inds_shape)
            cond = inds_size < 1
            l_detection = fluid.layers.create_global_var(
                shape=[1,8],
                value=0.0,
                dtype='float32',
                persistable=False,
                name='l_detection')
            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(cond):
                    fluid.layers.assign(
                        input=np.zeros((1,8)).astype('float32')-1, output=l_detection)
                with switch.default():
                    valid_detections = fluid.layers.gather(detections, l_inds)
                    fluid.layers.assign(input=valid_detections, output=l_detection)

            s_left_x = (2*s_detection[:,0] + s_detection[:,2])/3
            s_right_x = (s_detection[:,0] + 2*s_detection[:,2])/3
            s_top_y = (2*s_detection[:,1] + s_detection[:,3])/3
            s_bottom_y = (s_detection[:,1]+2*s_detection[:,3])/3

            pts_num = fluid.layers.shape(fluid.layers.unsqueeze(ct_xs, 1))
            ind_lx = fluid.layers.cast((ct_xs - fluid.layers.expand(fluid.layers.unsqueeze(s_left_x, axes=0), pts_num)) > 0, 'int32')
            ind_rx = fluid.layers.cast((ct_xs - fluid.layers.expand(fluid.layers.unsqueeze(s_right_x, axes=0), pts_num)) < 0, 'int32')
            ind_ty = fluid.layers.cast((ct_ys - fluid.layers.expand(fluid.layers.unsqueeze(s_top_y, axes=0), pts_num)) > 0, 'int32')
            ind_by = fluid.layers.cast((ct_ys - fluid.layers.expand(fluid.layers.unsqueeze(s_bottom_y, axes=0), pts_num)) < 0 , 'int32')
            ind_cls = fluid.layers.cast((ct_clses - fluid.layers.expand(fluid.layers.unsqueeze(s_detection[:, -1], axes=0), pts_num)) == 0, 'int32')

            ind_ct = ind_lx+ind_rx+ind_ty+ind_by+ind_cls
            ind_s_new_score = fluid.layers.reduce_max(ind_ct, dim=0)==5
            ind_s_new = fluid.layers.where(ind_s_new_score)
            inds_shape = fluid.layers.shape(ind_s_new)
            inds_size = fluid.layers.reduce_prod(inds_shape)
            cond = inds_size < 1
            new_s_detection = fluid.layers.create_global_var(
                shape=[1,6],
                value=0.0,
                dtype='float32',
                persistable=False,
                name='new_s_detection')
            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(cond):
                    fluid.layers.assign(
                        input=np.zeros((1,6)).astype('float32')-1, output=new_s_detection)
                with switch.default():
                    ind_ct_T = fluid.layers.transpose(ind_ct, [1, 0])
                    ind_ct_new = fluid.layers.gather(ind_ct_T, ind_s_new)
                    ind_ct_new = fluid.layers.transpose(ind_ct_new, [1, 0])
                    index_ct_new = fluid.layers.argmax(ind_ct_new)
                    s_ct_scores = fluid.layers.gather(ct_scores, index_ct_new)

                    s_det = fluid.layers.gather(s_detection, ind_s_new)
                    s_det_scores = s_det[:, 4:5]
                    s_det_scores = (s_det_scores + s_ct_scores * 2) / 3
                  
                    valid_detections = fluid.layers.concat([s_det[:, :4], s_det_scores, s_det[:, -1:]], axis=1)
    
                    fluid.layers.assign(input=valid_detections, output=new_s_detection)


            l_left_x = (3*l_detection[:,0] + 2*l_detection[:,2])/5
            l_right_x = (2*l_detection[:,0] + 3*l_detection[:,2])/5
            l_top_y = (3*l_detection[:,1] + 2*l_detection[:,3])/5
            l_bottom_y = (2*l_detection[:,1] + 3*l_detection[:,3])/5

            ind_lx = fluid.layers.cast((ct_xs - fluid.layers.expand(fluid.layers.unsqueeze(l_left_x, axes=0), pts_num)) > 0, 'int32')
            ind_rx = fluid.layers.cast((ct_xs - fluid.layers.expand(fluid.layers.unsqueeze(l_right_x, axes=0), pts_num)) < 0, 'int32')
            ind_ty = fluid.layers.cast((ct_ys - fluid.layers.expand(fluid.layers.unsqueeze(l_top_y, axes=0), pts_num)) > 0, 'int32')
            ind_by = fluid.layers.cast((ct_ys - fluid.layers.expand(fluid.layers.unsqueeze(l_bottom_y, axes=0), pts_num)) < 0 , 'int32')
            ind_cls = fluid.layers.cast((ct_clses - fluid.layers.expand(fluid.layers.unsqueeze(l_detection[:, -1], axes=0), pts_num)) == 0, 'int32')

            ind_ct = ind_lx+ind_rx+ind_ty+ind_by+ind_cls
            ind_l_new_score = fluid.layers.reduce_max(ind_ct, dim=0)==5
            ind_l_new = fluid.layers.where(ind_l_new_score)
            inds_shape = fluid.layers.shape(ind_l_new)
            inds_size = fluid.layers.reduce_prod(inds_shape)
            cond = inds_size < 1
            new_l_detection = fluid.layers.create_global_var(
                shape=[1,6],
                value=0.0,
                dtype='float32',
                persistable=False,
                name='new_l_detection')
            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(cond):
                    fluid.layers.assign(
                        input=np.zeros((1,6)).astype('float32')-1, output=new_l_detection)
                with switch.default():
                    ind_ct_T = fluid.layers.transpose(ind_ct, [1, 0])
                    ind_ct_new = fluid.layers.gather(ind_ct_T, ind_l_new)
                    ind_ct_new = fluid.layers.transpose(ind_ct_new, [1, 0])
                    index_ct_new = fluid.layers.argmax(ind_ct_new)
                    l_ct_scores = fluid.layers.gather(ct_scores, index_ct_new)

                    l_det = fluid.layers.gather(s_detection, ind_l_new)
                    l_det_scores = l_det[:, 4:5]
                    l_det_scores = (l_det_scores + l_ct_scores * 2) / 3
                    valid_detections = fluid.layers.concat([l_det[:, :4], l_det_scores, l_det[:, -1:]], axis=1)
    
                    fluid.layers.assign(input=valid_detections, output=new_l_detection)

            new_detections = fluid.layers.concat([new_s_detection, new_l_detection], axis=0)
            scores = new_detections[:, 4]
            
            keep_inds = fluid.layers.squeeze(
                fluid.layers.where(scores > -1), axes=[-1])
            inds_shape = fluid.layers.shape(keep_inds)
            inds_size = fluid.layers.reduce_prod(inds_shape)
            cond = inds_size < 1
            total_res = fluid.layers.create_global_var(
                shape=[1,6],
                value=0.0,
                dtype='float32',
                persistable=False,
                name='total_res')
            with fluid.layers.control_flow.Switch() as switch:
                with switch.case(cond):
                    fluid.layers.assign(
                        input=np.array([]).astype('float32'), output=total_res)
                with switch.default():
                    new_detections = fluid.layers.gather(new_detections, keep_inds)
                    _, sort_ind = fluid.layers.argsort(new_detections[:, 4], descending=True)
                    new_detections = fluid.layers.gather(new_detections, sort_ind)
                    total_out = self.nms(new_detections[:, :4], new_detections[:, 4],
                                         new_detections[:, -1])
                    fluid.layers.assign(input=total_out, output=total_res)
            return {'bbox': total_res}
   


