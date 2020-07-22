import paddle.fluid as fluid
from paddle.fluid.dygraph import Layer, Sequential

from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Normal, MSRA
from paddle.fluid.regularizer import L2Decay
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, Conv2DTranspose
from ppdet.core.workspace import register
from ..ops import RoIExtractor
from ..backbone.resnet import Blocks


@register
class MaskFeat(Layer):
    __inject__ = ['mask_roi_extractor']

    def __init__(self,
                 mask_roi_extractor,
                 num_convs=0,
                 feat_in=2048,
                 feat_out=256,
                 mask_num_stages=1,
                 share_bbox_feat=False):
        super(MaskFeat, self).__init__()
        self.num_convs = num_convs
        self.feat_in = feat_in
        self.feat_out = feat_out
        self.mask_roi_extractor = mask_roi_extractor
        self.mask_num_stages = mask_num_stages
        self.share_bbox_feat = share_bbox_feat
        self.upsample_module = []
        for i in range(self.mask_num_stages):
            name = 'stage_{}'.format(i)
            mask_conv = Sequential()
            for j in range(self.num_convs):
                conv_name = 'mask_inter_feat_{}'.format(j)
                conv = mask_conv.add_sublayer(conv_name, Conv2D())
            deconv_name = 'conv5_mask'
            mask_conv.add_sublayer(
                deconv_name,
                Conv2DTranspose(
                    num_channels=self.feat_in,
                    num_filters=self.feat_out,
                    filter_size=2,
                    stride=2,
                    act='relu',
                    param_attr=ParamAttr(initializer=MSRA(uniform=False)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.))))
            upsample = self.add_sublayer(name, mask_conv)
            self.upsample_module.append(upsample)

    def forward(self,
                body_feats,
                bboxes,
                bbox_feat,
                mask_index,
                spatial_scale,
                stage=0):
        if self.share_bbox_feat:
            rois_feat = fluid.layers.gather(bbox_feat, mask_index)
        else:
            bbox, bbox_num = bboxes
            rois_feat = self.mask_roi_extractor(body_feats, bbox, bbox_num,
                                                spatial_scale)
        # upsample 
        mask_feat = self.upsample_module[stage](rois_feat)
        return mask_feat


@register
class MaskHead(Layer):
    __shared__ = ['num_classes', 'num_stages']
    __inject__ = ['mask_feat']

    def __init__(self, mask_feat, feat_in=256, num_classes=81, num_stages=1):
        super(MaskHead, self).__init__()
        self.mask_feat = mask_feat
        self.feat_in = feat_in
        self.num_classes = num_classes
        self.num_stages = num_stages
        self.mask_num_stages = mask_num_stages
        self.mask_fcn_logits = []
        for i in range(self.num_stages):
            name = 'mask_fcn_logits_{}'.format(i)
            self.mask_fcn_logits.append(
                name,
                fluid.dygraph.Conv2D(
                    num_channels=self.feat_in,
                    num_filters=self.num_classes,
                    filter_size=1,
                    param_attr=ParamAttr(initializer=MSRA(uniform=False)),
                    bias_attr=ParamAttr(
                        learning_rate=2., regularizer=L2Decay(0.0))))

    def forward_train(self,
                      body_feats,
                      bboxes,
                      bbox_feat,
                      mask_index,
                      spatial_scale,
                      stage=0):
        # feat
        mask_feat = self.mask_feat(body_feats, bboxes, bbox_feat, mask_index,
                                   spatial_scale, stage)
        # logits
        mask_head_out = self.mask_fcn_logits[stage](mask_feat)
        return mask_head_out

    def forward_test(self,
                     im_info,
                     body_feats,
                     bboxes,
                     bbox_feat,
                     mask_index,
                     spatial_scale,
                     stage=0):
        bbox, bbox_num = bboxes
        if bbox.shape[0] == 0:
            mask_head_out = bbox
        else:
            im_info_expand = []
            for idx, num in enumerate(bbox_num):
                for n in range(num):
                    im_info_expand.append(im_info[idx, -1])
            im_info_expand = fluid.layers.concat(im_info_expand)
            rois = fluid.layers.elementwise_mul(
                bbox[:, 2:], im_info_expand, axis=0)
            mask_feat = self.mask_feat(body_feats, bboxes, bbox_feat,
                                       mask_index, spatial_scale, stage)
            mask_logit = self.mask_fcn_logits[stage](mask_feat)
            mask_head_out = fluid.layers.sigmoid(mask_logit)
        return mask_head_out

    def forward(self,
                inputs,
                body_feats,
                bboxes,
                bbox_feat,
                mask_index,
                spatial_scale,
                stage=0):
        if inputs['mode'] == 'train':
            mask_head_out = forward_train(body_feats, bboxes, bbox_feat,
                                          mask_index, spatial_scale, stage)
        else:
            im_info = inputs['im_info']
            mask_head_out = forward_test(im_info, body_feats, bboxes, bbox_feat,
                                         mask_index, spatial_scale, stage)
        return mask_head_out

    def loss(self, mask_head_out, mask_targets):
        reshape_dim = self.num_classes * self.mask_resolution * self.mask_resolution
        mask_logits = fluid.layers.reshape(mask_head_out, (-1, reshape_dim))
        mask_label = fluid.layers.cast(x=mask_targets, dtype='float32')

        loss_mask = fluid.layers.sigmoid_cross_entropy_with_logits(
            x=mask_logits, label=mask_label, ignore_index=-1, normalize=True)
        loss_mask = fluid.layers.reduce_sum(loss_mask, name='loss_mask')

        return {'loss_mask': loss_mask}
