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
from __future__ import unicode_literals

import numpy as np
from PIL import Image, ImageDraw
import cv2
from .colormap import colormap

__all__ = ['visualize_results']


def visualize_results(image,
                      im_id,
                      catid2name,
                      threshold=0.5,
                      bbox_results=None,
                      mask_results=None):
    """
    Visualize bbox and mask results
    """
    if mask_results:
        image = draw_mask(image, im_id, mask_results, threshold)
    if bbox_results:
        image = draw_bbox(image, im_id, catid2name, bbox_results, threshold)
    return image


def draw_mask(image, im_id, segms, threshold, alpha=0.7):
    """
    Draw mask on image
    """
    mask_color_id = 0
    w_ratio = .4
    color_list = colormap(rgb=True)
    img_array = np.array(image).astype('float32')
    for dt in np.array(segms):
        if im_id != dt['image_id']:
            continue
        segm, score = dt['segmentation'], dt['score']
        if score < threshold:
            continue
        import pycocotools.mask as mask_util
        mask = mask_util.decode(segm) * 255
        print('mask: ', mask)
        contour_mask = np.array(mask)/255
        contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        # mask_new, contours, hierarchy = cv2.findContours((mask).astype(np.uint8), cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
        #segmentation = []
        #for contour in contours:
        #    contour_list = contour.flatten().tolist()
        #    if len(contour_list) > 4:# and cv2.contourArea(contour)>10000
        #        segmentation.append(contour_list)
        #print('segmentation: ', segmentation)

        cv2.drawContours(img_array, contours, -1, (0, 255, 0), 1) 
        #idx = np.nonzero(mask)
        #img_array[idx[0], idx[1], :] *= 1.0 - alpha
        #img_array[idx[0], idx[1], :] += alpha * color_mask
    return Image.fromarray(img_array.astype('uint8'))


def draw_bbox(image, im_id, catid2name, bboxes, threshold):
    """
    Draw bbox on image
    """
    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        if im_id != dt['image_id']:
            continue
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        xmin, ymin, w, h = bbox
        xmax = xmin + w
        ymax = ymin + h

        if catid not in catid2color:
            idx = np.random.randint(len(color_list))
            catid2color[catid] = color_list[idx]
        color = tuple(catid2color[catid])

        # draw bbox
        draw.line(
            [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
             (xmin, ymin)],
            width=1,
            fill=(255,255,0))

        # draw label
        #text = "{} {:.2f}".format(catid2name[catid], score)
        #tw, th = draw.textsize(text)
        #draw.rectangle(
        #    [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        #draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    return image
