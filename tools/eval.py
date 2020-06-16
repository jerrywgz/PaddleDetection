from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
# ignore numba warning
import warnings
warnings.filterwarnings('ignore')
import random
import numpy as np
import paddle.fluid as fluid
from ppdet.core.workspace import load_config, merge_config, create
from ppdet.data.reader import create_reader
from ppdet.utils.check import check_gpu, check_version, check_config
from ppdet.utils.cli import ArgsParser
from ppdet.utils.eval_utils import coco_eval_results 


def main(FLAGS):
    cfg = load_config(FLAGS.config)
    merge_config(FLAGS.opt)
    check_config(cfg)
    # check if set use_gpu=True in paddlepaddle cpu version
    check_gpu(cfg.use_gpu)
    # check if paddlepaddle version is satisfied
    check_version()

    if FLAGS.use_gpu:
        devices_num = 1
    else:
        devices_num = int(os.environ.get('CPU_NUM', 1))

    # Model
    main_arch = cfg.architecture
    model = create(cfg.architecture)

    # Load weights 
    param_state_dict, opti_state_dict = fluid.load_dygraph(cfg.weights)
    model.set_dict(param_state_dict)

    # Reader 
    eval_reader = create_reader(cfg.EvalReader, devices_num=devices_num)

    # Eval
    bbox_res = []
    mask_res = []
    for iter_id, data in enumerate(eval_reader()):
        start_time = time.time()

        # forward 
        outs = model(data, mode='eval')

        # call eval 
        bbox_res += get_det_res(1, outs['bbox_nums'], outs['bbox'], data)
        if outs['mask'] is not None:
            mask_res += get_seg_res(1, outs['bbox_nums'], outs['bbox'], outs['mask'], data)

        # log 
        cost_time = time.time() - start_time
        print("Eval iter: {}, time: {}".format(iter_id, cost_time))
    
    coco_eval_results(bbox_res, mask_res)

if __name__ == '__main__':
    parser = ArgsParser()
    parser.add_argument(
        "--output_eval",
        default=None,
        type=str,
        help="Evaluation directory, default is current directory.")

    parser.add_argument(
        '--json_eval', action='store_true', default=False, help='')

    parser.add_argument(
        '--use_gpu', action='store_true', default=False, help='')

    FLAGS = parser.parse_args()
    place = fluid.CUDAPlace(fluid.dygraph.parallel.Env()
                            .dev_id) if FLAGS.use_gpu else fluid.CPUPlace()
    with fluid.dygraph.guard(place):
        main(FLAGS)
