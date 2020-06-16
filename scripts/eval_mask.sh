export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_fraction_of_gpu_memory_to_use=1
export PYTHONPATH=$PYTHONPATH:.PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
nohup python3 tools/eval.py -c configs/mask_rcnn_r50_1x.yml --use_gpu > output/mask_rcnn_eval.log 2>&1 &
tail -f output/mask_rcnn_eval.log 
