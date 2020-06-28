export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fast_eager_deletion_mode=1
export FLAGS_fraction_of_gpu_memory_to_use=1
export PYTHONPATH=$PYTHONPATH:.PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#GLOG_vmodule=operator=4
nohup python3 tools/train.py -c configs/yolov3_darknet.yml --use_gpu > output/yolov3_train.log 2>&1 &
#nohup python3 -m paddle.distributed.launch --selected_gpus=0,1 tools/train.py -c configs/faster_rcnn_r50_1x.yml --use_gpu > output/faster_rcnn_train.log 2>&1 &
#tail -f output/faster_rcnn_train.log 
