export PYTHONPATH=$PYTHONPATH:.PYTHONPATH
export CUDA_VISIBLE_DEVICES=2
#export CUDA_VISIBLE_DEVICES=0,1,2,3
#GLOG_vmodule=operator=4
#nohup python3 tools/train.py -c configs/faster_rcnn_r50_1x.yml --use_gpu > output/faster_rcnn_train.log 2>&1 &
nohup python3 tools/train.py -c configs/mask_rcnn_r50_1x.yml --use_gpu > output/mask_rcnn_train.log 2>&1 &
#nohup python3 -m paddle.distributed.launch --selected_gpus=0,1 tools/train.py -c configs/faster_rcnn_r50_1x.yml --use_gpu > output/faster_rcnn_train.log 2>&1 &
#tail -f output/mask_rcnn_train.log 
