export PYTHONPATH=$PYTHONPATH:.PYTHONPATH
export CUDA_VISIBLE_DEVICES=0
nohup python3 tools/eval.py -c configs/faster_rcnn_r50_1x.yml --use_gpu > output/faster_rcnn_eval.log 2>&1 &
tail -f output/faster_rcnn_eval.log 
