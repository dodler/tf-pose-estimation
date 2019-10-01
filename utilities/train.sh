CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/home/lyan/Documents/tf-pose-estimation/ python tf_pose/train.py --lr=0.0001 --batchsize=64 --input-width=224 --input-height=224 --datapath=/var/ssd_1t/coco/annotations/ --imgpath=/var/ssd_1t/coco/ --model=mobilenetv3_large

imgpath - путь до папки с coco
модель по умолчанию - mobilenet v2
плюс надо включать энвайронмент со всеми зависимостями