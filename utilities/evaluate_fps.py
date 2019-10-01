import sys

sys.path.append('/home/lyan/Documents/tf-pose-estimation/tf_pose/')
sys.path.append('/home/lyan/Documents/tf-pose-estimation/')

import cv2
import time
from tqdm import *
from tf_pose.estimator import TfPoseEstimator
import argparse


parser = argparse.ArgumentParser(description='inference speed tester')

parser.add_argument('--graph',
                    default=None,
                    type=str, required=True)
parser.add_argument('--img-size', type=int, default=224, required=False)
parser.add_argument('--video', type=str,
                    default="/var/ssd_1t/ptai/data/videos/squat/good/IMG_6110.MOV.mp4",
                    required=False)
args = parser.parse_args()

model = TfPoseEstimator(args.graph, (224, 224))
cap = cv2.VideoCapture(args.video)

ret, image = cap.read()
images = []
while ret:
    ret, image = cap.read()
    if image is not None:
        images.append(image)

start = time.time()
for im in tqdm(images):
    if im is not None:
        model.inference(im)
total_time = time.time() - start
print('total time', total_time, 'total image', len(images), 'avg fps', len(images)*1.0/total_time)