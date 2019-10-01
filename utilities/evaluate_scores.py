import argparse
import sys

sys.path.append('/home/lyan/Documents/tf-pose-estimation/tf_pose/')
sys.path.append('/home/lyan/Documents/tf-pose-estimation/')

import json

import numpy as np
import cv2

from tf_pose.estimator import TfPoseEstimator

parser = argparse.ArgumentParser(description='inference speed tester')

parser.add_argument('--graph',
                    default=None,
                    type=str, required=True)
parser.add_argument('--img-size', type=int, default=224, required=False)
parser.add_argument('--video', type=str,
                    default="/var/ssd_1t/ptai/data/videos/squat/good/IMG_6110.MOV.mp4",
                    required=False)
args = parser.parse_args()

model = TfPoseEstimator(args.graph, (args.img_size, args.img_size))

# fixme, add other videos
with open('alpha_landmarks/AlphaPose_video_2019-08-23_14-22-33.json') as f:
    ground_truth_keypoints = json.load(f)

cap = cv2.VideoCapture('video_2019-08-23_14-22-33.mp4')

sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
vars = (sigmas * 2) ** 2


def oks_score(gt, pred, image_shape):
    '''
    more about object keypoint similarity -
    http://presentations.cocodataset.org/COCO17-Keypoints-Overview.pdf
    :param gt:
    :param pred:
    :param image_shape
    :return:
    '''
    result = 0
    for i in range(18):
        dx = pred[i * 3] / image_shape[1] - gt[i * 3] / image_shape[1]
        dy = pred[i * 3 + 1] / image_shape[0] - gt[i * 3 + 1] / image_shape[0]

        e = (dx ** 2 + dy ** 2) / vars / 1 / 2
        if gt[i * 3] > 0:
            result += np.exp(-e.mean())
    return result / 18.0


def get_keypoints(humans, img_shape):
    '''
    transforms humans detection from tf-pose to list of keypoints
    takes by default only 0th human
    :param humans: list of human poses detected
    :param img_shape: tuple of size 3 - dimensions
    :return:
    '''
    assert len(humans) == 18, 'Should be 18 keypoints in detection'

    if humans is None or len(humans) == 0:
        return [0] * 18 * 3  # 0 is x, 1 is y, 2 is visibility

    kp_det = []

    for i in range(18):
        if i in humans[0].body_parts:
            kp_det.append(humans[0].body_parts[i].x * img_shape[1])
            kp_det.append(humans[0].body_parts[i].y * img_shape[0])
            kp_det.append(0)
        else:
            kp_det.append(0)
            kp_det.append(0)
            kp_det.append(0)

    return kp_det


def detect_poses(video_capture, det_model):
    result = []
    img_shape = None
    while True:
        ret, image = video_capture.read()
        if img_shape is None:
            img_shape = image.shape
        if image is None:
            break
        humans, _, _ = det_model.inference(image)
        result.append(humans)
    return result, img_shape


detected_poses, img_shape = detect_poses(cap, model)
n = len(detected_poses)

ground_truth_keypoints = {int(k.split('.')[0]): ground_truth_keypoints[k] for k in ground_truth_keypoints}

metrics = []
for k in ground_truth_keypoints:
    kp2 = ground_truth_keypoints[k]['bodies'][0]['joints']
    m = oks_score(kp2, get_keypoints(detected_poses[k], img_shape), img_shape)

    metrics.append(m)


def find_non_empty_detections(result):
    ids = []
    for i in range(len(result)):
        if len(result[i]) > 0:
            ids.append(i)

    return set(ids)


non_empty_ids = find_non_empty_detections(detected_poses)
print('ground truth labels - outputs of alpha pose')
print('detection intersection', len(non_empty_ids & ground_truth_keypoints.keys()) / float(n))
print('mean object keypoints similarity (oks)', np.array(metrics).mean())
