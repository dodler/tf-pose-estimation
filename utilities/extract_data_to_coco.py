import json
import os
import os.path as osp

import cv2
from PIL import Image
from tqdm import *

SCORE_THRESHOLD = 2.8

with open('/var/data/coco/annotations/person_keypoints_train2017.json') as f:
    data = json.load(f)

max_anno = max(
    [data['annotations'][k]['id'] for k in range(len(data['annotations'])) if 'id' in data['annotations'][k]])
max_img_id = max([data['images'][k]['id'] for k in range(len(data['images']))])

print('annotations total', len(data['annotations']),
      'images total', len(data['images']))

print('number of keypoints', len(data['annotations'][0]['keypoints']))

all_landmarks = os.listdir('/var/ssd_1t/ptai/alpha_pose_landmarks')
all_landmarks = [osp.join('/var/ssd_1t/ptai/alpha_pose_landmarks', k) for k in all_landmarks]

vid_path = '/var/ssd_1t/ptai/data/videos/squat2808/'
COCO_PATH = '/var/ssd_1t/coco/train2017/'
img_path = COCO_PATH

idx2path = {}
for vid in os.listdir(osp.join(vid_path, 'bad')):
    vid_idx = vid.split('.')[0]
    idx2path[vid_idx] = osp.join(vid_path, 'bad', vid)

for vid in os.listdir(osp.join(vid_path, 'good')):
    vid_idx = vid.split('.')[0]
    idx2path[vid_idx] = osp.join(vid_path, 'good', vid)

bad_ids = set([k.split('.')[0] for k in os.listdir(osp.join(vid_path, 'bad'))])
good_ids = set([k.split('.')[0] for k in os.listdir(osp.join(vid_path, 'good'))])


def get_vid(fp):
    vid = fp.split('/')[-1].replace('AlphaPose_', '')
    vid = vid[:vid.find('_ext')]
    cap = cv2.VideoCapture(osp.join('/var/ssd_1t/ptai/data/videos/squat2808/good/' + vid))
    _, frame = cap.read()
    if frame is None:
        cap = cv2.VideoCapture(osp.join('/var/ssd_1t/ptai/data/videos/squat2808/bad/' + vid))
        _, frame = cap.read()
        if frame is None:
            return None, None
        else:
            return osp.join('/var/ssd_1t/ptai/data/videos/squat2808/bad/' + vid), frame
    else:
        return osp.join('/var/ssd_1t/ptai/data/videos/squat2808/good/' + vid), frame


## sanity check
for i in range(len(all_landmarks)):
    vid_idx = all_landmarks[i].split('/')[-1].replace('.json', '').replace('AlphaPose_', '')
    if vid_idx not in idx2path:
        raise Exception('not in idx2path')

missing = []
cnt = max_anno + 1
img_cnt = max_img_id + 1
for j in range(len(all_landmarks)):

    assert all_landmarks[j].endswith('json')

    with open(all_landmarks[j]) as f:
        anno_alpha1 = json.load(f)

    vid_idx = all_landmarks[j].split('/')[-1].replace('.json', '').replace('AlphaPose_', '')

    height = -1
    width = -1
    for i in tqdm(range(len(anno_alpha1))):

        fr_name = 'AlphaPose_' + vid_idx + '_' + str(i) + '.jpg'

        t = anno_alpha1[i]

        img_t = {
            'file_name': fr_name,
            'id': img_cnt
        }

        t['image_id'] = img_cnt
        t['keypoints'] = [int(k) for k in t['keypoints']]
        for i in range(17):
            t['keypoints'][(i * 3) + 2] = 2
        t['num_keypoints'] = 17
        t['id'] = cnt

        img_cnt += 1
        cnt += 1

        if t['score'] > SCORE_THRESHOLD and osp.exists(COCO_PATH + img_t['file_name']):
            if height == -1 and width == -1:
                try:
                    img_ = Image.open(COCO_PATH + img_t['file_name'])
                except:
                    print(img_t['file_name'])
                    continue
                img_t['width'] = img_.width
                img_t['height'] = img_.height
                height = img_.height
                width = img_.width
            else:
                img_t['width'] = width
                img_t['height'] = height

            data['annotations'].append(t)
            data['images'].append(img_t)

print('annotations len', len(data['annotations']),
      'images len', len(data['images']))

for i in tqdm(range(len(data['images']))):
    if 'AlphaPose' in data['images'][i]['file_name']:
        fname = osp.join(COCO_PATH, data['images'][i]['file_name'])
        if osp.exists(fname):
            img = cv2.imread(fname)
            if img is None:
                print('removing', fname)
                os.remove(fname)

with open('ext_train.json', 'w') as f:
    json.dump(data, f)
print('dupmed to ext_train.json')
