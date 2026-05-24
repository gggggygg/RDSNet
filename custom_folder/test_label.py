import numpy as np
from custom_folder.visual import draw_rotated_boxes
from ultralytics.utils import ops
import torch
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
import json

label_segmention = []
label_rotated_box = []
ann_segmention = []
ann_rotated_box = []

with open('../images/label_test/0071_09030_segmention.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        x = np.array(line.strip().split()[1:], dtype=np.float32)
        y = [xi*1024 for xi in x]
        label_segmention.append(y)

with open('../images/label_test/0071_09030_rotated_box.txt', 'r') as f:
    lines = f.readlines()
    for line in lines:
        x = np.array(line.strip().split()[1:], dtype=np.float32)
        y = [xi*1024 for xi in x]
        label_rotated_box.append(y)

with open('../images/label_test/annotations.json', 'r') as f:
    data = json.load(f)
    for ann in data['annotations']:
        segmention = np.array(ann['segmentation'][0], dtype=np.float32)
        rotated_box = np.array(ann['rotated_box'], dtype=np.float32)

        ann_segmention.append(segmention)
        ann_rotated_box.append(rotated_box[:4])

label_segmention = np.array(label_segmention)

label_rotated_box = np.array(label_rotated_box)
label_rotated_box_xyxy = ops.loaf_xywh2xyxyxyxy(torch.from_numpy(label_rotated_box).cuda()).cpu().numpy()
label_rotated_box_xyxy = label_rotated_box_xyxy.reshape(-1,8).tolist()

draw_rotated_boxes('../images/label_test/0071_09030.jpg', label_segmention, label_rotated_box_xyxy, save_dir = '../images/label_test/out_put/', conf = None)

ann_segmention = np.array(ann_segmention)
ann_rotated_box = np.array(ann_rotated_box)
ann_rotated_box_xyxy = ops.loaf_xywh2xyxyxyxy(torch.from_numpy(ann_rotated_box).cuda()).cpu().numpy()
ann_rotated_box_xyxy = ann_rotated_box_xyxy.reshape(-1,8).tolist()

draw_rotated_boxes('../images/label_test/0071_09030.jpg', ann_segmention, ann_rotated_box_xyxy, save_dir = '../images/label_test/out_put_ann/', conf = None)

x=1