import cv2
import json
from od_toolbox.read_anno_coco import read_json
from od_toolbox.merge_overlap_bbox import merge_overlap_bbox

train_json_dir = './example/source/train_traffic_sign_dataset.json'

with open(train_json_dir, 'r') as train_dir:
    data_json = json.load(train_dir)

data = read_json(data_json)
data = merge_overlap_bbox(data)
