import cv2
import json
from od_toolbox.read_anno_coco import read_json

train_json_dir = './example/source/train_traffic_sign_dataset.json'

with open(train_json_dir, 'r') as train_dir:
    data_json = json.load(train_dir)

data = read_json(data_json)

image_after_draw = data[3].draw_bbox('./example/source')
folder_test_dir = './example/test'
cv2.imwrite('{}/{}'.format(folder_test_dir, '3_draw_bbox.jpg'), image_after_draw)