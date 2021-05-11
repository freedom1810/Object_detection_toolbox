import json
import numpy as np
import cv2

from od_toolbox.read_anno_coco import read_json, data2json
from od_toolbox.split_image import create_split_image

if __name__ == "__main__":
    #read json original and split image and annotations
    train_json_dir = './example/source/train_traffic_sign_dataset.json'

    with open(train_json_dir, 'r') as train_dir:
        data_json = json.load(train_dir)
        
    data = read_json(data_json)

    data_split = create_split_image(data[3],
                        stride = (222, 114), #(w,h)
                        folder_image_dir = './example/source/',
                        folder_image_split_dir = './example/source/',
                        save_image = True,
                        small_image_h = 512, small_image_w = 512,)
                        
    data_split = data2json(data_split)
    train_json_dir_split_image = './example/source/train_traffic_sign_dataset_split_image.json'

    with open(train_json_dir_split_image, 'w') as train_dir:
        json.dump(data_split, train_dir)

    

    #read json split and draw bounding box
    train_json_dir = './example/source/train_traffic_sign_dataset_split_image.json'
    
    with open(train_json_dir, 'r') as train_dir:
        data_json = json.load(train_dir)

    data = read_json(data_json)

    image_after_draw = data['3_3_0'].draw_bbox('./example/source/')
    folder_test_dir = './example/test/'
    cv2.imwrite('{}/{}'.format(folder_test_dir, '3_3_0_draw_bbox.jpg'), image_after_draw)


