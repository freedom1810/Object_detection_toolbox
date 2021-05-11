import json
import numpy as np
import cv2

import os
from multiprocessing import Pool
from functools import partial

from od_toolbox.read_anno_coco import read_json, data2json
from od_toolbox.split_image import create_split_image

if __name__ == "__main__":
    #read json original and split image and annotations
    train_json_dir = '/home/hana/sonnh/data/zalo_2020/traffic_train/train_traffic_sign_dataset_merge_overlap_bbox.json'

    with open(train_json_dir, 'r') as train_dir:
        data_json = json.load(train_dir)
        
    data = read_json(data_json)

    pool = Pool()
    data_split = pool.map(partial(create_split_image, stride = (222, 114), #(w,h)
                                            folder_image_dir = '/home/hana/sonnh/data/zalo_2020/traffic_train/images/',
                                            folder_image_split_dir = '/home/hana/sonnh/data/zalo_2020/traffic_train/images_split/',
                                            save_image = True,
                                            small_image_h = 512, small_image_w = 512,), 
                            list(data.values()))
    
    data_splits = {}
    for image_annos in data_split:
        data_splits.update(image_annos)
                        
    data_splits = data2json(data_splits)
    train_json_dir_split_image = '/home/hana/sonnh/data/zalo_2020/traffic_train/train_traffic_sign_dataset_split_image.json'

    with open(train_json_dir_split_image, 'w') as train_dir:
        json.dump(data_splits, train_dir)

    
    #read json split and draw bounding box
    train_json_dir = '/home/hana/sonnh/data/zalo_2020/traffic_train/train_traffic_sign_dataset_split_image.json'

    with open(train_json_dir, 'r') as train_dir:
        data_json = json.load(train_dir)

    data = read_json(data_json)

    image_after_draw = data['3_3_0'].draw_bbox('/home/hana/sonnh/data/zalo_2020/traffic_train/images_split/')
    folder_test_dir = 'test/'
    cv2.imwrite('{}/{}'.format(folder_test_dir, '3_3_0_draw_bbox.jpg'), image_after_draw)



