import json
import numpy 
import cv2

from read_anno_coco import read_json
from eda import count_bbox
from utils import bbox_iou, xywh2xyxy, merge_bbox

def merge_overlap_bbox_(list_bbox, iou_threshold=0.45):
    black_list = []
 
    for i, bbox_class_id in enumerate(list_bbox):
        if i in black_list: continue
            
        bbox, class_id = bbox_class_id
        weight_1 = 1
        weight_2 = 1
        
        for j in range(i+1, len(list_bbox)):
            bbox2, class_id_2 = list_bbox[j]
            if bbox_iou(xywh2xyxy(bbox), xywh2xyxy(bbox2)) > iou_threshold and class_id == class_id_2:

                black_list.append(j)
                
                bbox = merge_bbox(bbox, bbox2, weight_1, weight_2)
                weight_1 += 1
        
        list_bbox[i] = [bbox, class_id]
                
    black_list = sorted(list(set(black_list)))
    # print('black_list {}'.format(black_list))
    for i in black_list[::-1]:
        list_bbox.pop(i)
        
    return list_bbox


def merge_overlap_bbox(data, iou_threshold = 0.45):
    print("Merge overlap bbox with same class_id, iou_threshold = {}".format(iou_threshold))
    for image_id in data:
        data[image_id].annotations = merge_overlap_bbox_(data[image_id].annotations, iou_threshold = iou_threshold)
    return data


if __name__ == "__main__":
    train_json_dir = '/media/sonnh/ssd/data/zalo_2020/traffic_train/train_traffic_sign_dataset.json'
    data = read_json(train_json_dir)
    print('Before merge')
    count_bbox(data)

    data = merge_overlap_bbox(data, iou_threshold = 0.45)

    print('After merge')
    count_bbox(data)

