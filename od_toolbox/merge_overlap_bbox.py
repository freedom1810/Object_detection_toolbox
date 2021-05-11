import json
import numpy 
import cv2

from od_toolbox.read_anno_coco import read_json, data2json
from od_toolbox.eda import count_bbox
from od_toolbox.utils import bbox_iou, xywh2xyxy, merge_bbox

def merge_overlap_bbox_(list_bbox, iou_threshold=0.45):
    '''
    Merge overlap bounding box with iou > threshold

    Input: 
    list_annotations: list annotation coco format 
    iou_threshold: int

    Output: 
    list_annotations: list annotation coco format after merge
    '''
    black_list = []
 
    for i, anno in enumerate(list_bbox):
        if i in black_list: continue

        bbox = anno['bbox']
        class_id = anno['category_id']

        weight_1 = 1
        weight_2 = 1
        
        for j in range(i+1, len(list_bbox)):
            bbox2 = list_bbox[j]['bbox']
            class_id_2 = list_bbox[j]['category_id']
            if bbox_iou(xywh2xyxy(bbox), xywh2xyxy(bbox2)) > iou_threshold and class_id == class_id_2:

                black_list.append(j)
                
                bbox = merge_bbox(bbox, bbox2, weight_1, weight_2)
                weight_1 += 1
        
        list_bbox[i] = anno
                
    black_list = sorted(list(set(black_list)))
    # print('black_list {}'.format(black_list))
    for i in black_list[::-1]:
        list_bbox.pop(i)
        
    return list_bbox


def merge_overlap_bbox(data, iou_threshold = 0.45):

    '''
    Merge overlap bounding box with iou > threshold

    Input: 
    data: dict{image_id: ImageAnnotation}
    iou_threshold: int

    Output: 
    data: dict{image_id: ImageAnnotation} after merge bounding box
    '''

    print("Merge overlap bbox with same class_id, iou_threshold = {}".format(iou_threshold))
    print('Before merge')
    count_bbox(data)

    for image_id in data:
        data[image_id].annotations = merge_overlap_bbox_(data[image_id].annotations, iou_threshold = iou_threshold)

    print('After merge')
    count_bbox(data)
    return data


if __name__ == "__main__":
    train_json_dir = '/media/sonnh/ssd/data/zalo_2020/traffic_train/train_traffic_sign_dataset.json'
    
    with open(train_json_dir, 'r') as train_dir:
        data_json = json.load(train_dir)
        
    data = read_json(data_json)

    data = merge_overlap_bbox(data, iou_threshold = 0.45)

    data_json_merge_overlap_bbox = data2json(data)
    train_json_dir_merge_overlap_bbox = '/media/sonnh/ssd/data/zalo_2020/traffic_train/train_traffic_sign_dataset_merge_overlap_bbox.json'

    with open(train_json_dir_merge_overlap_bbox, 'w') as train_dir:
        json.dump(data_json_merge_overlap_bbox, train_dir)

