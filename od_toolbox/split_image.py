import cv2
import math
from od_toolbox.read_anno_coco import ImageAnno
from od_toolbox.utils import *

def create_new_bbox(split_bbox, bbox):
    xmin = bbox[0]
    xmax = bbox[0] + bbox[2]
    ymin = bbox[1]
    ymax = bbox[1] + bbox[3]

    if xmin < split_bbox[0]:
        xmin = split_bbox[0]
    if xmax > split_bbox[2]:
        xmax = split_bbox[2]
    if ymin < split_bbox[1]:
        ymin = split_bbox[1]
    if ymax > split_bbox[3]:
        ymax = split_bbox[3]
    
    return [xmin, ymin, xmax - xmin, ymax - ymin]

def create_split_image(image_anno,
                       stride = (222, 114), #(w,h)
                       folder_image_dir = 'za_traffic_2020/traffic_train/images_split',
                       folder_image_split_dir = '',
                       save_image = False,
                       small_image_h = 512, small_image_w = 512,
                      ):
    image_split_annos = {}

    if save_image:
        original_image = cv2.imread('{}/{}'.format(folder_image_dir, image_anno.file_name))
    
    image_ori_width = image_anno.width
    image_ori_height = image_anno.height
    for i in range(math.ceil((image_ori_width-small_image_w)/stride[0]) + 1):
        for j in range(math.ceil((image_ori_height-small_image_h)/stride[1]) + 1):

            split_img_id = '{}_{}_{}'.format(image_anno.id, i, j)                    
            split_img_file_name = '{}.jpg'.format(split_img_id)

            if i*stride[0] + small_image_w < image_ori_width:
                xmin = i*stride[0]
                xmax = i*stride[0] + small_image_w
            else:
                xmin = image_ori_width - small_image_w
                xmax = image_ori_width
            if j*stride[1]  + small_image_h< image_ori_height:
                ymin = j*stride[1]
                ymax = j*stride[1] + small_image_h
            else:
                ymin = image_ori_height - small_image_h
                ymax = image_ori_height

            split_bbox = [xmin, ymin, xmax, ymax]

            if save_image:
                split_image = original_image[split_bbox[1]:split_bbox[3], split_bbox[0]:split_bbox[2]]
            
            split_image_anno = ImageAnno(split_img_file_name, small_image_h, small_image_w, split_img_id)
            bboxes = []
            for anno_data in image_anno.annotations:
                if bbox_iou(split_bbox, xywh2xyxy(anno_data['bbox'])) > 0:
                    bbox = create_new_bbox(split_bbox, anno_data['bbox'])
                    
                    #x,y in original image to split image
                    bbox[0] -= split_bbox[0]
                    bbox[1] -= split_bbox[1]
                    
                    category_id = anno_data["category_id"]
                    
                    split_image_anno_info = {'segmentation': [],
                                                 'area': bbox[2]* bbox[3],
                                                 'iscrowd': 0,
                                                 'image_id': split_img_id,
                                                 'bbox': bbox,
                                                 'category_id': category_id,
                                                 'id': 0}
                    split_image_anno.update_anno(split_image_anno_info)

            image_split_annos[split_img_id] = split_image_anno
            
            if save_image:
                cv2.imwrite('{}/{}'.format(folder_image_split_dir, split_img_file_name), split_image)

    return image_split_annos