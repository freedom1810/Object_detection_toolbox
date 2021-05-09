import json
import numpy 
import cv2


def count_bbox(data):
    counter = 0
    for image_id in data:
        counter += len(data[image_id].annotations)
    
    print('Data have {} bounding box'.format(counter))
    return counter

