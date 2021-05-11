import json
import numpy 
import cv2

class ImageAnno():
    def __init__(self, file_name, height, width, image_id):
        self.file_name = file_name
        self.height = height
        self.width = width
        self.id = image_id
        self.annotations = []
    
    def update_anno(self, anno):
        '''
        anno format:{'segmentation': [],
                    'area': 342,
                    'iscrowd': 0,
                    'image_id': 3,
                    'bbox': [880, 333, 19, 18],
                    'category_id': 2,
                    'id': 0}
        '''
        self.annotations.append(anno)
        
    
    def draw_bbox(self, folder_image_dir):
        color = (255, 255, 255)
        thickness = 1
        
        fontScale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness_text =1
        color_text = (255, 0, 0)

        image = cv2.imread('{}/{}'.format(folder_image_dir, self.file_name))
        
        for anno in self.annotations:
            (xmin, ymin, w, h) = anno['bbox']
            class_id = anno['category_id']

            xmax = xmin + w
            ymax = ymin + h
            
            start_point = (xmin, ymin)
            end_point = (xmax, ymax)
            
            #draw bounding box
            image = cv2.rectangle(image, start_point, end_point, color, thickness)
            
            #draw class
            org = (xmin, ymin)
            image = cv2.putText(image, '{}'.format(class_id) , org, font, fontScale, color_text, thickness_text, cv2.LINE_AA)
            
        return image
            

def read_json(data_json):
    '''
    Convert coco json format to ImageAnno

    Input: json file 

    Output: dict{image_id: ImageAnno}
    '''

    data = {}

    for image_info in data_json['images']:
    #     {'file_name': '4289.png',
    #      'height': 626,
    #      'width': 1622,
    #      'id': 4289,
    #      'street_id': 36}
        image_id = image_info['id']
        data[image_id] = ImageAnno(image_info['file_name'],
                            image_info['height'],
                            image_info['width'],
                            image_info['id']
                            )

    for anno in data_json['annotations']:
    # {'segmentation': [],
    #  'area': 342,
    #  'iscrowd': 0,
    #  'image_id': 3,
    #  'bbox': [880, 333, 19, 18],
    #  'category_id': 2,
    #  'id': 0}
        image_id = anno['image_id']
        data[image_id].update_anno(anno)
    
    return data

def data2json(data):
    '''
    Convert ImageAnno to coco json format 

    Input: dict{image_id: ImageAnno}

    Output: json file 
    '''

    data_json = {'images': [],
                'annotations': []}
    
    anno_id = 0
    for image_id in data:
        # print(data[image_id].annotations)

        image_info = {'width' : data[image_id].width,
                        'height' : data[image_id].height,
                        'id' : data[image_id].id,
                        'file_name' : data[image_id].file_name,
                        }
        data_json['images'].append(image_info)

        for anno in data[image_id].annotations:

            data_json['annotations'].append(anno)

    return data_json

if __name__ == "__main__":
    train_json_dir = '/media/sonnh/ssd/data/zalo_2020/traffic_train/train_traffic_sign_dataset.json'

    with open(train_json_dir, 'r') as train_dir:
        data_json = json.load(train_dir)

    data = read_json(data_json)

    image_after_draw = data[3].draw_bbox('/media/sonnh/ssd/data/zalo_2020/traffic_train/images')
    folder_test_dir = '/media/sonnh/Object_detection_toolbox/test'
    cv2.imwrite('{}/{}'.format(folder_test_dir, '3_after_draw.jpg'), image_after_draw)