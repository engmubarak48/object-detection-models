import numpy as np
import pandas as pd
import requests
import random
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import urllib
import cv2
import os
from dicttoxml import dicttoxml
import glob
import shutil

# path = '/home/novelty/CPP/Detection_annotated/Indian_plates'
# number_plates_json = pd.read_json(os.path.join(path, 'Indian_Number_plates.json'), lines=True)

def saving_images(data):
    for index, row in data.iterrows():
        resp = urllib.request.urlopen(row[0])
        im = Image.open(resp)
        try:
            im.save(f'{path}/{index}.jpg')
        except:
            im = im.convert('RGB')
            im.save(f'{path}/{index}.jpg')

def delete_images(number_plates_json):
    for index, row in number_plates_json.iterrows():
        os.remove(f'{path}/{index}.jpg')

# delete_images(number_plates_json)
# print('FINISHED DELETING')
# downloading car images and creating annotation xml files
## downloaded images will be stored under 'number_plate/images' dir
## and the corresponding annotation xml files will be stored under 'number_plate/annotations'

# this method additionally creates a dataset to view the image dimesions later
## feel free to remove the dataset creation code in this method if not needed

def get_annotation_xml(filename, top_x=0, top_y=0, bottom_x=1, bottom_y=1, width=224, height=224):
    annotation = {
        "folder": "images",
        "filename": filename,
        "path": "/content/gdrive/My Drive/alpr/images/" + filename,
        "source": { "database": "Unknown" },
        "size": { "width": width, "height": height, "depth": 3 },
        "segmented": 0,
        "object": {
            "name": "number_plate",
            "pose": "Unspecified",
            "truncated": 0,
            "difficult": 0,
            "bndbox": {
                "xmin": int(top_x * width),
                "ymin": int(top_y * height),
                "xmax": int(bottom_x * width),
                "ymax": int(bottom_y * height)
            }
        }
    }
    
    # xml as bytes
    xml = dicttoxml(annotation, custom_root='annotation', attr_type=False)
    
    # xml as string
    xml_string = xml.decode("utf-8")
    
    # remove first line
    xml_string = xml_string.replace('<?xml version="1.0" encoding="UTF-8" ?>', '')
    
    return xml_string

def prepare_data():
    # os.system('mkdir number_plate')
    
    images_dir = 'number_plate/images/'
    # os.system('mikdir number_plate/images/')
    
    annotations_dir = 'number_plate/annotations/'
    # # !mkdir $annotations_dir
    # os.system('mikdir number_plate/annotations/')
    
    # dataset = {
    #     'img_file': [], 'img_width': [], 'img_height': [],
    #     'top_x': [], 'top_y': [], 'bottom_x': [], 'bottom_y': []
    # }
    
    for index, row in number_plates_json.iterrows():
        print (f'making files for row #{index+1}..')
        
        file_name = f'car_{index + 1}'
        img_file = f'{file_name}.jpg'
        txt_file = f'{file_name}.txt'
        
        img_url = row['content']
        img = urllib.request.urlopen(img_url)
        img = Image.open(img).convert('RGB')
        img.save(f'{images_dir}{img_file}', 'JPEG')
        
        annotation = row['annotation'][0]
        img_width = annotation['imageWidth']
        img_height = annotation['imageHeight']
        top_x = annotation['points'][0]['x']
        top_y = annotation['points'][0]['y']
        bottom_x = annotation['points'][1]['x']
        bottom_y = annotation['points'][1]['y']
        
        # dataset['img_file'].append(img_file)
        # dataset['img_width'].append(img_width)
        # dataset['img_height'].append(img_height)
        # dataset['top_x'].append(top_x)
        # dataset['top_y'].append(top_y)
        # dataset['bottom_x'].append(bottom_x)
        # dataset['bottom_y'].append(bottom_y)
        
        annotation_xml = get_annotation_xml(img_file, top_x, top_y, bottom_x, bottom_y, img_width, img_height)
        annotation_file = open(f'{annotations_dir}{file_name}.xml','w+')
        annotation_file.write(annotation_xml)
        annotation_file.close()
        
    # print(f'Number of car images downloaded - {len(dataset["img_file"])}\n')
    # print ('\n')
    # ! ls -ltrh $images_dir | grep .jpg | head -10
    
    # print ('\n')
    # ! ls -ltrh $annotations_dir | grep .xml | head -10
    
    # return pd.DataFrame(dataset)

# prepare_data()


def convert_data_to_darknet_format(images_path, images_saving_path, labels_saving_path, image_labels_text):

    image_labels = open(image_labels_text, 'r')
    image_labels = image_labels.readlines() 
    images = {f'{image_name}': image_name for image_name in os.listdir(images_path)}
    data_size = len(images)
    train_size = int(0.8 * data_size)
    category_idx = 0
    
    for i, annotation in zip(range(data_size), image_labels):
        if i!=0:
            _, image_name, x, y, bbox_width, bbox_height = annotation.split(',')
            x, y, bbox_width, bbox_height = int(x), int(y), int(bbox_width), int(bbox_height)
            image_path = images[image_name]
            image = cv2.imread(os.path.join(images_path, image_path))
            if (type(image) is np.ndarray):
                image_height, image_width = image.shape[0:2]
            else:
                print("Unable to load image at path {}".format(os.path.join(images_path, image_path)))
                continue
            # print('image_name: ', image_name, 'image_path:', image_path)
            name = os.path.splitext(image_name)[0]
            label_name = f"{name}.txt"
            x_center, y_center = x + bbox_width/2, y + bbox_height/2
            x_center, bbox_width = x_center/image_width, bbox_width/image_width
            y_center, bbox_height= y_center/image_height, bbox_height/image_height

            if i <= train_size:
                partition = 'train'
                label_file = open(f'{labels_saving_path}/{partition}/{label_name}', "a+")
                # cv2.imwrite(f'{images_saving_path}/{partition}/{image_name}', image)
                shutil.move(f'{images_path}/{image_name}', f'{images_saving_path}/{partition}/{image_name}')
                label_file.write(f"{category_idx} {x_center} {y_center} {bbox_width} {bbox_height} \n")
                label_file.close()
            else:
                partition = 'val'
                label_file = open(f'{labels_saving_path}/{partition}/{label_name}', "a+")
                # cv2.imwrite(f'{images_saving_path}/{partition}/{image_name}', image)
                shutil.move(f'{images_path}/{image_name}', f'{images_saving_path}/{partition}/{image_name}')
                label_file.write(f"{category_idx} {x_center} {y_center} {bbox_width} {bbox_height} \n")
                label_file.close()
        
            print('image saved as: ', f'{images_saving_path}/{partition}/{image_name}--{i}')
            print('label saved as: ', f'{labels_saving_path}/{partition}/{label_name}--{i}')

        # if i == 3:
        #     break
    # return 

images_path = '/home/novelty/CPP/annotated'
images_saving_path = '/home/novelty/CPP/big_data/images'
labels_saving_path = '/home/novelty/CPP/big_data/labels'
image_labels_text = '/home/novelty/CPP/report.txt'

convert_data_to_darknet_format(images_path, images_saving_path, labels_saving_path, image_labels_text)

print('DONE')

# def delete_with_2_annot():
#     for partition in os.listdir(data):
#         label_annot = open()
