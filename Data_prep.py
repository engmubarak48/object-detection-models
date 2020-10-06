#%%
import pandas as pd 
import os
import collections 

file_path = open('/home/novelty/CPP/report.txt', mode='r')
file_ = file_path.readlines()
dic = {'filename':[], 'width':[], 'height':[], 'class':[], 'xmin':[], 'ymin':[], 'xmax':[], 'ymax':[]}
column_names = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
class_name = 'car plate'
for i, line in enumerate(file_):
    if i==0:
        continue
    index, file_name, xmin, ymin, bbox_width, bbox_height = line.split(',')
    xmin, ymin, bbox_width, bbox_height = int(xmin), int(ymin), int(bbox_width), int(bbox_height)
    xmax, ymax = xmin+bbox_width, xmax+bbox_height
    info =  [file_name, bbox_width, bbox_height, class_name, xmin, ymin, xmax, ymax]
    for column_name, value in zip(column_names, info):
        dic[column_name]+=[value]

dataframe = pd.DataFrame(dic)
dataframe.head()

    
