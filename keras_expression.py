# using FER set - facial expression recognition set

#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
from PIL import Image
# install pillow if u no have it

# Pixel values range from 0 to 255 (0 is normally black and 255 is white)
basedir = os.path.join('C:/Users/trivenikuchi/Desktop/Portfolio/Python/Sketches', 'data', 'raw')
file_origin = os.path.join(basedir, 'fer2013.csv')
# file = 'fer2013.csv'
data_raw = pd.read_csv(file_origin)
data_input = pd.DataFrame(data_raw, columns=['emotion', 'pixels', 'Usage'])

data_input.rename({'Usage': 'usage'}, inplace=True)
print(data_input.head())
label_map = {
    0: '0_Anger',
    1: '1_Disgust',
    2: '2_Fear',
    3: '3_Happy',
    4: '4_Neutral',
    5: '5_Sad',
    6: '6_Surprise'
}

# Creating the folders
output_folders = data_input['Usage'].unique().tolist()
all_folders = []

for folder in output_folders:
    for label in label_map:
        all_folders.append(os.path.join(basedir,folder,label_map[label]))
# all_folders.append(os.path.join(basedir, folder, label_map[label]))
# for folder in all_folders:
# if not os.path.exists(folder):
# os.makedirs(folder)
# else:
# print('Folder {} exists already'.format(folder))
# counter_error = 0
# counter_correct = 0
