# config/directry.py

import os

DIR_INPUT = '/workspace/dataset/flower-ct-dataset/flower_ct/'

DIR_JSON  = DIR_DATASET + 'annotations.json'
#DIR_IMAGE = DIR_DATASET + 'JPEGImages/'

DIR_OUTPUT = '/workspace/dataset/crop_annotation/'
os.makedirs(DIR_OUTPUT, exist_ok=True)