# config/directry.py

import os

DIR_BASE = '/workspace/dataset/'

DIR_INPUT = DIR_BASE + 'flower-ct-dataset/flower_ct/'
DIR_INPUT_JSON  = DIR_INPUT + 'annotations.json'
#DIR_INPUT_IMAGE = DIR_INPUT + 'JPEGImages/'

DIR_OUTPUT = DIR_BASE + 'crop_annotation/'
os.makedirs(DIR_OUTPUT, exist_ok=True)