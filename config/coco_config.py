# config/coco_config.py

import cv2
import os
import glob
import datetime as dt
import collections as cl

def licenses():
    tmp = cl.OrderedDict()
    tmp["name"] = ""
    tmp["id"] = 1
    tmp["url"] = ""
    return tmp

def info():
    tmp = cl.OrderedDict()
    tmp["contributor"] = ""
    tmp["data_created"] = ""
    tmp["description"] = ""
    tmp["url"] = ""
    tmp["version"] = ""
    tmp["year"] = ""
    return tmp

def categories():
    tmps = []
    sup = ["petal"]
    cat = ["petal"]
    for i in range(len(sup)):
        tmp = cl.OrderedDict()
        tmp["id"] = i+1
        tmp["name"] = cat[i]
        tmp["supercategory"] = ""#sup[i]
        tmps.append(tmp)
    return tmps

def images(mask_path):
    tmps = []
    
    files = glob.glob(os.path.join(mask_path, "*"))
    files.sort()

    for i, file in enumerate(files):
        img = cv2.imread(file, 0)
        height, width = img.shape[:3]

        tmp = cl.OrderedDict()
        tmp["id"] = i + 1
        tmp["width"] = width
        tmp["height"] = height
        tmp["file_name"] = os.path.basename(file)
        tmp["license"] = 1
        tmp["coco_url"] = ""
        tmp["flickr_url"] = ""
        tmp["date_captured"] = ""
        tmps.append(tmp)
    return tmps
