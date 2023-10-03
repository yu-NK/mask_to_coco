# config/coco_config.py

import cv2
import os
import glob
import datetime as dt
import collections as cl

def info():
    tmp = cl.OrderedDict()
    tmp["description"] = "Flower CT Image Dataset"
    tmp["url"] = None
    tmp["version"] = None
    tmp["year"] = str(dt.datetime.now(dt.timezone.utc).year)
    tmp["contributor"] = "imp-plant"
    tmp["data_created"] = "2020/03/19"
    return tmp

def licenses():
    tmp = cl.OrderedDict()
    tmp["id"] = 1
    tmp["url"] = None
    tmp["name"] = None
    return tmp

def images(mask_path):
    tmps = []
    
    files = glob.glob(mask_path + "/*.png")
    files.sort()

    for i, file in enumerate(files):
        img = cv2.imread(file, 0)
        height, width = img.shape[:3]

        tmp = cl.OrderedDict()
        tmp["license"] = 1
        tmp["id"] = i + 1
        tmp["file_name"] = os.path.basename(file)
        tmp["width"] = width
        tmp["height"] = height
        tmp["date_captured"] = None
        tmp["coco_url"] = None
        tmp["flickr_url"] = None
        tmps.append(tmp)
    return tmps

def categories():
    tmps = []
    sup = ["petal"]
    cat = ["petal"]
    for i in range(len(sup)):
        tmp = cl.OrderedDict()
        tmp["id"] = i+1
        tmp["name"] = cat[i]
        tmp["supercategory"] = sup[i]
        tmps.append(tmp)
    return tmps