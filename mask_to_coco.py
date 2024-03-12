"""
File Name: mask_coco_coco.py
Description: 
    - This script is designed to create a COCO-format dataset from images and their
      corresponding mask files for instance segmentation tasks.
    - It generates a COCO format dataset (JSON file) from mask images annotated 
      for instance segmentation of a single class.
    - Editing info, licenses, categories required for COCO format in 
      ./config/coco_config.py is necessary.
    - The output segmentation format will be in polygons.

Author: Yuki Naka
Created Date: 2023-10-04
Last Modified: 2024-03-12

Usage:
    python mask_to_coco.py <dataset> --type [train|val|test] --name [output JSON file name]

    <dataset>:
        The base directory path of the dataset where the images and masks are stored.
    --type [train|val|test]: 
        Specifies the dataset type. It can be 'train', 'val', or 'test'. Default is 'train'.
    --name [output JSON file name]: 
        Specifies the name of the output JSON file for COCO annotations. 
        The default is auto-generated based on the type, following the pattern '[TYPE]_annotations.json'.

Dependencies:
    - Python 3.x, numpy, OpenCV (cv2), tqdm

    Additional configuration files:
    - config/coco_config.py: Contains settings for the COCO dataset.
    - utils/tools.py: Contains additional tools required for dataset creation.

License: MIT License
"""

# Standard Libraries
import json
import sys
import os
import glob
import argparse
import collections as cl

# Data Handling
import numpy as np

# Image Processing
import cv2

# Utility
from tqdm import tqdm

# Specified in config/coco_config.py
from config.coco_config import info, licenses, images, categories
from utils.tools import background_del, assign_cluster_number

def parse_args():
    
    parser = argparse.ArgumentParser(
        description="Create a COCO format dataset from mask images for instance segmentation "
                    "(single class). The output segmentation format will be in polygons."
    )
    
    # Specifying the directory that contains the images to be converted to COCO format.
    parser.add_argument("dataset", type=str, help="The base directory path of the dataset.")
    
    # train or val or test
    parser.add_argument(
        "--type", 
        type=str, 
        default='train', 
        help="train or val or test. Default is train.")
    
    # Specifying the name of the JSON file.
    parser.add_argument(
        "--name",
        type=str, 
        default='auto', 
        help="Specify the JSON file name. The default is '[TYPE]_annotations.json'.")
    
    args = parser.parse_args()

    return args

def annotations(mask_path):
    tmps = []
    
    mask_files = glob.glob(os.path.join(mask_path, "*"))
    mask_files.sort()
    
    annotation_id = 1
    
    for i, file in tqdm(enumerate(mask_files)):
        img = cv2.imread(file)

        seg, seg_bool = background_del(img)
        cluster, cluster_color = assign_cluster_number(seg, seg_bool)

        for c in range(1,cluster.max()+1):
            tmp = cl.OrderedDict()

            cluster_temp = (np.where(cluster == c, 255, 0)).astype("u1")

            # Contour extraction (CHAIN_APPROX_TC89_L1: Omits contour points that can be approximated by a straight line for parts of the contour)
            contours, _ = cv2.findContours(cluster_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            segmentation_list = []

            # point[0] -> [X, Y]
            for contour in contours:
                for point in contour:
                    segmentation_list.append(int(point[0, 0]))
                    segmentation_list.append(int(point[0, 1]))

            nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(cluster_temp, 8, cv2.CV_16U, cv2.CCL_DEFAULT)
            
            x, y, width, height, area = data[1]
    
            tmp["segmentation"] = [segmentation_list]
            tmp["id"] = annotation_id
            tmp["image_id"] = i + 1
            tmp["category_id"] = 1
            tmp["area"] = float(area)
            tmp["iscrowd"] = 0
            tmp["bbox"] =  [float(x), float(y), float(width), float(height)]
            
            if(len(segmentation_list)>4):
                tmps.append(tmp)
                annotation_id += 1
    
    return tmps

def main():

    args = parse_args()

    img_path = os.path.join(args.dataset, "images", args.type)
    mask_path = os.path.join(args.dataset, "masks", args.type)
    if(args.name == "auto"):
        json_path = os.path.join(args.dataset, args.type + "_annotations.json")
    else:
        json_path = os.path.join(args.dataset, args.name)
    
    query_list = ["info", "licenses", "images", "annotations", "categories", "segment_info"]
    js = {
        "info": [],
        "licenses": [],
        "images":[],
        "annotations":[],
        "categories":[],
        "segment_info":[]
    }
    
    for i in range(len(query_list)):
        tmp = []
        # Info
        if query_list[i] == "info":
            tmp = info()
    
        # licenses
        elif query_list[i] == "licenses":
            tmp = licenses()
    
        elif query_list[i] == "images":
            tmp = images(img_path)
    
        elif query_list[i] == "annotations":
            tmp = annotations(mask_path)
    
        elif query_list[i] == "categories":
            tmp = categories()

        # save it
        js[query_list[i]] = tmp
    
    # write
    fw = open(json_path,'w')
    json.dump(js,fw,indent=2)

if __name__=='__main__':
    main()