"""
File Name: mask_coco_coco_parallel_polygon.py
Description: 
    - This script is designed to create a COCO-format dataset from images and their
      corresponding mask files for instance segmentation tasks. (Parallelized version)
    - It generates a COCO format dataset (JSON file) from mask images annotated 
      for instance segmentation of a single class.
    - Editing info, licenses, categories required for COCO format in 
      ./config/coco_config.py is necessary.
    - The output segmentation format will be in polygons.

Author: Yuki Naka
Created Date: 2023-10-22
Last Modified: 2024-03-12

Usage:
    python mask_coco_coco_parallel_polygon.py <dataset> --type [train|val|test] --name [output JSON file name] --proc-num [number of cores]

    <dataset>:
        The base directory path of the dataset where the images and masks are stored.
    --type [train|val|test]: 
        Specifies the dataset type. It can be 'train', 'val', or 'test'. Default is 'train'.
    --name [output JSON file name]: 
        Specifies the name of the output JSON file for COCO annotations. 
        The default is auto-generated based on the type, following the pattern '[TYPE]_annotations.json'.
    --proc-num [number of cores]:
        The number of cores used for parallelization. Defaults to the system's physical core count.

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

# Parallel Processing
import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

# Specified in config/coco_config.py
from config.coco_config import info, licenses, images, categories
from utils.tools import background_del, assign_cluster_number

def parse_args():
    parser = argparse.ArgumentParser(
        description="Create a dataset in COCO format from mask images for instance segmentation "
                    "(single class) with parallel processing. The output segmentation format will be in polygons."
    )
    
    # Specify the dataset type (train, validation, or test)
    parser.add_argument(
        "--type",
        type=str, 
        default='train', 
        help="The dataset type: train, val, or test. Defaults to 'train'."
    )
    
    # Specify the JSON file name
    parser.add_argument(
        "--name",
        type=str,
        default='auto', 
        help="The JSON file name for COCO annotations. Defaults to '[TYPE]_annotations.json'."
    )
    
    # Specify the number of cores for parallel processing
    parser.add_argument(
        "--proc-num", 
        type=int, 
        default=multiprocessing.cpu_count(), 
        help="The number of cores used for parallelization. Defaults to the system's physical core count."
    )

    args = parser.parse_args()

    return args

def annotations(i, file):
    
    tmps = []
    
    annotation_id = 1
    
    img = cv2.imread(file)

    seg, seg_bool = background_del(img)
    cluster, cluster_color = assign_cluster_number(seg, seg_bool)
    
    annotation_id = 1

    for c in range(1,cluster.max()+1):
        cluster_temp = (np.where(cluster == c, 255, 0)).astype("u1")

        nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(cluster_temp, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

        sizes = data[:, 4]
        label_fix = np.where(sizes < 15, 0, np.arange(0, nLabels))

        valid_labels = np.logical_and(labelImages > 0, label_fix[labelImages.astype(int)] > 0)

        fix_array = np.zeros((img.shape[0], img.shape[1]),dtype="u1")
        fix_array[valid_labels] = label_fix[labelImages[valid_labels]]

        for ann_label in range(1,fix_array.max() + 1):

            tmp = cl.OrderedDict()

            seg = np.zeros((img.shape[0], img.shape[1]),dtype="u1")
            mask_ind = np.argwhere(fix_array == ann_label)
            seg[mask_ind[:, 0], mask_ind[:, 1]] = 255

            # Contour extraction (CHAIN_APPROX_TC89_L1: Omits contour points that can be approximated by a straight line for parts of the contour)
            contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            segmentation_list = []

            # point[0] -> [X, Y]
            for contour in contours:
                for point in contour:
                    segmentation_list.append(int(point[0, 0]))
                    segmentation_list.append(int(point[0, 1]))

            nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(seg, 8, cv2.CV_16U, cv2.CCL_DEFAULT)
            if nLabels>1:
                x, y, width, height, area = data[1]
                
                tmp["id"] = i * 10000 + annotation_id
                tmp["image_id"] = i + 1
                tmp["category_id"] = 1
                tmp["segmentation"] = [segmentation_list]
                tmp["area"] = float(area)
                tmp["bbox"] =  [float(x), float(y), float(width), float(height)]
                tmp["iscrowd"] = 0

                if(len(segmentation_list)>4):
                    tmps.append(tmp)
                    annotation_id += 1
    
    return tmps

def process_images(args):
    start, end, mask_file, process_num = args
    results = []
    with tqdm(range(start, end)) as pbar:
        for i in pbar:
            pbar.set_description(f"[Process {process_num:02}] {i:04}")
        
            result = annotations(i, mask_file[i])
            results.extend(result)
        
    return results

def main():

    args = parse_args()

    img_path = os.path.join(args.dataset, "images", args.type)
    mask_path = os.path.join(args.dataset, "masks", args.type)
    if args.name == "auto":
        json_path = os.path.join(args.dataset, args.type + "_annotations.json")
    else:
        json_path = os.path.join(args.dataset, args.name)
        
    mask_files = glob.glob(os.path.join(mask_path, "*"))
    mask_files.sort()
    
    print("Start processing. CPU core: ", args.proc_num)
    
    parallel_args = []
    for i in range(args.proc_num):
        start = i * len(mask_files) // args.proc_num
        end = (i + 1) * len(mask_files) // args.proc_num
        parallel_args.append(np.array([start, end, mask_files, i],dtype=object))
    
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_images, parallel_args)
        
    ann_result = []
    for result in results:
        ann_result.extend(result)

    print("Begin writing the JSON file.")
    js = {
        "licenses": licenses(),
        "info": info(),
        "categories": categories(),
        "images": images(img_path),
        "annotations": ann_result
    }
    
    fw = open(json_path,'w')
    json.dump(js, fw, indent=2)
    
    print("Finished writing JSON file.")

if __name__=='__main__':
    main()