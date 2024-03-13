"""
File Name: mask_to_coco_parallel_segtype.py
Description: 
    - This script is designed to create a COCO-format dataset from images and their
      corresponding mask files for instance segmentation tasks. (Parallelized version)
    - It generates a COCO format dataset (JSON file) from mask images annotated 
      for instance segmentation of a single class.
    - Editing info, licenses, categories required for COCO format in 
      ./config/coco_config.py is necessary.
    - The format of the output segmentation part can be chosen between polygon and RLE (Run-Length Encoding) formats.

Author: Yuki Naka
Created Date: 2024-02-18
Last Modified: 2024-03-12

Usage:
    python mask_to_coco_parallel_segtype.py <dataset> --type [train|val|test] --ply --name [output JSON file name] --proc-num [number of cores]

    <dataset>:
        The base directory path of the dataset where the images and masks are stored.
    --type [train|val|test]: 
        Specifies the dataset type. It can be 'train', 'val', or 'test'. Default is 'train'.
    --ply:
        Outputs in polygon format if selected. Defaults to RLE format.
        A choice between RLE and Polygon formats for the output segmentation annotations, with RLE as the default format.
    --name [output JSON file name]: 
        Specifies the name of the output JSON file for COCO annotations. 
        The default is auto-generated based on the type, following the pattern '[TYPE]_annotations.json'.
    --proc-num [number of cores]:
        The number of cores used for parallelization. Defaults to the system's physical core count.

Dependencies:
    - Python 3.x, numpy, OpenCV (cv2), tqdm, pycocotools

    Additional configuration files:
    - config/coco_config.py: Contains settings for the COCO dataset.

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

# COCO Format Handling
import pycocotools.mask as mask

# Specified in config/coco_config.py
from config.coco_config import info, licenses, images, categories

def parse_args():
    parser = argparse.ArgumentParser(
        description=("Create a dataset in COCO format from mask images for instance segmentation "
                     "(single class) with parallel processing. Allows choosing between "
                     "RLE (Run-Length Encoding) and Polygon format for segmentation. **Installation of pycocotools is required**")
    )
    
    # Specify the directory containing images to be converted to COCO format
    parser.add_argument("dataset", type=str, help="The base directory path for the dataset.")
    
    # Specify the dataset type (train, validation, or test)
    parser.add_argument(
        "--type",
        type=str, 
        default='train', 
        help="The dataset type: train, val, or test. Defaults to 'train'."
    )

    # Option for output in polygon format (default is RLE)
    parser.add_argument(
        "--ply", 
        action='store_true',
        help="Outputs in polygon format if selected. Defaults to RLE format."
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

def polygonFromMask(maskedArr):
    # adapted from https://github.com/hazirbas/coco-json-converter/blob/master/generate_coco_json.py
    contours, _ = cv2.findContours(maskedArr, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    segmentation = []
    valid_poly = 0
    for contour in contours:
        # Valid polygons have >= 6 coordinates (3 points)
        if contour.size >= 6:
            segmentation.append(contour.astype(float).flatten().tolist())
            valid_poly += 1
    #if valid_poly == 0:
        #raise ValueError
    return segmentation

def generate_unique_color_mask_opencv(image_path):

    """
    Generates a mask array from an image, where each unique color is represented by a unique index in the mask. 
    This function uses OpenCV to read the image and numpy to handle array operations. 
    The resulting mask array can be used to identify and separate unique colors in the image.

    Parameters:
        image_path (str): The path to the image file.

    Returns:
        numpy.ndarray: An array where each pixel's value corresponds to the index of its color in the unique colors found in the image.
    """

    # Load the image using OpenCV
    image = cv2.imread(image_path)
    
    # Create a set of unique colors
    unique_colors, unique_indices = np.unique(image.reshape(-1, 3), axis=0, return_inverse=True)
    
    # Use the unique color indices to create a mask array
    mask_array = unique_indices.reshape(image.shape[:2])
    
    return mask_array #, {tuple(color): idx for idx, color in enumerate(unique_colors)}


def find_coordinates(binary_image):

    """
    Finds the bounding box coordinates for a given binary image where the target objects are marked with non-zero values (e.g., 1 for foreground). 
    This function calculates and returns the bounding box in the format [x_min, y_min, width, height], 
    where (x_min, y_min) is the top-left coordinate of the bounding box, and 'width' and 'height' are the dimensions of the bounding box. 

    Parameters:
        binary_image (numpy.ndarray): A binary image array where target pixels have a non-zero value.

    Returns:
        list: The bounding box coordinates and dimensions [x_min, y_min, width, height], all as float values.
    """

    # Find the coordinates of the target pixels (e.g., pixels with value 1) in the image
    coordinates = np.argwhere(binary_image > 0)
    
    # Calculate the coordinates of the top-left (minimum) and bottom-right (maximum) points
    top_left = coordinates.min(axis=0)
    bottom_right = coordinates.max(axis=0)
    
    # Extract coordinates
    top_left_y, top_left_x = top_left
    bottom_right_y, bottom_right_x = bottom_right

    # Calculate width and height
    width = bottom_right_x - top_left_x
    height = bottom_right_y - top_left_y

    return [float(top_left_x), float(top_left_y), float(width), float(height)]

def annotations(i, file, format):
    
    tmps = []
    
    annotation_id = 1
    
    img = cv2.imread(file)

    cluster = generate_unique_color_mask_opencv(file)
    
    annotation_id = 1

    for c in range(1,cluster.max()+1):
        cluster_temp = (np.where(cluster == c, 255, 0)).astype("u1")
        
        nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(cluster_temp, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

        sizes = data[:, 4]
        label_fix = np.where(sizes < 3, 0, np.arange(0, nLabels))

        valid_labels = np.logical_and(labelImages > 0, label_fix[labelImages.astype(int)] > 0)

        fix_array = np.zeros((img.shape[0], img.shape[1]),dtype="u1")
        fix_array[valid_labels] = label_fix[labelImages[valid_labels]]

        for ann_label in range(1,fix_array.max() + 1):

            tmp = cl.OrderedDict()

            seg = np.zeros((img.shape[0], img.shape[1]),dtype="u1")
            mask_ind = np.argwhere(fix_array == ann_label)
            seg[mask_ind[:, 0], mask_ind[:, 1]] = 255

            if((seg > 0).sum()>2):
                
                tmp = cl.OrderedDict()
                
                tmp["id"] = i * 10000 + annotation_id
                tmp["image_id"] = i + 1
                tmp["category_id"] = 1

                # RLE
                if(format == False):
                    coco_rle = mask.encode(np.asfortranarray(seg))
                    coco_rle["counts"] = coco_rle["counts"].decode('utf-8')
                    tmp["segmentation"] = coco_rle
                # Polygon
                elif(format == True):
                    tmp["segmentation"] = polygonFromMask(cluster_temp)
                    
                tmp["area"] = float((seg > 0.0).sum())
                tmp["bbox"] =  find_coordinates(seg)
                tmp["iscrowd"] = 0
                
                tmps.append(tmp)
                annotation_id += 1
    
    return tmps

def process_images(args):
    start, end, mask_file, process_num, format = args
    results = []
    with tqdm(range(start, end)) as pbar:
        for i in pbar:
            pbar.set_description(f"[Process {process_num:02}] {i:04}")
        
            result = annotations(i, mask_file[i], format)
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
        parallel_args.append(np.array([start, end, mask_files, i, args.ply],dtype=object))
    
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
    json.dump(js, fw)
    
    print("Finished writing JSON file.")

if __name__=='__main__':
    main()