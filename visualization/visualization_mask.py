"""
File Name: visualization_mask.py
Description: 
    This script is intended to visualize dataset masks in COCO format, facilitating 
    the analysis and verification of mask annotations for instance segmentation tasks. 

Author: Yuki Naka
Created Date: 2023-10-17
Last Modified: 2024-03-13

Usage:
    python visualization_mask.py <dataset> --type [train|val|test] --output [output directory]

    <dataset>    : The base directory path of the dataset containing COCO format mask images.
    --type [train|val|test]: 
        Specifies the dataset type. It can be 'train', 'val', or 'test'. Default is 'train'.
    --output     : The output directory where mask visualizations will be saved. Defaults to './out'.
    
Dependencies:
    - Python 3.x
    - Numerical Operations: numpy
    - Image Processing: PIL (Image, ImageDraw), OpenCV (cv2)
    - Utility: tqdm

License: MIT License
"""

# Standard Libraries
import json
import os
import random
import argparse

# Numerical Operations
import numpy as np

# Image Processing
from PIL import Image, ImageDraw
import cv2

# Utility
from tqdm import tqdm

def parse_args():

    parser = argparse.ArgumentParser(
        description="Code to Visualize Dataset Masks(COCO Format)"
    )
    
    parser.add_argument("dataset", type=str, help="The base directory path of the dataset.")
    
    parser.add_argument(
        "--type",
        type=str, 
        default='train', 
        help="The dataset type: train, val, or test. Defaults to 'train'."
    )

    parser.add_argument(
        "--output", 
        type=str, 
        default='out', 
        help="The output directory. Default is ./out")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    DIR_INPUT_JSON = os.path.join(args.dataset, args.type + "_annotations.json")
    
    # Load the COCO dataset JSON file
    with open(DIR_INPUT_JSON, 'r') as f:
        coco_data = json.load(f)
    
    os.makedirs(args.output, exist_ok=True)

    for image_info in tqdm(coco_data["images"]):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        if annotations:
            # Load images
            image = Image.open(os.path.join(args.dataset ,file_name))
            width, height = image.size

            # Generate random colors (default is 100)
            colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(100)])

            mask = Image.new('L', (width, height), 0)
            mask_col_data = np.zeros((height, width, 3), dtype = "u1")

            # Initialize a new annotation ID
            new_annotation_id = 1

            mask = Image.new('L', (width, height), 0)
            
            # Draw the segmentation masks of the annotations
            draw = ImageDraw.Draw(mask)
            
            for annotation in annotations:
                for seg in annotation["segmentation"]:

                    seg_temp = [(int(x), int(y)) for x, y in zip(seg[::2], seg[1::2])]
                    draw.polygon(seg_temp, fill= int(new_annotation_id))
                    
                    new_annotation_id += 1
                    
            mask_array = np.array(mask)

            mask_indices = mask_array > 0
            mask_col_data[mask_indices] = colors[mask_array[mask_indices]]

            cv2.imwrite(os.path.join(args.output, os.path.splitext(os.path.basename(file_name))[0]+".png"), mask_col_data)
                
if __name__ == "__main__":
    main()