"""
File Name: crop_rotate.py
Description: 
    This script is designed for cropping an image and a corresponding mask image while applying rotation, 
    facilitating the generation of augmented datasets.

Author: Yuki Naka
Created Date: 2023-11-07
Last Modified: 2024-03-13

Usage:
    python crop_rotate.py <dataset> <output> 
        --crop-w [width] --crop-h [height] --step [angle of rotation] 
        --type [train|val|test] --rot-x [x-coordinate] --rot-y [y-coordinate]

    <dataset>:
        The base directory path of the dataset containing images and masks.
    <output>:
        Storage location for cropped images and masks.

    --crop-w [width]:
        Width of the cropped image. Default is 900 pixels.
    --crop-h [height]:
        Height of the cropped image. Default is 32 pixels.
    --step [angle of rotation]:
        Angle of rotation. Default is 1.
    --type [train|val|test]:
        Specifies the dataset type. Default is 'train'.
    --rot-x [x-coordinate]:
        X-coordinate of the rotation center. Default is 421.
    --rot-y [y-coordinate]:
        Y-coordinate of the rotation center. Default is 435.

Dependencies:
    - Numerical Operations: numpy
    - Image Processing: PIL (Image, ImageDraw), OpenCV (cv2)
    - Utility: tqdm

License: MIT License
"""

# Standard Libraries
import json
import sys
import os
import glob
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
        description="Code for cropping an image and a mask image while rotating"
    )
    
    parser.add_argument("dataset", type=str, help="The base directory path of the dataset.")
    parser.add_argument("output", type=str, help="Storage location for cropped images and masks")
    
    parser.add_argument(
        "--crop-w", 
        dest="crop_width", 
        type=int, 
        default=900, 
        help="width of the cropped image. Default is 900.")
    parser.add_argument(
        "--crop-h", 
        dest="crop_height", 
        type=int, 
        default=32, 
        help="height of the cropped image. Default is 32.")
    
    parser.add_argument(
        "--step",
        dest="angle_step",
        type=int, 
        default=1, 
        help="angle of rotation. Default is 1.")
    
    parser.add_argument(
        "--type",
        dest="type",
        type=str, 
        default='train', 
        help="The dataset type: train, val, or test. Defaults to 'train'.")
    
    parser.add_argument(
        "--rot-x", 
        dest="rotate_x", 
        type=int, 
        default=421, 
        help="X-coordinate of the rotation center. Default is 421.")
    parser.add_argument(
        "--rot-y", 
        dest="rotate_y", 
        type=int, 
        default=435, 
        help="Y-coordinate of the rotation center. Default is 435.")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    img_files = glob.glob(os.path.join(args.dataset, "images", args.type, "*"))
    mask_files = glob.glob(os.path.join(args.dataset, "masks", args.type, "*"))

    img_files.sort()
    mask_files.sort()

    DIR_CROP_IMAGE = os.path.join(args.output, "images", args.type)
    DIR_CROP_MASK  = os.path.join(args.output, "masks", args.type)

    os.makedirs(DIR_CROP_IMAGE, exist_ok=True)
    os.makedirs(DIR_CROP_MASK , exist_ok=True)

    # Set the average value of background pixels.
    bg_mean = 33

    with tqdm(range(len(img_files))) as pbar:
        for file_num in pbar:
            
            img_path  = img_files[file_num]
            mask_path = mask_files[file_num]

            img_name  = os.path.splitext(os.path.basename(img_path))[0]
            
            pbar.set_description(f"Process: {img_name}")

            image = (Image.open(img_path)).convert('L')
            mask  = (Image.open(mask_path)).convert('RGB')
            
            center_x, center_y = image.size[0] // 2, image.size[1] // 2
            center_x_fix, center_y_fix = image.size[0] // 2 - args.rotate_x, image.size[1] // 2 - args.rotate_y
            
            # Move the center coordinates to the center of the flower.
            image_center = image.rotate(0, translate=(center_x_fix, center_y_fix))
            mask_center  = mask.rotate(0, translate=(center_x_fix, center_y_fix))
            
            # Fill the margin areas with random luminance values within ±5 of the average background pixel value.
            img_center_np = np.array(image_center)
            zero_positions = np.argwhere(img_center_np == 0)
            random_values = np.random.randint(27, 41, size=zero_positions.shape[0])
            img_center_np[zero_positions[:, 0], zero_positions[:, 1]] = random_values
            
            image = (Image.fromarray(img_center_np)).convert('L')
            
            rect_x = center_x - args.crop_width // 2
            rect_y = center_y - args.crop_height // 2
            
            for angle in range(0, 360, args.angle_step):
                rotated_image = image.rotate(angle, resample=Image.BICUBIC)
                rotated_mask  = mask_center.rotate(angle)
                
                crop_image = rotated_image.crop((rect_x, rect_y, rect_x + args.crop_width, rect_y + args.crop_height))
                crop_mask  = rotated_mask.crop((rect_x, rect_y, rect_x + args.crop_width, rect_y + args.crop_height))
                
                crop_image_np = np.array(crop_image)
                zero_positions = np.argwhere(crop_image_np == 0)
                random_values = np.random.randint(27, 41, size=zero_positions.shape[0])
                crop_image_np[zero_positions[:, 0], zero_positions[:, 1]] = random_values

                crop_image = (Image.fromarray(crop_image_np)).convert('L')
                
                new_file_name = img_name + f'_{angle:03d}.png'
                
                crop_image.save(os.path.join(DIR_CROP_IMAGE, new_file_name))
                crop_mask.save(os.path.join(DIR_CROP_MASK, new_file_name))
    
if __name__ == "__main__":
    main()