"""
File Name: crop_image-mask_resize.py
Description: 
    - This script is designed to generate cropped images from directories containing images 
      and their corresponding mask images, where mask images are colored differently for each object.
    - The cropped images are then resized to a specified size. 

Author: Yuki Naka
Created Date: 2023-11-01
Last Modified: 2024-03-13

Usage:
    python crop_image-mask_resize.py <dataset> <output> 
        --type [train|val|test] --crop-n [number of crops per image] 
        --crop-w [crop width] --crop-h [crop height] 
        --resize-w [resize width] --resize-h [resize height]

    <dataset>:
        The base directory path of the dataset containing images and masks.
    <output>:
        Storage location for cropped and resized images and masks.
    --type [train|val|test]:
        Specifies the dataset type. Defaults to 'train'.
    --crop-n [number of crops per image]:
        Number of cropped images to generate per original image. Default is 10.
    --crop-w [crop width]:
        Width of the cropped image. Default is 50.
    --crop-h [crop height]:
        Height of the cropped image. Default is 50.
    --resize-w [resize width]:
        Width of the image after resizing. Default is 150.
    --resize-h [resize height]:
        Height of the image after resizing. Default is 150.

Dependencies:
    - Numerical Operations: numpy
    - Image Processing: PIL (Image, ImageDraw), OpenCV (cv2)
    - Utility: tqdm
    - Custom Utilities: utils.tools (background_del, assign_cluster_number)

License: MIT License
"""

# Standard Libraries
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

sys.path.append("..")
from utils.tools import background_del, assign_cluster_number

def parse_args():

    parser = argparse.ArgumentParser(
        description="Code to generate cropped images from directories of images and mask images (mask images "
                    "colored differently for each object). The generated cropped images are resized to a specified image size."
    )
    
    parser.add_argument("dataset", type=str, help="The base directory path of the dataset.")
    parser.add_argument("output", type=str, help="Storage location for cropped images and masks")
    
    parser.add_argument(
        "--type",
        dest="type",
        type=str, 
        default='train', 
        help="The dataset type: train, val, or test. Defaults to 'train'.")
    
    # Number of cropped images to generate per image
    parser.add_argument(
        "--crop-n",
        dest="crop_num",
        type=int, 
        default=10, 
        help="Number of Crop Image Generations per Image. Default is 10.")
    
    # Set the crop size
    parser.add_argument(
        "--crop-w",
        dest="crop_width",
        type=int, 
        default=50, 
        help="width of the cropped image. Default is 150.")
    
    parser.add_argument(
        "--crop-h",
        dest="crop_height",
        type=int, 
        default=50, 
        help="height of the cropped image. Default is 150.")
    
    # Set the image size after resizing
    parser.add_argument(
        "--resize-w",
        dest="resize_width" ,
        type=int, 
        default=150, 
        help="width after resizing. Default is 150.")
    
    parser.add_argument(
        "--resize-h",
        dest="resize_height",
        type=int, 
        default=150, 
        help="height after resizing. Default is 150.")

    args = parser.parse_args()

    return args

def main():

    args=parse_args()

    img_files = glob.glob(os.path.join(args.dataset, "images", args.type, "*"))
    mask_files = glob.glob(os.path.join(args.dataset, "masks", args.type, "*"))
    
    img_files.sort()
    mask_files.sort()
        
    DIR_CROP_IMAGE = os.path.join(args.output, "images", args.type)
    DIR_CROP_MASK  = os.path.join(args.output, "masks", args.type)
    
    os.makedirs(DIR_CROP_IMAGE, exist_ok=True)
    os.makedirs(DIR_CROP_MASK , exist_ok=True)

    new_image_id = 1

    with tqdm(range(len(img_files))) as pbar:
        for file_num in pbar:
        
            img_path  = img_files[file_num]
            mask_path = mask_files[file_num]

            img_name  = os.path.splitext(os.path.basename(img_path))[0]
            
            pbar.set_description(f"Process: {img_name}")

            # Load Image
            image = (Image.open(img_path)).convert('L')
            mask  = (Image.open(mask_path)).convert('RGB')
            width, height = image.size

            for i in range(args.crop_num):

                flag = True

                while(flag):

                    # Set random coordinates for cropping
                    x1 = random.randint(0, width - args.crop_width)
                    y1 = random.randint(0, height - args.crop_height)
                    x2 = x1 + args.crop_width
                    y2 = y1 + args.crop_height

                    # Save the cropped image
                    cropped_image = image.crop((x1, y1, x2, y2))
                    cropped_mask  = mask.crop((x1, y1, x2, y2))

                    # Code to crop areas containing foreground pixels (needs to be adjusted according to the dataset)
                    cropped_image_np = np.array(cropped_image)

                    cropped_image_NLMD = cv2.fastNlMeansDenoising(cropped_image_np, h=6)
                    cropped_image_th = np.where(cropped_image_NLMD > 100, 255, 0)

                    white_pixel_count = np.count_nonzero(cropped_image_th == 255)
                    if(white_pixel_count>200): flag = False
                    #########################################################################

                new_file_name = img_name + f'_{i:03d}_crop_resize.png'
                cropped_image_resize = cropped_image.resize((args.resize_width, args.resize_height))
                cropped_image_resize.save(os.path.join(DIR_CROP_IMAGE, new_file_name))

                cropped_mask_np = np.array(cropped_mask)
                seg, seg_bool = background_del(cropped_mask_np)
                cluster, cluster_color = assign_cluster_number(seg, seg_bool)

                # Generate random colors (default is 100)
                colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(100)])

                mask_data = np.zeros((args.crop_height, args.crop_width), dtype = "u2")

                # Initialize a new annotation ID
                new_annotation_id = 1

                for c in range(1,cluster.max()+1):
                    cluster_temp = (np.where(cluster == c, 255, 0)).astype("u1")

                    nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(cluster_temp, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

                    label_fix = np.zeros(nLabels-1)
                    sizes = data[:, 4]
                    label_fix = np.where(sizes < 15, 0, np.arange(0, nLabels))

                    valid_labels = np.logical_and(labelImages > 0, label_fix[labelImages.astype(int)] > 0)

                    if len(label_fix[labelImages[valid_labels]]):

                        fix_array = np.zeros((args.crop_width, args.crop_height),dtype="u1")
                        fix_array[valid_labels] = label_fix[labelImages[valid_labels]] + new_annotation_id - 1
                        mask_data = np.where(fix_array > 0, fix_array, mask_data)

                        new_annotation_id += label_fix[labelImages[valid_labels]].max()

                mask_col_data = np.zeros((args.crop_width, args.crop_height, 3), dtype = "u1")

                mask_indices = mask_data > 0
                mask_col_data[mask_indices] = colors[mask_data[mask_indices]]

                mask_col_resize = cv2.resize(mask_col_data, (args.resize_height, args.resize_width), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(os.path.join(DIR_CROP_MASK, new_file_name), mask_col_resize)

                new_image_id += 1
                
if __name__ == "__main__":
    main()