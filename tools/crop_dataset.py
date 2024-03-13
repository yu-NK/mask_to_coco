"""
File Name: crop_datset.py
Description: 
    This script is designed to randomly crop images from an existing dataset
    in COCO format and generate a new dataset. It does not extend to the creation of JSON files.

Author: Yuki Naka
Created Date: 2023-10-22
Last Modified: 2024-03-13

Usage:
    python crop_dataset.py <dataset> -t [train|val|test] -x [crop width] -y [crop height] -n [number of crops per image] -o [output directory]

    <dataset>              : The base directory path of the dataset.
    -t, --type             : The dataset type (train, val, or test). Defaults to 'train'.
    -x, --crop-width       : Width of the cropped image. Default is 150 pixels.
    -y, --crop-height      : Height of the cropped image. Default is 150 pixels.
    -n, --crop-num         : Number of cropped images to generate per original image. Default is 10.
    -o, --dir-out          : The output directory for cropped images and their annotations. Default is './out/crop_dataset'.
    
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
        description="This script is designed to randomly crop images from an existing dataset"
                    "in COCO format and generate a new dataset. It does not extend to the creation of JSON files."
    )

    parser.add_argument("dataset", type=str, help="The base directory path of the dataset.")
    
    parser.add_argument(
        "-t", "--type",
        dest="type",
        type=str, 
        default='train', 
        help="The dataset type: train, val, or test. Defaults to 'train'.")

    # Set the crop size
    parser.add_argument(
        "-x", "--crop-width",
        dest="crop_width" ,
        type=int, 
        default=150, 
        help="width of the cropped image. Default is 150.")
    
    parser.add_argument(
        "-y", "--crop-height",
        dest="crop_height",
        type=int, 
        default=150, 
        help="height of the cropped image. Default is 150.")
    
    # Number of cropped images to generate per image
    parser.add_argument(
        "-n", "--crop-num",
        dest="crop_num",
        type=int, 
        default=10, 
        help="Number of Crop Image Generations per Image. Default is 10.")

    # Set the output directory
    parser.add_argument(
        "-o", "--output",
        dest="dir_out", 
        type=str, 
        default='out/crop_dataset', 
        help="The output directory. Default is ./out/crop_dataset")

    args = parser.parse_args()

    return args

def main():

    args = parse_args()

    DIR_INPUT_JSON = os.path.join(args.dataset, args.type + "_annotations.json")
    with open(DIR_INPUT_JSON, 'r') as f:
        coco_data = json.load(f)
        
    DIR_CROP_IMAGE = args.dir_out + "images/" + args.type
    DIR_CROP_MASK  = args.dir_out + "masks/"  + args.type
    
    os.makedirs(DIR_CROP_IMAGE, exist_ok=True)
    os.makedirs(DIR_CROP_MASK , exist_ok=True)

    new_image_id = 1

    # Randomly select and crop images from the original dataset

    for image_info in tqdm(coco_data["images"]):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        if annotations:

            # Load Images
            image = Image.open(os.path.join(args.dataset ,file_name))
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

                    # Code to crop areas containing foreground pixels (needs to be adjusted according to the dataset)
                    val_array = np.array(cropped_image)
                    
                    val_NLMD = cv2.fastNlMeansDenoising(val_array, h=6)
                    th_img = np.where(val_NLMD > 50, 255, 0)

                    white_pixel_count = np.count_nonzero(th_img == 255)
                    if(white_pixel_count>1000): flag = False
                    #########################################################################

                # Generate random colors (default is 100)
                colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(100)])

                new_file_name = f'{new_image_id:06d}.png'
                cropped_image.save(os.path.join(DIR_CROP_IMAGE, new_file_name))

                mask = Image.new('L', (width, height), 0)
                mask_data = np.zeros((args.crop_height, args.crop_width), dtype = "u1")

                # Initialize a new annotation ID
                new_annotation_id = 1

                # Update the corresponding annotation information
                for annotation in annotations:
                    x, y, w, h = annotation["bbox"]

                    mask = Image.new('L', (width, height), 0)

                    # Draw the segmentation masks of the annotations
                    draw = ImageDraw.Draw(mask)
                    for seg in annotation["segmentation"]:
                        
                        # Convert segmentation coordinates to fit within the cropped image
                        seg_temp = [(int(x), int(y)) for x, y in zip(seg[::2], seg[1::2])]
                        draw.polygon(seg_temp, fill= int(new_annotation_id))

                    temp = np.array(mask)[y1:y2, x1:x2]

                    # Calculate the area of the annotation
                    annotation_area = np.sum(temp)

                    if annotation_area:

                        mask_binary = (np.where(temp==new_annotation_id, 255, 0)).astype("u1")
                        nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(mask_binary, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

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
                
                cv2.imwrite(os.path.join(DIR_CROP_MASK, new_file_name), mask_col_data)

                new_image_id += 1
                
if __name__ == "__main__":
    main()