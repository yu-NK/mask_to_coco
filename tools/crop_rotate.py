import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import glob

import argparse
import sys

def get_args():
    # 準備
    parser = argparse.ArgumentParser(
        description="Code for cropping an image and a mask image while rotating"
    )

    # 標準入力以外の場合
    parser = argparse.ArgumentParser()
    
    # クロップ元の画像とマスク画像が含まれているディレクトリの指定
    parser.add_argument("input_dir", type=str, help="The base directory path of images and masks.")
    # 出力先のディレクトリの指定
    parser.add_argument("output_dir", type=str, help="Storage location for cropped images and masks")
    
    # クロップサイズを設定
    parser.add_argument("--crop-w", dest="width", type=int, default=900, help="width of the cropped image. Default is 900.")
    parser.add_argument("--crop-h", dest="height", type=int, default=32, help="height of the cropped image. Default is 32.")
    
    # 画像の回転ステップ
    parser.add_argument("--step", dest="step", type=int, default=10, help="Number of Crop Image Generations per Image. Default is 10.")
    
    # train or val or test
    parser.add_argument("--type", dest="type", type=str, default='train', help="train or val or test. Default is train.")
    
    # 回転中心
    parser.add_argument("--rotate-x", dest="center_x", type=int, default=421, help="X-coordinate of the rotation center. Default is 421.")
    parser.add_argument("--rotate-y", dest="center_y", type=int, default=435, help="Y-coordinate of the rotation center. Default is 435.")

    args = parser.parse_args()

    # 引数から画像番号
    DIR_INPUT   = args.input_dir
    DIR_OUTPUT  = args.output_dir
    crop_width  = args.width
    crop_height = args.height
    angle_step  = args.step
    output_type = args.type
    rotate_x    = args.center_x
    rotate_y    = args.center_y  

    return DIR_INPUT, DIR_OUTPUT, crop_width, crop_height, angle_step, output_type, rotate_x, rotate_y

def main(DIR_INPUT, DIR_OUTPUT, crop_width, crop_height, angle_step, output_type, rotate_x, rotate_y):

    img_files = glob.glob(os.path.join(DIR_INPUT, "images", output_type, "*"))
    mask_files = glob.glob(os.path.join(DIR_INPUT, "masks", output_type, "*"))

    img_files.sort()
    mask_files.sort()

    DIR_CROP_IMAGE = os.path.join(DIR_OUTPUT, "images", output_type)
    DIR_CROP_MASK  = os.path.join(DIR_OUTPUT, "masks", output_type)

    os.makedirs(DIR_CROP_IMAGE, exist_ok=True)
    os.makedirs(DIR_CROP_MASK , exist_ok=True)

    with tqdm(range(len(img_files))) as pbar:
        for file_num in pbar:
            
            img_path  = img_files[file_num]
            mask_path = mask_files[file_num]

            img_name  = os.path.splitext(os.path.basename(img_path))[0]
            
            pbar.set_description(f"Process: {img_name}")

            image = (Image.open(img_path)).convert('L')
            mask  = (Image.open(mask_path)).convert('RGB')
            
            center_x, center_y = image.size[0] // 2, image.size[1] // 2
            center_x_fix, center_y_fix = image.size[0] // 2 - rotate_x, image.size[1] // 2 - rotate_y
            
            # 中心座標を花弁の中心に移動
            image_center = image.rotate(0, translate=(center_x_fix, center_y_fix))
            mask_center  = mask.rotate(0, translate=(center_x_fix, center_y_fix))
            
            # 余白の部分を27〜41のランダムな輝度値で埋める
            img_center_np = np.array(image_center)
            zero_positions = np.argwhere(img_center_np == 0)
            random_values = np.random.randint(27, 41, size=zero_positions.shape[0])
            img_center_np[zero_positions[:, 0], zero_positions[:, 1]] = random_values
            
            image = (Image.fromarray(img_center_np)).convert('L')
            
            rect_x = center_x - crop_width // 2
            rect_y = center_y - crop_height // 2
            
            for angle in range(0, 360, angle_step):
                rotated_image = image.rotate(angle, resample=Image.BICUBIC)
                rotated_mask  = mask_center.rotate(angle)
                
                crop_image = rotated_image.crop((rect_x, rect_y, rect_x + crop_width, rect_y + crop_height))
                crop_mask  = rotated_mask.crop((rect_x, rect_y, rect_x + crop_width, rect_y + crop_height))
                
                crop_image_np = np.array(crop_image)
                zero_positions = np.argwhere(crop_image_np == 0)
                random_values = np.random.randint(27, 41, size=zero_positions.shape[0])
                crop_image_np[zero_positions[:, 0], zero_positions[:, 1]] = random_values

                crop_image = (Image.fromarray(crop_image_np)).convert('L')
                
                new_file_name = img_name + f'_{angle:03d}.png'
                
                crop_image.save(os.path.join(DIR_CROP_IMAGE, new_file_name))
                crop_mask.save(os.path.join(DIR_CROP_MASK, new_file_name))
    
if __name__ == "__main__":
    DIR_INPUT, DIR_OUTPUT, crop_width, crop_height, angle_step, output_type, rotate_x, rotate_y = get_args()
    main(DIR_INPUT, DIR_OUTPUT, crop_width, crop_height, angle_step, output_type, rotate_x, rotate_y)