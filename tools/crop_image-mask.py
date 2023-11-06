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

sys.path.append("..")
from utils.tools import background_del, assign_cluster_number

def get_args():
    # 準備
    parser = argparse.ArgumentParser(
        description="Code to generate individual cropped images from directories of images and mask images (mask images color-coded for each object)"
    )

    # 標準入力以外の場合
    parser = argparse.ArgumentParser()
    
    # クロップ元の画像とマスク画像が含まれているディレクトリの指定
    parser.add_argument("input_dir", type=str, help="The base directory path of images and masks.")
    # 出力先のディレクトリの指定
    parser.add_argument("output_dir", type=str, help="Storage location for cropped images and masks")
    
    # クロップサイズを設定
    parser.add_argument("-x", dest="width", type=int, default=150, help="width of the cropped image. Default is 150.")
    parser.add_argument("-y", dest="height", type=int, default=150, help="height of the cropped image. Default is 150.")
    
    # 各画像におけるクロップ画像の生成回数
    parser.add_argument("-n", dest="num", type=int, default=10, help="Number of Crop Image Generations per Image. Default is 10.")
    
    # train or val or test
    parser.add_argument("-t", dest="type", type=str, default='train', help="train or val or test. Default is train.")

    args = parser.parse_args()

    # 引数から画像番号
    DIR_INPUT   = args.input_dir
    DIR_OUTPUT  = args.output_dir
    crop_width  = args.width
    crop_height = args.height
    crop_num    = args.num
    output_type = args.type

    return DIR_INPUT, DIR_OUTPUT, crop_width, crop_height, crop_num, output_type

def main(DIR_INPUT, DIR_OUTPUT, crop_width, crop_height, crop_num, output_type):

    img_files = glob.glob(os.path.join(DIR_INPUT, "images", output_type, "*"))
    mask_files = glob.glob(os.path.join(DIR_INPUT, "masks", output_type, "*"))
    
    img_files.sort()
    mask_files.sort()
        
    DIR_CROP_IMAGE = os.path.join(DIR_OUTPUT, "images", output_type)
    DIR_CROP_MASK  = os.path.join(DIR_OUTPUT, "masks", output_type)
    
    os.makedirs(DIR_CROP_IMAGE, exist_ok=True)
    os.makedirs(DIR_CROP_MASK , exist_ok=True)

    # 新しい画像番号を初期化
    new_image_id = 1

    with tqdm(range(len(img_files))) as pbar:
        for file_num in pbar:
        
            img_path  = img_files[file_num]
            mask_path = mask_files[file_num]

            img_name  = os.path.splitext(os.path.basename(img_path))[0]
            
            pbar.set_description(f"Process: {img_name}")

            # 画像を読み込む
            image = (Image.open(img_path)).convert('L')
            mask  = (Image.open(mask_path)).convert('RGB')
            width, height = image.size

            for i in range(crop_num):

                flag = True

                while(flag):
                    # ランダムな座標で切り出す
                    x1 = random.randint(0, width - crop_width)
                    y1 = random.randint(0, height - crop_height)
                    x2 = x1 + crop_width
                    y2 = y1 + crop_height

                    # 画像の切り出し
                    cropped_image = image.crop((x1, y1, x2, y2))
                    cropped_mask  = mask.crop((x1, y1, x2, y2))

                    # 前景画素がある場所をクロップするためのコード（データセットに応じて変更の必要あり） ##########
                    cropped_image_np = np.array(cropped_image)

                    cropped_image_NLMD = cv2.fastNlMeansDenoising(cropped_image_np, h=6)
                    cropped_image_th = np.where(cropped_image_NLMD > 100, 255, 0)

                    white_pixel_count = np.count_nonzero(cropped_image_th == 255)
                    if(white_pixel_count>1000): flag = False
                    #########################################################################

                new_file_name = img_name + f'_{i:03d}.png'
                cropped_image.save(os.path.join(DIR_CROP_IMAGE, new_file_name))

                cropped_mask_np = np.array(cropped_mask)
                seg, seg_bool = background_del(cropped_mask_np)
                cluster, cluster_color = assign_cluster_number(seg, seg_bool)

                # ランダムな色を生成（デフォルトは100）
                colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(100)])

                mask_data = np.zeros((crop_height, crop_width), dtype = "u2")

                # 新しいアノテーションIDを初期化
                new_annotation_id = 1

                for c in range(1,cluster.max()+1):
                    cluster_temp = (np.where(cluster == c, 255, 0)).astype("u1")

                    nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(cluster_temp, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

                    label_fix = np.zeros(nLabels-1)
                    sizes = data[:, 4]
                    label_fix = np.where(sizes < 15, 0, np.arange(0, nLabels))

                    valid_labels = np.logical_and(labelImages > 0, label_fix[labelImages.astype(int)] > 0)

                    if len(label_fix[labelImages[valid_labels]]):

                        fix_array = np.zeros((crop_width, crop_height),dtype="u1")
                        fix_array[valid_labels] = label_fix[labelImages[valid_labels]] + new_annotation_id - 1
                        mask_data = np.where(fix_array > 0, fix_array, mask_data)

                        new_annotation_id += label_fix[labelImages[valid_labels]].max()

                mask_col_data = np.zeros((crop_width, crop_height, 3), dtype = "u1")

                mask_indices = mask_data > 0
                mask_col_data[mask_indices] = colors[mask_data[mask_indices]]

                cv2.imwrite(os.path.join(DIR_CROP_MASK, new_file_name), mask_col_data)

                new_image_id += 1
                
if __name__ == "__main__":
    DIR_INPUT, DIR_OUTPUT, crop_width, crop_height, crop_num, output_type = get_args()
    main(DIR_INPUT, DIR_OUTPUT, crop_width, crop_height, crop_num, output_type)