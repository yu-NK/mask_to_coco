import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2

import argparse

def get_args():
    # 準備
    parser = argparse.ArgumentParser(
        description="Code to Visualize Dataset Masks(COCO Format)"
    )

    # 標準入力以外の場合
    parser = argparse.ArgumentParser()
    
    # データセットのパスを設定
    parser.add_argument("dir", type=str, help="The base directory path of the dataset.")
    
    # 出力先を設定
    parser.add_argument("-o", "--output-dir", dest="out", type=str, default='out', help="The output directory. Dafault is ./out")

    args = parser.parse_args()
    
    dir_base = args.dir
    dir_out  = args.out

    return dir_base, dir_out

def main(dir_base, dir_out):

    DIR_INPUT_JSON = os.path.join(dir_base, 'annotations.json')
    
    # COCOデータセットのJSONファイルを読み込む
    with open(DIR_INPUT_JSON, 'r') as f:
        coco_data = json.load(f)
    
    os.makedirs(dir_out, exist_ok=True)

    for image_info in tqdm(coco_data["images"]):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        if annotations:
            # 画像を読み込む
            image = Image.open(os.path.join(dir_base ,file_name))
            width, height = image.size

            # ランダムな色を生成（デフォルトは100）
            colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(100)])

            mask = Image.new('L', (width, height), 0)
            mask_col_data = np.zeros((height, width, 3), dtype = "u1")

            # 新しいアノテーションIDを初期化
            new_annotation_id = 1

            mask = Image.new('L', (width, height), 0)
            
            # アノテーションのセグメンテーションマスクを描画
            draw = ImageDraw.Draw(mask)
            
            for annotation in annotations:
                for seg in annotation["segmentation"]:

                    seg_temp = [(int(x), int(y)) for x, y in zip(seg[::2], seg[1::2])]
                    draw.polygon(seg_temp, fill= int(new_annotation_id))
                    
                    new_annotation_id += 1
                    
            mask_array = np.array(mask)

            mask_indices = mask_array > 0
            mask_col_data[mask_indices] = colors[mask_array[mask_indices]]

            cv2.imwrite(os.path.join(dir_out, os.path.splitext(os.path.basename(file_name))[0]+".png"), mask_col_data)
                
if __name__ == "__main__":
    dir_base, dir_out = get_args()
    main(dir_base, dir_out)