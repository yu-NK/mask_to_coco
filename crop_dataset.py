import json
import os
import random
import numpy as np
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2

import argparse
from config.directory import DIR_DATASET, DIR_INPUT_JSON, DIR_OUTPUT

def get_args():
    # 準備
    parser = argparse.ArgumentParser(
        description="Crop an Existing Dataset (COCO Format) to Generate Mask Images"
    )

    # 標準入力以外の場合
    parser = argparse.ArgumentParser()
    
    # クロップサイズを設定
    parser.add_argument("-x", dest="width", type=int, default=150, help="width of the cropped image. Default is 150.")
    parser.add_argument("-y", dest="height", type=int, default=150, help="height of the cropped image. Default is 150.")
    
    # 各画像におけるクロップ画像の生成回数
    parser.add_argument("-n", dest="num", type=int, default=10, help="Number of Crop Image Generations per Image. Default is 10.")
    
    # train or val or test
    parser.add_argument("-t", dest="type", type=str, default='train', help="train or val or test. Default is train.")

    args = parser.parse_args()

    # 引数から画像番号
    crop_width  = args.width
    crop_height = args.height
    crop_num    = args.num
    output_type = args.type

    return crop_width, crop_height, crop_num, output_type

def main(crop_width, crop_height, crop_num, output_type):

    # COCOデータセットのJSONファイルを読み込む
    with open(DIR_INPUT_JSON, 'r') as f:
        coco_data = json.load(f)
        
    DIR_CROP_IMAGE = DIR_OUTPUT + "images/" + output_type
    DIR_CROP_MASK  = DIR_OUTPUT + "masks/"  + output_type
    
    os.makedirs(DIR_CROP_IMAGE, exist_ok=True)
    os.makedirs(DIR_CROP_MASK , exist_ok=True)

    # 新しい画像番号を初期化
    new_image_id = 1

    # 元のデータセットから画像をランダムに選択して切り出す
    for image_info in tqdm(coco_data["images"]):
        image_id = image_info["id"]
        file_name = image_info["file_name"]
        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]

        if annotations:
            # 画像を読み込む
            image = Image.open(os.path.join(DIR_DATASET ,file_name))
            width, height = image.size

            for i in range(crop_num):

                flag = True

                while(flag):
                    # ランダムな座標で切り出す
                    x1 = random.randint(0, width - crop_width)
                    y1 = random.randint(0, height - crop_height)
                    x2 = x1 + crop_width
                    y2 = y1 + crop_height

                    # 切り出し画像を保存
                    cropped_image = image.crop((x1, y1, x2, y2))

                    # 前景画素がある場所をクロップするためのコード（データセットに応じて変更の必要あり） ##########
                    val_array = np.array(cropped_image)
                    
                    val_NLMD = cv2.fastNlMeansDenoising(val_array, h=6)
                    th_img = np.where(val_NLMD > 50, 255, 0)

                    white_pixel_count = np.count_nonzero(th_img == 255)
                    if(white_pixel_count>1000): flag = False
                    #########################################################################

                # ランダムな色を生成（デフォルトは100）
                colors = np.array([[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(100)])

                new_file_name = f'{new_image_id:06d}.png'
                cropped_image.save(os.path.join(DIR_CROP_IMAGE, new_file_name))

                mask = Image.new('L', (width, height), 0)
                mask_data = np.zeros((crop_height, crop_width), dtype = "u1")

                # 新しいアノテーションIDを初期化
                new_annotation_id = 1

                # 対応するアノテーション情報も更新
                for annotation in annotations:
                    x, y, w, h = annotation["bbox"]

                    mask = Image.new('L', (width, height), 0)

                    # アノテーションのセグメンテーションマスクを描画
                    draw = ImageDraw.Draw(mask)
                    for seg in annotation["segmentation"]:
                        
                        # セグメンテーション座標を切り出し画像内に変換
                        seg_temp = [(int(x), int(y)) for x, y in zip(seg[::2], seg[1::2])]
                        draw.polygon(seg_temp, fill= int(new_annotation_id))

                    temp = np.array(mask)[y1:y2, x1:x2]

                    # アノテーションの面積を計算
                    annotation_area = np.sum(temp)

                    if annotation_area:

                        mask_binary = (np.where(temp==new_annotation_id, 255, 0)).astype("u1")
                        nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(mask_binary, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

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
    crop_width, crop_height, crop_num, output_type = get_args()
    main(crop_width, crop_height, crop_num, output_type)