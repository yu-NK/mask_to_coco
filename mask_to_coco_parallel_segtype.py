import json
import collections as cl
import numpy as np

import cv2
import glob
import sys
import os
from tqdm import tqdm

import argparse
# config/coco_config.pyで指定
from config.coco_config import info, licenses, images, categories

import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor
import pycocotools.mask as mask

def get_args():
    # 準備
    parser = argparse.ArgumentParser(
        description="Create a dataset in COCO format from a mask image for instance segmentation (single class) with parallel processing. The format of segmentation can be chosen between RLE (Run-Length Encoding) or Polygon.　**Installation of pycocotools is required**"
    )

    # 標準入力以外の場合
    parser = argparse.ArgumentParser()
    
    # COCOフォーマットに変換する画像が含まれているディレクトリの指定
    parser.add_argument("dir", type=str, help="The base directory path of the dataset.")
    
    # train or val or test
    parser.add_argument("-t", "--type", dest="type", type=str, default='train', help="train or val or test. Default is train.")

    # RLE or Polygon
    parser.add_argument("-f", "--format", dest="format", type=int, default=0, help="Selection between RLE format or Polygon format (0: RLE format, 1: Polygon format). The default is 0: RLE format.")
    
    # jsonファイル名の指定
    parser.add_argument("-n", "--name", dest="name", type=str, default='auto', help="Specify the JSON file name. The default is '[TYPE]_annotations.json'.")
    
    # jsonファイル名の指定
    parser.add_argument("-c", "--core", dest="core", type=int, default=multiprocessing.cpu_count(), help="The number of cores to be used for parallelization. The default is the number of physical cores in the system.")
    num_processes = multiprocessing.cpu_count()
    
    args = parser.parse_args()
    
    dir_base   = args.dir
    input_type = args.type
    format = args.format
    json_name  = args.name
    
    if(args.core > multiprocessing.cpu_count()):
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = args.core

    return dir_base, input_type, format, json_name, num_processes

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
    # OpenCVを使って画像を読み込む
    image = cv2.imread(image_path)
    
    # ユニークな色の集合を作成
    unique_colors, unique_indices = np.unique(image.reshape(-1, 3), axis=0, return_inverse=True)
    
    # ユニークな色のインデックスを使ってマスク配列を作成
    mask_array = unique_indices.reshape(image.shape[:2])
    
    return mask_array #, {tuple(color): idx for idx, color in enumerate(unique_colors)}

def find_coordinates(binary_image):
    # 画像内の対象ピクセル（例えば、値が1のピクセル）の座標を見つける
    coordinates = np.argwhere(binary_image > 0)
    
    # 左上（最小）と右下（最大）の座標を計算する
    top_left = coordinates.min(axis=0)
    bottom_right = coordinates.max(axis=0)
    
    # 座標を抽出
    top_left_y, top_left_x = top_left
    bottom_right_y, bottom_right_x = bottom_right

    # 幅と高さを計算
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
        
        # 分割されたマスクに対する処理
        nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(cluster_temp, 8, cv2.CV_16U, cv2.CCL_DEFAULT)
        #label_fix = np.zeros(nLabels-1)
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
                if(format == 0):
                    coco_rle = mask.encode(np.asfortranarray(seg))
                    coco_rle["counts"] = coco_rle["counts"].decode('utf-8')
                    tmp["segmentation"] = coco_rle
                # Polygon
                elif(format == 1):
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

def main(dir_base, input_type, format, json_name, num_processes):
    img_path = os.path.join(dir_base, "images", input_type)
    mask_path = os.path.join(dir_base, "masks", input_type)
    if json_name == "auto":
        json_path = os.path.join(dir_base, input_type + "_annotations.json")
    else:
        json_path = os.path.join(dir_base, json_name)
        
    mask_files = glob.glob(os.path.join(mask_path, "*"))
    mask_files.sort()
    
    print("Start processing. CPU core: ", num_processes)
    
    args = []
    for i in range(num_processes):
        start = i * len(mask_files) // num_processes
        end = (i + 1) * len(mask_files) // num_processes
        args.append(np.array([start, end, mask_files, i, format],dtype=object))
    
    with ProcessPoolExecutor() as executor:
        results = executor.map(process_images, args)
        
    ann_result = []
    for result in results:
        ann_result.extend(result)

    print("Begin writing the JSON file.")
    # 各プロセスの結果を結合してJSONファイルに保存
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
    dir_base, input_type, format, json_name, num_processes = get_args()
    main(dir_base, input_type, format, json_name, num_processes)