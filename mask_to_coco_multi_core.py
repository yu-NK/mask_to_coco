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
from utils.tools import background_del, assign_cluster_number

import multiprocessing
from concurrent.futures.process import ProcessPoolExecutor

def get_args():
    # 準備
    parser = argparse.ArgumentParser(
        description="Creating a COCO-format dataset from mask images for instance segmentation."
    )

    # 標準入力以外の場合
    parser = argparse.ArgumentParser()
    
    # COCOフォーマットに変換する画像が含まれているディレクトリの指定
    parser.add_argument("dir", type=str, help="The base directory path of the dataset.")
    
    # train or val or test
    parser.add_argument("-t", "--type", dest="type", type=str, default='train', help="train or val or test. Default is train.")
    
    # jsonファイル名の指定
    parser.add_argument("-n", "--name", dest="name", type=str, default='auto', help="Specify the JSON file name. The default is '[TYPE]_annotations.json'.")
    
    # jsonファイル名の指定
    parser.add_argument("-c", "--core", dest="core", type=int, default=multiprocessing.cpu_count(), help="The number of cores to be used for parallelization. The default is the number of physical cores in the system.")
    num_processes = multiprocessing.cpu_count()
    
    args = parser.parse_args()
    
    dir_base   = args.dir
    input_type = args.type
    json_name  = args.name
    
    if(args.core > multiprocessing.cpu_count()):
        num_processes = multiprocessing.cpu_count()
    else:
        num_processes = args.core

    return dir_base, input_type, json_name, num_processes

def annotations(i, file):
    
    tmps = []
    
    annotation_id = 1
    
    img = cv2.imread(file)

    seg, seg_bool = background_del(img)
    cluster, cluster_color = assign_cluster_number(seg, seg_bool)
    
    annotation_id = 1

    for c in range(1,cluster.max()+1):
        cluster_temp = (np.where(cluster == c, 255, 0)).astype("u1")

        # 分割されたマスクに対する処理
        nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(cluster_temp, 8, cv2.CV_16U, cv2.CCL_DEFAULT)
        #label_fix = np.zeros(nLabels-1)
        sizes = data[:, 4]
        label_fix = np.where(sizes < 15, 0, np.arange(0, nLabels))

        valid_labels = np.logical_and(labelImages > 0, label_fix[labelImages.astype(int)] > 0)

        fix_array = np.zeros((img.shape[0], img.shape[1]),dtype="u1")
        fix_array[valid_labels] = label_fix[labelImages[valid_labels]]

        for ann_label in range(1,fix_array.max() + 1):

            tmp = cl.OrderedDict()

            seg = np.zeros((img.shape[0], img.shape[1]),dtype="u1")
            mask_ind = np.argwhere(fix_array == ann_label)
            seg[mask_ind[:, 0], mask_ind[:, 1]] = 255

            # 輪郭抽出(CHAIN_APPROX_TC89_L1: 輪郭の座標を直線で近似できる部分の輪郭の点を省略)
            contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            segmentation_list = []

            # point[0] -> [X, Y]
            for contour in contours:
                for point in contour:
                    segmentation_list.append(int(point[0, 0]))
                    segmentation_list.append(int(point[0, 1]))

            nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(seg, 8, cv2.CV_16U, cv2.CCL_DEFAULT)
            if nLabels>1:
                x, y, width, height, area = data[1]
                
                tmp["id"] = i * 10000 + annotation_id
                tmp["image_id"] = i + 1
                tmp["category_id"] = 1
                tmp["segmentation"] = [segmentation_list]
                tmp["area"] = float(area)
                tmp["bbox"] =  [float(x), float(y), float(width), float(height)]
                tmp["iscrowd"] = 0

                if(len(segmentation_list)>4):
                    tmps.append(tmp)
                    annotation_id += 1
    
    return tmps

def process_images(args):
    start, end, mask_file, process_num = args
    results = []
    with tqdm(range(start, end)) as pbar:
        for i in pbar:
            pbar.set_description(f"[Process {process_num:02}] {i:04}")
        
            result = annotations(i, mask_file[i])
            results.extend(result)
        
    return results

def main(dir_base, input_type, json_name, num_processes):
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
        args.append(np.array([start, end, mask_files, i],dtype=object))
    
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
    json.dump(js, fw, indent=2)
    
    print("Finished writing JSON file.")

if __name__=='__main__':
    dir_base, input_type, json_name, num_processes = get_args()
    main(dir_base, input_type, json_name, num_processes)