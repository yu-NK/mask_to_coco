import json
import collections as cl
import numpy as np

import cv2
import glob
import sys
import os
from tqdm import tqdm

import argparse
from config.coco_config import info, licenses, images, categories
from utils.tools import background_del, assign_cluster_number

def get_args():
    # 準備
    parser = argparse.ArgumentParser(
        description="Creating a COCO-format dataset from mask images for instance segmentation."
    )

    # 標準入力以外の場合
    parser = argparse.ArgumentParser()
    
    # クロップサイズを設定
    parser.add_argument("dir", type=str, help="The base directory path of the dataset.")
    
    # train or val or test
    parser.add_argument("-t", "--type", dest="type", type=str, default='train', help="train or val or test. Default is train.")
    
    # jsonファイル名の指定
    parser.add_argument("-n", "--name", dest="name", type=str, default='auto', help="Specify the JSON file name. The default is '[TYPE]_annotations.json'.")
    
    args = parser.parse_args()

    # 引数から画像番号
    dir_base   = args.dir
    input_type = args.type
    json_name  = args.name

    return dir_base, input_type, json_name

def annotations(mask_path):
    tmps = []
    
    mask_files = glob.glob(os.path.join(mask_path, "*"))
    mask_files.sort()
    
    print(mask_files)

    annotation_id = 1
    
    for i, file in tqdm(enumerate(mask_files)):
        img = cv2.imread(file)

        seg, seg_bool = background_del(img)
        cluster, cluster_color = assign_cluster_number(seg, seg_bool)

        for c in range(1,cluster.max()+1):
            tmp = cl.OrderedDict()

            cluster_temp = (np.where(cluster == c, 255, 0)).astype("u1")
        
            # 輪郭抽出(CHAIN_APPROX_TC89_L1: 輪郭の座標を直線で近似できる部分の輪郭の点を省略)
            contours, _ = cv2.findContours(cluster_temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_L1)
            segmentation_list = []

            # point[0] -> [X, Y]
            for contour in contours:
                for point in contour:
                    segmentation_list.append(int(point[0, 0]))
                    segmentation_list.append(int(point[0, 1]))

            nLabels, labelImages, data, center = cv2.connectedComponentsWithStatsWithAlgorithm(cluster_temp, 8, cv2.CV_16U, cv2.CCL_DEFAULT)

            label_fix = np.zeros(nLabels-1)
            x, y, width, height, area = data[1]
    
            tmp["segmentation"] = [segmentation_list]
            tmp["id"] = str(annotation_id)
            tmp["image_id"] = i + 1
            tmp["category_id"] = 1
            tmp["area"] = float(area)
            tmp["iscrowd"] = 0
            tmp["bbox"] =  [float(x), float(y), float(width), float(height)]
            
            if(len(segmentation_list)>4):
                tmps.append(tmp)
                annotation_id += 1
    
    return tmps

def main(dir_base, input_type, json_name):

    img_path = os.path.join(dir_base, "images", input_type)
    mask_path = os.path.join(dir_base, "masks", input_type)
    if(json_name == "auto"):
        json_path = os.path.join(dir_base, input_type + "_annotations.json")
    else:
        json_path = os.path.join(dir_base, json_name)
    
    query_list = ["info", "licenses", "images", "annotations", "categories", "segment_info"]
    js = {
        "info": [],
        "licenses": [],
        "images":[],
        "annotations":[],
        "categories":[],
        "segment_info":[]
    }
    
    for i in range(len(query_list)):
        tmp = []
        # Info
        if query_list[i] == "info":
            tmp = info()
    
        # licenses
        elif query_list[i] == "licenses":
            tmp = licenses()
    
        elif query_list[i] == "images":
            tmp = images(img_path)
    
        elif query_list[i] == "annotations":
            tmp = annotations(mask_path)
    
        elif query_list[i] == "categories":
            tmp = categories()

        # save it
        js[query_list[i]] = tmp
    
    # write
    fw = open(json_path,'w')
    json.dump(js,fw,indent=2)

if __name__=='__main__':
    dir_base, input_type, json_name = get_args()
    main(dir_base, input_type, json_name)
