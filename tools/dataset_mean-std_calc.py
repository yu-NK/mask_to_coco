"""
File Name: dataset_mean-std_calc.py
Description: 
    - This code is intended to calculate the mean and standard deviation of 
      images in a dataset. It was created based on code provided by callzhang 
      in an issue within the open-mmlab/mmdetection repository 
      (https://github.com/open-mmlab/mmdetection/issues/354). 
    - The correctness of this code has not been verified.

Usage:
    python dataset_mean-std_calc.py <img-dir>

    <img-dir>:
        Directory to store images of the dataset.

Dependencies:
    - Numerical Operations: numpy
    - Image Processing: PIL (Image, ImageDraw)
    - Utility: tqdm
    - Deep Learning Library: torch
"""

# Standard Libraries
import argparse
import glob
import os

# Numerical Operations
import numpy as np

# Image Processing
from PIL import Image, ImageFilter

# Deep Learning Library
from torch.utils.data import Dataset, DataLoader
from torch import Tensor

# Utility
from tqdm import tqdm

def parse_args():

    parser = argparse.ArgumentParser(
        description="code to calculate the mean and standard deviation of a dataset."
    )
    
    parser.add_argument("img-dir", type=str, help="Directory to store images of the dataset.")
    
    args = parser.parse_args()

    return args

class MyDataset(Dataset):
    def __init__(self, image_list):
        assert isinstance(image_list, list)
        self.image_list = image_list
    
    def __getitem__(self, index):
        path = self.image_list[index]
        image = Image.open(path).convert("RGB")
        img_arr = np.asarray(image, dtype=np.float32)
        assert img_arr.dtype == np.float32
        return img_arr

    def __len__(self):
        return len(self.image_list)

def main():

    args = parse_args()

    temp = glob.glob(os.path.join(args.img_dir, "*"))
    dataset = MyDataset(temp)
    loader = DataLoader(
        dataset,
        batch_size=100,
        num_workers=1,
        shuffle=False
    )

    mean = Tensor([0,0,0])
    std = Tensor([0,0,0])
    n_samples= 0

    for data in tqdm(loader):
        batch_samples = data.size(0)
        data2 = data.view(-1, 3)
        mean += data2.mean(0)
        std += data2.std(0)
        n_samples += 1

    mean /= n_samples
    std /= n_samples

    print("mean:", mean)
    print("std:", std)
    
if __name__=='__main__':
    main()