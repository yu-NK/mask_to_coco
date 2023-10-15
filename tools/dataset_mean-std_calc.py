from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from tqdm.notebook import tqdm
from PIL import Image, ImageFilter
import numpy as np

import argparse
import glob
import os

def get_args():
    # 準備
    parser = argparse.ArgumentParser(
        description="code to calculate the mean and standard deviation of a dataset."
    )

    # 標準入力以外の場合
    parser = argparse.ArgumentParser()
    
    # クロップサイズを設定
    parser.add_argument("dir", type=str, help="Directory to store images of the dataset.")
    
    args = parser.parse_args()

    # 引数から画像番号
    dir_image   = args.dir

    return dir_image

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

def main(dir_image):
    temp = glob.glob(os.path.join(dir_image, "*"))
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
    dir_image = get_args()
    main(dir_image)