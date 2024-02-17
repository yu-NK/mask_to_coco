# The original code in this file is quoted from RandAugment/augmentations.py in pytorch-randaugment by ildoonet and is provided under the MIT License.

# Copyright (c) 2019 Ildoo Kim
# Released under the MIT license
# https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py

# I have modified the code to perform similar data augmentation on color-segmented mask images and original images for instance segmentation. There are parts that are specialized for specific tasks.

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from PIL import Image

import argparse
import glob
import os

from tqdm import tqdm

def get_args():
    # 準備
    parser = argparse.ArgumentParser(
        description="RandAugment: images and masks"
    )

    # 標準入力以外の場合
    parser = argparse.ArgumentParser()
    
    # クロップサイズを設定
    parser.add_argument("input_dir", type=str,  help="Directory containing images and masks for data augmentation.")
    parser.add_argument("-o", "--output-dir", dest="output_dir", default="./out", type=str, help="Output directory of images and masks after data augmentation. Default is ./out")
    parser.add_argument("-n", "--pram-n", dest="N", type=int, default=4,  help="The parameter 'n' in RandAugment. Dafault is 4.")
    parser.add_argument("-m", "--pram-m", dest="M", type=int, default=10, help="The parameter 'm' in RandAugment. Dafault is 10")
    parser.add_argument("-a", "--aug-num", dest="num", type=int, default=100, help="Number of augmentations. Default is 100.")
    
    args = parser.parse_args()

    # 引数から画像番号
    dir_input  = args.input_dir
    dir_output = args.output_dir
    n = args.N
    m = args.M
    num = args.num
    
    return dir_input, dir_output, n, m, num

def ShearX(img, mask, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.Transform.AFFINE, (1, v, 0, 0, 1, 0), resample=Image.Resampling.BICUBIC), mask.transform(img.size, PIL.Image.Transform.AFFINE, (1, v, 0, 0, 1, 0))


def ShearY(img, mask, v):  # [-0.3, 0.3]
    assert -0.3 <= v <= 0.3
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, 0, v, 1, 0), resample=Image.Resampling.BICUBIC), mask.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, 0, v, 1, 0))


def TranslateX(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, v, 0, 1, 0)), mask.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateXabs(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, v, 0, 1, 0)), mask.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, v, 0, 1, 0))


def TranslateY(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert -0.45 <= v <= 0.45
    if random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, 0, 0, 1, v)), mask.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, 0, 0, 1, v))


def TranslateYabs(img, mask, v):  # [-150, 150] => percentage: [-0.45, 0.45]
    assert 0 <= v
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, 0, 0, 1, v)), mask.transform(img.size, PIL.Image.Transform.AFFINE, (1, 0, 0, 0, 1, v))


def Rotate(img, mask, v):  # [-30, 30]
    assert -30 <= v <= 30
    if random.random() > 0.5:
        v = -v
    return img.rotate(v, resample=Image.Resampling.BICUBIC), mask.rotate(v)


def AutoContrast(img, mask, _):
    return PIL.ImageOps.autocontrast(img), mask


def Invert(img, mask, _):
    return PIL.ImageOps.invert(img), mask


def Equalize(img, mask, _):
    return PIL.ImageOps.equalize(img), mask


def Flip(img, mask, _):  # not from the paper
    return PIL.ImageOps.mirror(img), PIL.ImageOps.mirror(mask)

def Solarize(img, mask, v):  # [0, 256]
    assert 0 <= v <= 256
    return PIL.ImageOps.solarize(img, v), mask


def SolarizeAdd(img, mask, addition=0, threshold=128):
    img_np = np.array(img).astype(np.int)
    img_np = img_np + addition
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold), mask


def Posterize(img, mask, v):  # [4, 8]
    v = int(v)
    v = max(1, v)
    return PIL.ImageOps.posterize(img, v), mask


def Contrast(img, mask, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Contrast(img).enhance(v), mask


def Color(img, mask, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Color(img).enhance(v), mask


def Brightness(img, mask, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Brightness(img).enhance(v), mask


def Sharpness(img, mask, v):  # [0.1,1.9]
    assert 0.1 <= v <= 1.9
    return PIL.ImageEnhance.Sharpness(img).enhance(v), mask


def Cutout(img, mask, v):  # [0, 60] => percentage: [0, 0.2]
    assert 0.0 <= v <= 0.2
    if v <= 0.:
        return img

    v = v * img.size[0]
    return CutoutAbs(img, mask, v)


def CutoutAbs(img, mask, v):  # [0, 60] => percentage: [0, 0.2]
    # assert 0 <= v <= 20
    if v < 0:
        return img
    w, h = img.size
    x0 = np.random.uniform(w)
    y0 = np.random.uniform(h)

    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = min(w, x0 + v)
    y1 = min(h, y0 + v)

    xy = (x0, y0, x1, y1)
    # CT画像のためグレースケール用を用意
    color_gray = (32,)
    # 基本はcolor_rgbを使う
    color_rgb  = (0, 0, 0)
    
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color_gray)
    mask = mask.copy()
    PIL.ImageDraw.Draw(mask).rectangle(xy, color_rgb)
    
    return img, mask


def SamplePairing(imgs):  # [0, 0.4]
    def f(img1, v):
        i = np.random.choice(len(imgs))
        img2 = PIL.Image.fromarray(imgs[i])
        return PIL.Image.blend(img1, img2, v)

    return f


def Identity(img, mask, v):
    return img, mask


def augment_list():  # 16 oeprations and their ranges
    # https://github.com/google-research/uda/blob/master/image/randaugment/policies.py#L57
    # l = [
    #     (Identity, 0., 1.0),
    #     (ShearX, 0., 0.3),  # 0
    #     (ShearY, 0., 0.3),  # 1
    #     (TranslateX, 0., 0.33),  # 2
    #     (TranslateY, 0., 0.33),  # 3
    #     (Rotate, 0, 30),  # 4
    #     (AutoContrast, 0, 1),  # 5
    #     (Invert, 0, 1),  # 6
    #     (Equalize, 0, 1),  # 7
    #     (Solarize, 0, 110),  # 8
    #     (Posterize, 4, 8),  # 9
    #     # (Contrast, 0.1, 1.9),  # 10
    #     (Color, 0.1, 1.9),  # 11
    #     (Brightness, 0.1, 1.9),  # 12
    #     (Sharpness, 0.1, 1.9),  # 13
    #     # (Cutout, 0, 0.2),  # 14
    #     # (SamplePairing(imgs), 0, 0.4),  # 15
    # ]

    # https://github.com/tensorflow/tpu/blob/8462d083dd89489a79e3200bcc8d4063bf362186/models/official/efficientnet/autoaugment.py#L505
    l = [
        #(AutoContrast, 0, 1),
        #(Equalize, 0, 1),
        #(Invert, 0, 1),
        (Rotate, 0, 30),
        #(Posterize, 0, 4),
        #(Solarize, 0, 256),
        #(SolarizeAdd, 0, 110),
        #(Color, 0.1, 1.9),
        #(Contrast, 0.1, 1.9),
        #(Brightness, 0.1, 1.9),
        #(Sharpness, 0.1, 1.9),
        #(ShearX, 0., 0.3),
        #(ShearY, 0., 0.3),
        #(CutoutAbs, 0, 40),
        (TranslateXabs, 0., 100),
        (TranslateYabs, 0., 100),
        (Flip, 0, 1)
    ]

    return l

"""
# これを使わなければPyTorchの環境は不要
class Lighting(object):
    #Lighting noise(AlexNet - style PCA - based noise)

    def __init__(self, alphastd, eigval, eigvec):
        self.alphastd = alphastd
        self.eigval = torch.Tensor(eigval)
        self.eigvec = torch.Tensor(eigvec)

    def __call__(self, img):
        if self.alphastd == 0:
            return img

        alpha = img.new().resize_(3).normal_(0, self.alphastd)
        rgb = self.eigvec.type_as(img).clone() \
            .mul(alpha.view(1, 3).expand(3, 3)) \
            .mul(self.eigval.view(1, 3).expand(3, 3)) \
            .sum(1).squeeze()

        return img.add(rgb.view(3, 1, 1).expand_as(img))


class CutoutDefault(object):

    #Reference : https://github.com/quark0/darts/blob/master/cnn/utils.py
    
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img
        
        
class RandAugment:
    def __init__(self, n, m):
        self.n = n
        self.m = m      # [0, 30]
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (float(self.m) / 30) * float(maxval - minval) + minval
            img = op(img, val)

        return img
"""
    
    
def main(dir_input, dir_output, n, m, num):
    
    DIR_OUTPUT_IMAGE = os.path.join(dir_output, "images/train")
    DIR_OUTPUT_MASK  = os.path.join(dir_output, "masks/train")
    
    os.makedirs(DIR_OUTPUT_IMAGE, exist_ok=True)
    os.makedirs(DIR_OUTPUT_MASK , exist_ok=True)
    
    img_list  = sorted(glob.glob(os.path.join(dir_input, "images/train/*")))
    mask_list = sorted(glob.glob(os.path.join(dir_input, "masks/train/*")))
    
    for img_path, mask_path in tqdm(zip(img_list, mask_list), total=len(img_list), desc="Processing"):
        img_name  = os.path.splitext(os.path.basename(img_path))[0]
        mask_name = os.path.splitext(os.path.basename(mask_path))[0]
        
        for i in tqdm(range(num), total=num, desc=f'Augmentation: {img_name}'):
            #　CT画像はグレースケールのためグレースケールで読み込み（その他の画像の場合は変更する必要あり）
            img  = (Image.open(img_path)).convert('L')
            mask = (Image.open(mask_path)).convert('RGB')
            
            ops = random.choices(augment_list(), k=n)
            for op, minval, maxval in ops:
                val = (float(m) / 30) * float(maxval - minval) + minval
                img, mask = op(img, mask, val)
                
            # Pillow -> Numpy
            img_np  = np.array(img)
            mask_np = np.array(mask)
            
            # アフィン変換などによりできた余白をランダムに輝度値の設定（CT画像に合うように）
            zero_positions = np.argwhere(img_np == 0)
            random_values = np.random.randint(27, 41, size=zero_positions.shape[0])
            img_np[zero_positions[:, 0], zero_positions[:, 1]] = random_values
            
            mask_indices = (mask_np == (0, 0, 0)).all(axis=-1)
            mask_np[mask_indices] = (0,0,0)
            
            img_pil  = Image.fromarray(img_np)
            mask_pil = Image.fromarray(mask_np)
            
            img_pil.save(os.path.join(dir_output, f'images/train/{img_name}_aug{i:03}.png'))
            mask_pil.save(os.path.join(dir_output, f'masks/train/{img_name}_aug{i:03}.png'))
        
if __name__ == "__main__":
    dir_input, dir_output, n, m, num = get_args()
    main(dir_input, dir_output, n, m, num)
