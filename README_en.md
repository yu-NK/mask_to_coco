# mask_to_coco

[日本語版READMEはこちら](/README.md)

> [!WARNING]
> This README has been translated from the Japanese version by ChatGPT. Please be cautious as the translation may contain inaccuracies.

## Overview
This repository is designed to generate COCO format datasets (JSON files) from mask images annotated for single-class instance segmentation. In these mask images, each object within the image, excluding the background, is marked with a mask of a distinct color.

## Table of Contents
- [Directory Structure](#directory-structure)
- [Usage](#usage)
  - [Creating a COCO Format Dataset](#creating-a-coco-format-dataset)
  - [Visualization](#visualization-visualization)
  - [Tools](#tools-tools)
- [References Sites](#reference-sites)

## Directory Structure
```
mask_to_coco
├── config                  - Stores settings for directories and the COCO format.
│   ├── __init__.py
│   ├── coco_config.py
│   └── directory.py
├── tools                   - Contains additional codes related to the dataset.
│   ├── RandAugment.py
│   ├── crop_dataset.py
│   ├── crop_image-mask.py
│   ├── crop_image-mask_resize.py
│   ├── crop_rotate.py
│   └── dataset_mean-std_calc.py
├── utils                   - Contains necessary functions for the code.
│   ├── __init__.py
│   └── tools.py
├── visualization           - Holds code for dataset visualization.
│   ├── COCO_Image_Viewer.ipynb
│   └── visualization_mask.py
├── README.md
├── mask_to_coco.py
├── mask_to_coco_parallel_polygon.py
└── mask_to_coco_parallel_segtype.py    - Recommended for use with this code
```

## Usage

> [!IMPORTANT]
> - Some codes contain specific processes to accommodate flower CT images, which may need adjustments for handling other data types.
> - Installation of `pycocotools` is required to use `mask_to_coco_parallel_segtype.py`.

### Creating a COCO Format Dataset
This section explains the code for generating a COCO format dataset (JSON file) from mask images annotated for instance segmentation of a single class. **While there are three types of scripts available, it is generally recommended to use `./mask_to_coco_parallel_segtype.py`.**

- [`mask_to_coco.py`](#mask_to_cocopy)
- [`mask_to_coco_parallel_polygon.py`](#mask_to_coco_parallel_polygonpy)
- [`mask_to_coco_parallel_segtype.py` **(Recommended)**](#mask_to_coco_parallel_segtypepy-recommended)

You can generate a JSON file by setting up the dataset in the following directory structure. Using the code explained here, `[TYPE]_annotation.json` will be generated under `[Dataset Name]/`. `[TYPE]` corresponds to the specified dataset type (`train`, `val`, `test`).

```
[Dataset Name]
├── images
│   ├── train
│   │   ├── aaa.png
│   │   ├── bbb.png
│   │   └── ccc.png
│   ├── val
│   │   ├── www.png
│   │   └── xxx.png
│   └── test
│       ├── yyy.png
│       └── zzz.png
└── masks
    ├── train
    │   ├── aaa.png
    │   ├── bbb.png
    │   └── ccc.png
    ├── val
    │   ├── www.png
    │   └── xxx.png
    └── test
        ├── yyy.png
        └── zzz.png
```

> [!NOTE]
> - It is recommended that the image extension in `images/` be **PNG or TIF**.
> - All images in `masks/` must be stored as **PNG** and have the same name as the images in `images/`.
> - It is necessary to edit **info, licenses, categories** in `./config/coco_config.py` as required for the COCO format.

#### `./mask_to_coco.py`
Code to create a COCO format dataset from mask images annotated for instance segmentation (excluding the background, one class) with each object colored differently.

```bash
python mask_to_coco.py [-h] [-t TYPE] [-n NAME] dataset

# positional arguments:
#   dataset               The base directory path of the dataset.

# optional arguments:
#   -h, --help   show this help message and exit
#   --type TYPE  train, val, or test. Default is train.
#   --name NAME  Specify the JSON file name. The default is '[TYPE]_annotations.json'.
```

#### `./mask_to_coco_parallel_polygon.py`
Creates a COCO format dataset in parallel processing from mask images annotated for instance segmentation (excluding the background, one class) with each object colored differently.

> [!NOTE]
> If the same color mask is separated due to cropping or other processing, each is treated as a different object.

```bash
python mask_to_coco_parallel_polygon.py [-h] [--type TYPE] [--name NAME] [--proc-num PROC_NUM]

# positional arguments:
#   dataset               The base directory path of the dataset.

# optional arguments:
#   -h, --help           show this help message and exit
#   --type TYPE          The dataset type: train, val, or test. Defaults to 'train'.
#   --name NAME          The JSON file name for COCO annotations. Defaults to '[TYPE]_annotations.json'.
#   --proc-num PROC_NUM  The number of cores used for parallelization. Defaults to the system's physical core count.
```

#### `./mask_to_coco_parallel_segtype.py` **(Recommended)**
Creates a COCO format dataset in parallel processing from mask images annotated for instance segmentation (excluding the background, one class) with each object colored differently. The format of the segmentation can be chosen between RLE (Run-Length Encoding) or Polygon.

> [!NOTE]
> - Installation of pycocotools is required
>   ```bash
>   pip install pycocotools
>   ```
> - If the same color mask is separated due to cropping or other processing, each is treated as a different object.

```bash
python mask_to_coco_parallel_segtype.py [-h] [--type TYPE] [--ply] [--name NAME] [--proc-num PROC_NUM] dataset

# positional arguments:
#   dataset              The base directory path for the dataset.

# options:
#   -h, --help           show this help message and exit
#   --type TYPE          The dataset type: train, val, or test. Defaults to 'train'.
#   --ply                Outputs in polygon format if selected. Defaults to RLE format.
#   --name NAME          The JSON file name for COCO annotations. Defaults to '[TYPE]_annotations.json'.
#   --proc-num PROC_NUM  The number of cores used for parallelization. Defaults to the system's physical core count.
```

### Visualization (`visualization/`)
This section describes the code for visualizing the dataset.
#### `./visualization/visualization_mask.py`
Code for visualizing a dataset in COCO format.

```bash
python visualization_mask.py [-h] [--type TYPE] [--output OUTPUT] dataset

# positional arguments:
#   dataset          The base directory path of the dataset.

# options:
#   -h, --help       show this help message and exit
#   --type TYPE      The dataset type: train, val, or test. Defaults to 'train'.
#   --output OUTPUT  The output directory. Default is ./out
```

It is necessary to follow the directory structure explained in [Creating a COCO Format Dataset](#creating-a-coco-format-dataset) (excluding `masks/`). If no output is specified, it will be output under `./out`.

### Tools (`tools/`)
This section describes useful codes for creating datasets. The `dataset` directory mentioned here should follow the directory structure explained in [Creating a COCO Format Dataset](#creating-a-coco-format-dataset) (which may exclude `masks/`).

#### `./tools/crop_dataset.py`
Code to randomly crop images from an existing COCO format dataset and generate a new dataset. This does not extend to the creation of JSON files. **Specific processing is included for compatibility with flower CT images, and modifications are necessary when dealing with other data.**

<img src="/assets/crop_dataset_en.jpg" width="90%" />

```bash
python crop_dataset.py [-h] [-x WIDTH] [-y HEIGHT] [-n NUM] [-t TYPE]

# positional arguments:
#   dataset               The base directory path of the dataset.

# options:
#   -h, --help            show this help message and exit
#   -t TYPE, --type TYPE  The dataset type: train, val, or test. Defaults to 'train'.
#   -x CROP_WIDTH, --crop-width CROP_WIDTH
#                         width of the cropped image. Default is 150.
#   -y CROP_HEIGHT, --crop-height CROP_HEIGHT
#                         height of the cropped image. Default is 150.
#   -n CROP_NUM, --crop-num CROP_NUM
#                         Number of Crop Image Generations per Image. Default is 10.
#   -o DIR_OUT, --output DIR_OUT
#                         The output directory. Default is ./out/crop_dataset
```

If not specified, the output will be stored under `./out/crop_dataset/`.

#### `./tools/crop_image-mask.py`
Code to randomly generate cropped images from directories of images and mask images (mask images colored differently for each object). **Specific processing is included for compatibility with flower CT images, and modifications are necessary when dealing with other data.**

<img src="/assets/crop_image-mask_en.jpg" width="75%" />

```bash
python crop_image-mask.py [-h] [--type TYPE] [--crop-n CROP_NUM] [--crop-w CROP_WIDTH] [--crop-h CROP_HEIGHT] dataset output

# positional arguments:
#   dataset               The base directory path of the dataset.
#   output                Storage location for cropped images and masks

# options:
#   -h, --help            show this help message and exit
#   --type TYPE           The dataset type: train, val, or test. Defaults to 'train'.
#   --crop-n CROP_NUM     Number of Crop Image Generations per Image. Default is 10.
#   --crop-w CROP_WIDTH   width of the cropped image. Default is 150.
#   --crop-h CROP_HEIGHT  height of the cropped image. Default is 150.
```

It is generally assumed that the directory structure is that of `mask_to_coco.py`.

#### `./tools/crop_image-mask_resize.py`
Code to randomly generate cropped images from directories of images and mask images (each object is colored differently in mask images). The generated cropped images are resized to a specified image size. **Specific processing for CT images of flowers is included, requiring modifications when dealing with other data.**

<img src="/assets/crop_image-mask_resize_en.jpg" width="90%" />

```bash
python crop-resize.py [-h] [--type TYPE] [--crop-n CROP_NUM] [--crop-w CROP_WIDTH] [--crop-h CROP_HEIGHT] [--resize-w RESIZE_WIDTH] [--resize-h RESIZE_HEIGHT] dataset output

# positional arguments:
#   dataset               The base directory path of the dataset.
#   output                Storage location for cropped images and masks

# options:
#   -h, --help            show this help message and exit
#   --type TYPE           The dataset type: train, val, or test. Defaults to 'train'.
#   --crop-n CROP_NUM     Number of Crop Image Generations per Image. Default is 10.
#   --crop-w CROP_WIDTH   width of the cropped image. Default is 150.
#   --crop-h CROP_HEIGHT  height of the cropped image. Default is 150.
#   --resize-w RESIZE_WIDTH
#                         width after resizing. Default is 150.
#   --resize-h RESIZE_HEIGHT
#                         height after resizing. Default is 150.
```

#### `./tools/crop_rotate.py`
Code to generate cropped images from directories of images and mask images (each object is colored differently in mask images). This code crops the image like a horizontally elongated rectangle while rotating it. As a preprocessing step, it translates so that the specified rotation center coordinates become the center of the image. **Specific processing for CT images of flowers is included, requiring modifications when dealing with other data.**

<img src="/assets/crop_rotate_en.jpg" width="90%" />

```bash
python crop_rotate.py [-h] [--crop-w CROP_WIDTH] [--crop-h CROP_HEIGHT] [--step ANGLE_STEP] [--type TYPE] [--rot-x ROTATE_X] [--rot-y ROTATE_Y] dataset output

# positional arguments:
#   dataset               The base directory path of the dataset.
#   output                Storage location for cropped images and masks

# options:
#   -h, --help            show this help message and exit
#   --crop-w CROP_WIDTH   width of the cropped image. Default is 900.
#   --crop-h CROP_HEIGHT  height of the cropped image. Default is 32.
#   --step ANGLE_STEP     angle of rotation. Default is 1.
#   --type TYPE           The dataset type: train, val, or test. Defaults to 'train'.
#   --rot-x ROTATE_X      X-coordinate of the rotation center. Default is 421.
#   --rot-y ROTATE_Y      Y-coordinate of the rotation center. Default is 435.
```

> [!NOTE]
> The default settings are tailored for cropping tulip CT images.

#### `./tools/dataset_mean-std_calc.py`
Code to calculate the mean and standard deviation of a dataset.

> [!WARNING]
> - A PyTorch-enabled environment is required.
> - Caution is advised as its accuracy has not been verified.

```bash
python dataset_mean-std_calc.py [-h] img-dir

# positional arguments:
#   img-dir     Directory to store images of the dataset.

# options:
#  -h, --help  show this help message and exit
```

#### `./tools/RandAugment.py`
Code for the data augmentation technique [RandAugment](https://proceedings.neurips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html).

> [!NOTE]
> Modifications are necessary for the choice of augmentation methods and handling of images.

```bash
python RandAugment.py [-h] [-n N] [-m M] [--aug-num AUG_NUM] dataset output

# positional arguments:
#   dataset            The base directory path of the dataset.
#   output             Directory path for the output of augmented images and masks.

# options:
#   -h, --help         show this help message and exit
#   -n N               The parameter 'n' in RandAugment. Default is 4.
#   -m M               The parameter 'm' in RandAugment. Default is 10
#   --aug-num AUG_NUM  Number of augmentations. Default is 100.
```

## Reference Sites
- [自前のMask画像からCOCO format jsonを作成](https://salt22g.hatenablog.jp/entry/2020/12/20/210419)
- [COCO Formatの作り方](https://qiita.com/harmegiddo/items/da131ae5bcddbbbde41f)
- [how to calculate “img_norm_cfg” for custom dataset #354](https://github.com/open-mmlab/mmdetection/issues/354)
- [voc2coco
/COCO_Image_Viewer.ipynb](https://github.com/Tony607/voc2coco/blob/master/COCO_Image_Viewer.ipynb)
- [ildoonet/pytorch-randaugment/RandAugment/augmentations.py](https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py)