# mask_to_coco

[日本語版READMEはこちら](./README.md)

> [!WARNING]
> This README has been translated from the Japanese version by ChatGPT. Please be cautious as the translation may contain inaccuracies.

## Overview
This repository is designed to generate COCO format datasets (JSON files) from mask images annotated for single-class instance segmentation. In these mask images, each object within the image, excluding the background, is marked with a mask of a distinct color.

## Directory Structure
```
mask_to_coco
|-- README.md
|-- config
|   |-- __init__.py
|   |-- coco_config.py
|   `-- directory.py
|-- mask_to_coco.py
|-- mask_to_coco_parallel_polygon.py
|-- mask_to_coco_parallel_segtype.py
|-- tools
|   |-- RandAugment.py
|   |-- crop_dataset.py
|   |-- crop_image-mask.py
|   |-- crop_image-mask_resize.py
|   |-- crop_rotate.py
|   `-- dataset_mean-std_calc.py
|-- utils
|   |-- __init__.py
|   `-- tools.py
`-- visualization
    |-- COCO_Image_Viewer.ipynb
    `-- visualization_mask.py
```

### `./config`
Stores settings for directories and the COCO format.

### `./utils`
Contains necessary functions for the code.

### `./visualization`
Holds code for dataset visualization.

### `./tools`
Contains additional codes related to the dataset.

## Usage

> [!IMPORTANT]
> - Some codes contain specific processes to accommodate flower CT images, which may need adjustments for handling other data types.
> - Installation of `pycocotools` is required to use `mask_to_coco_parallel_segtype.py`.

### `./mask_to_coco.py`
A script for creating COCO format datasets from instance segmentation mask images colored for each object. Editing **info, licenses, categories** required for COCO format in `./config/coco_config.py` is necessary.

```bash
python mask_to_coco.py [-h] [-t TYPE] [-n NAME] dir

# positional arguments:
#   dir                   The base directory path of the dataset.[Dataset Name]

# optional arguments:
#   -h, --help            show this help message and exit
#   -t TYPE, --type TYPE  train or val or test. Default is train.
#   -n NAME, --name NAME  Specify the JSON file name. The default is '[TYPE]_annotations.json'.
```

The dataset can be converted by setting it to the following directory structure. However, all images in masks must be stored in **PNG** and have the same name as the images in images (images in images are also recommended to be in PNG).

```
[Dataset Name]
|-- images
|   |-- train
|   |   |-- aaa.jpg
|   |   |-- bbb.jpg
|   |   `-- ccc.jpg 
|   `-- val
|       |-- zzz.jpg
|       `-- zzz.jpg
`-- masks
    |-- train
    |   |-- aaa.png
    |   |-- bbb.png
    |   `-- ccc.png 
    `-- val
        |-- zzz.png
        `-- zzz.png
```

### `./mask_to_coco_parallel_polygon.py`
Parallel processing script for creating COCO format datasets from mask images colored for each object. The dataset requirements are the same as `mask_to_coco.py`.

```bash
python mask_to_coco_parallel_polygon.py [-h] [-t TYPE] [-n NAME] [-c CORE] dir

# positional arguments:
#   dir                   The base directory path of the dataset.

# optional arguments:
#   -h, --help            show this help message and exit
#   -t TYPE, --type TYPE  train or val or test. Default is train.
#   -n NAME, --name NAME  Specify the JSON file name. The default is '[TYPE]_annotations.json'.
#   -c CORE, --core CORE  The number of cores to be used for parallelization. The default is the number of physical cores in the system.
```

**Changes from `mask_to_coco.py`:** If masks of the same color are separated, each is treated as a different object.

### `./mask_to_coco_parallel_segtype.py`
Parallel processing script for creating COCO format datasets from instance segmentation mask images colored for each object. The segmentation format can be chosen from RLE (Run-Length Encoding) or Polygon. The dataset requirements are the same as `mask_to_coco.py`.

**Installation of `pycocotools` is required.**

```bash
python mask_to_coco_parallel_segtype.py [-h] [-t TYPE] [-f FORMAT] [-n NAME] [-c CORE] dir

# positional arguments:
#   dir                   The base directory path of the dataset.

# options:
#   -h, --help            show this help message and exit
#   -t TYPE, --type TYPE  train or val or test. Default is train.
#   -f FORMAT, --format FORMAT
#                         Selection between RLE format or Polygon format (0: RLE format, 1: Polygon format). The default is 0: RLE format.
#   -n NAME, --name NAME  Specify the JSON file name. The default is '[TYPE]_annotations.json'.
#   -c CORE, --core CORE  The number of cores to be used for parallelization. The default is the number of physical cores in the system.
```

**Changes from `mask_to_coco.py`:** If masks of the same color are separated, each is treated as a different object.

### `./visualization/visualization_mask.py`
Code for visualizing a COCO format dataset.

```bash
python visualization_mask.py [-h] [-o OUT] dir

# positional arguments:
#   dir                       The base directory path of the dataset.

# optional arguments:
#   -h, --help                show this help message and exit
#   -o OUT, --output-dir OUT  The output directory. Default is ./out
```

The directory structure (excluding masks) must be the same as in `mask_to_coco.py`. The output is stored in `./out` by default unless specified otherwise.

### `./tools/crop_dataset.py`
Code for creating cropped images from COCO format dataset images and generating a new dataset. **Contains specific processing for flower CT images, requiring modifications for handling other data.**

```bash
python crop_dataset.py [-h] [-x WIDTH] [-y HEIGHT] [-n NUM] [-t TYPE]

# optional arguments:
#   -h, --help  show this help message and exit
#   -x WIDTH    width of the cropped image. Default is 150.
#   -y HEIGHT   height of the cropped image. Default is 150.
#   -n NUM      Number of Crop Image Generations per Image. Default is 10.
#   -t TYPE     train or val or test. Default is train.
```

The storage location for input datasets and output, among other settings, need to be changed in `./config/directory.py`. It assumes a directory structure (excluding masks) similar to `mask_to_coco.py`.

### `./tools/crop_image-mask.py`
This script generates cropped images from directories of images and mask images (mask images color-coded for each object). **It contains specific processes for flower CT images, requiring modifications when handling other data types.**

```bash
python crop_image-mask.py [-h] [-x WIDTH] [-y HEIGHT] [-n NUM] [-t TYPE] input_dir output_dir

# positional arguments:
#   input_dir   The base directory path of images and masks.
#   output_dir  Storage location for cropped images and masks

# optional arguments:
#   -h, --help  show this help message and exit
#   -x WIDTH    width of the cropped image. Default is 150.
#   -y HEIGHT   height of the cropped image. Default is 150.
#   -n NUM      Number of Crop Image Generations per Image. Default is 10.
#   -t TYPE     train or val or test. Default is train.
```

It assumes a directory structure similar to `mask_to_coco.py`.

### `./tools/crop_image-mask_resize.py`
This script generates cropped images from directories of images and mask images (mask images color-coded for each object). The cropped images produced are resized to a specified image size. **It includes specific processes for flower CT images, requiring modifications when dealing with other data types.**

```bash
python crop_image-mask_resize.py [-h] [--crop-x CROP_WIDTH] [--crop-y CROP_HEIGHT] [--resize-x RESIZE_WIDTH] [--resize-y RESIZE_HEIGHT] [-n NUM] [-t TYPE] input_dir output_dir

# positional arguments:
#   input_dir             The base directory path of images and masks.
#   output_dir            Storage location for cropped images and masks

# optional arguments:
#   -h, --help            show this help message and exit
#   --crop-x CROP_WIDTH   width of the cropped image. Default is 50.
#   --crop-y CROP_HEIGHT  height of the cropped image. Default is 50.
#   --resize-x RESIZE_WIDTH
#                         width after resizing. Default is 150.
#   --resize-y RESIZE_HEIGHT
#                         height after resizing. Default is 150.
#   -n NUM                Number of Crop Image Generations per Image. Default is 10.
#   -t TYPE               train or val or test. Default is train.
```

It assumes a directory structure similar to `mask_to_coco.py`.

### `./tools/crop_rotate.py`
Generates cropped images from directories of images and mask images (mask images color-coded for each object), cutting out images like a horizontal rectangle while rotating. As preprocessing, it translates so that the specified rotation center coordinates become the center of the image. **Contains specific processes for flower CT images, requiring modifications when handling other data types.**

```bash
python crop_rotate.py [-h] [--crop-w WIDTH] [--crop-h HEIGHT] [--step STEP] [--type TYPE] [--rotate-x CENTER_X] [--rotate-y CENTER_Y] input_dir output_dir

# positional arguments:
#   input_dir            The base directory path of images and masks.
#   output_dir           Storage location for cropped images and masks

# optional arguments:
#   -h, --help           show this help message and exit
#   --crop-w WIDTH       width of the cropped image. Default is 900.
#   --crop-h HEIGHT      height of the cropped image. Default is 32.
#   --step STEP          Number of Crop Image Generations per Image. Default is 10.
#   --type TYPE          train or val or test. Default is train.
#   --rotate-x CENTER_X  X-coordinate of the rotation center. Default is 421.
#   --rotate-y CENTER_Y  Y-coordinate of the rotation center. Default is 435.
```

Assumes a directory structure similar to `mask_to_coco.py`.

### `./tools/dataset_mean-std_calc.py`
Code to calculate the mean and standard deviation of the dataset.

> [!NOTE]
> Requires an environment where PyTorch is available. Caution is advised as its accuracy has not been verified.

```bash
python dataset_mean-std_calc.py [-h] dir

# positional arguments:
#   dir         Directory to store images of the dataset.

# options:
#  -h, --help  show this help message and exit
```

### `./tools/RandAugment.py`
Code for the data augmentation method, [RandAugment](https://proceedings.neurips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html). 

> [!NOTE]
> Changes are necessary for the selection of data augmentation methods and the handling of images.

```bash
python RandAugment.py [-h] [-o OUTPUT_DIR] [-n N] [-m M] [-a NUM] input_dir

# positional arguments:
#   input_dir             Directory containing images and masks for data augmentation.

# optional arguments:
#   -h, --help            show this help message and exit
#   -o OUTPUT_DIR, --output-dir OUTPUT_DIR
#                         Output directory of images and masks after data augmentation. Default is ./out
#   -n N, --pram-n N      The parameter 'n' in RandAugment. Dafault is 4.
#   -m M, --pram-m M      The parameter 'm' in RandAugment. Dafault is 10
#   -a NUM, --aug-num NUM
#                         Number of augmentations. Default is 100.
```

## Reference Sites
- [自前のMask画像からCOCO format jsonを作成](https://salt22g.hatenablog.jp/entry/2020/12/20/210419)
- [COCO Formatの作り方](https://qiita.com/harmegiddo/items/da131ae5bcddbbbde41f)
- [how to calculate “img_norm_cfg” for custom dataset #354](https://github.com/open-mmlab/mmdetection/issues/354)
- [voc2coco
/COCO_Image_Viewer.ipynb](https://github.com/Tony607/voc2coco/blob/master/COCO_Image_Viewer.ipynb)
- [ildoonet/pytorch-randaugment/RandAugment/augmentations.py](https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py)