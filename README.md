# mask_to_coco
各オブジェクトごとに色分けされたマスク画像から、COCOフォーマットのjsonファイルを生成するためのリポジトリ．

This repository for generating a COCO format JSON file from mask images color-coded for each object.

## Directry Structure
```
mask_to_coco
|-- README.md
|-- config
|   |-- __init__.py
|   |-- coco_config.py
|   `-- directory.py
|-- mask_to_coco.py
|-- mask_to_coco_multi_core.py
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
ディレクトリやCOCOフォーマットなどの設定を格納

Stores configurations, such as settings related to COCO format.

### `./utils`
コードに必要な関数を格納

Contains necessary functions for the code.

### `./visualization`
データセットの可視化に関するコードを格納

Stores code related to dataset visualization.

### `./tools`
データセットに関わるその他のコードを格納

Stores other code related to the dataset.

## How to Use
### `./mask_to_coco.py`
各オブジェクトごとに色分けされたマスク画像から、COCOフォーマットのjsonファイルを生成するコード．

Code to generate a COCO format JSON file from mask images color-coded for each object.

`./config/coco_config.py`でCOCOフォーマットに必要な**info，licenses，categories**を編集する必要がある．

You need to edit **'info,' 'licenses,' and 'categories'** in the COCO format within `./config/coco_config.py`.

```
usage: mask_to_coco.py [-h] [-t TYPE] [-n NAME] dir

positional arguments:
  dir                   The base directory path of the dataset.[Dataset Name]

optional arguments:
  -h, --help            show this help message and exit
  -t TYPE, --type TYPE  train or val or test. Default is train.
  -n NAME, --name NAME  Specify the JSON file name. The default is '[TYPE]_annotations.json'.
```

データセットは以下のディレクトリ構造に設定することで変換することができる．ただし，masksの画像は全て**PNG**で格納し，imagesの画像と同じ名前にする必要がある．（imagesの画像もPNG推奨）

You can perform the conversion by organizing the dataset with the following directory structure. However, all images in the 'masks' directory must be stored in **PNG format** and have the same names as the images in the 'images' directory.（It is recommended to use PNG format for images as well.）

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

### `./mask_to_coco_multi_core.py`
各オブジェクトごとに色分けされたマスク画像から、COCOフォーマットのjsonファイルを生成するコード（並列化バージョン）． データセットの要件は`mask_to_coco.py`と同様です．

Code to generate a COCO format JSON file from mask images color-coded for each object（**Parallel Version**).　The dataset requirements are the same as in `mask_to_coco.py`.

```
usage: mask_to_coco_multi_core.py [-h] [-t TYPE] [-n NAME] [-c CORE] dir

positional arguments:
  dir                   The base directory path of the dataset.

optional arguments:
  -h, --help            show this help message and exit
  -t TYPE, --type TYPE  train or val or test. Default is train.
  -n NAME, --name NAME  Specify the JSON file name. The default is '[TYPE]_annotations.json'.
  -c CORE, --core CORE  The number of cores to be used for parallelization. The default is the number of physical cores in the system.
```
**mask_to_coco.pyからの変更点：**　同じ色のマスクが分離している場合、それぞれを別のオブジェクトとして扱う．

**Changes from mask_to_coco.py:** When masks of the same color are separated, each of them is treated as a separate object


### `./visualization/visualization_mask.py`
COCOフォーマットのデータセットを可視化するためのコード．

Code for visualizing a COCO format dataset.

```
usage: visualization_mask.py [-h] [-o OUT] dir

positional arguments:
  dir                       The base directory path of the dataset.

optional arguments:
  -h, --help                show this help message and exit
  -o OUT, --output-dir OUT  The output directory. Default is ./out
```

mask_to_coco.pyでのディレクトリ構造（masksを除く）にする必要がある．出力は指定がなければ，`./out`以下に出力される．

The directory structure needs to match that in `mask_to_coco.py` (excluding the 'masks' directory). If no output path is specified, the output will be placed in `./out`.

### `./tools/crop_dataset.py`
COCOフォーマットのデータセットの画像からクロップ画像を作成し，新たなデータセットを作成するコード．**花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

Code to create cropped images from images in a COCO format dataset and generate a new dataset. **Specific processing is included to correspond to flower CT images, so modifications may be necessary when working with other data.**

```
usage: crop_dataset.py [-h] [-x WIDTH] [-y HEIGHT] [-n NUM] [-t TYPE]

optional arguments:
  -h, --help  show this help message and exit
  -x WIDTH    width of the cropped image. Default is 150.
  -y HEIGHT   height of the cropped image. Default is 150.
  -n NUM      Number of Crop Image Generations per Image. Default is 10.
  -t TYPE     train or val or test. Default is train.
```

入力するデータセットの格納先や出力先などは`./config/directory.py`で変更する必要がある．基本的には`mask_to_coco.py`でのディレクトリ構造（masksを除く）であることを仮定している．

Settings such as the storage location of the input dataset and the output destination need to be modified in `./config/directory.py`. It is assumed, in essence, that the directory structure (excluding 'masks') follows that in `mask_to_coco.py`.

### `./tools/crop_image-mask.py`
画像とマスク画像（各オブジェクトごとに色分けされたマスク画像）のディレクトリからそれぞれのクロップ画像を生成するコード． **花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

Code to generate individual cropped images from directories of images and mask images (mask images color-coded for each object). **Specific processing is included to correspond to flower CT images, so modifications may be necessary when working with other data.**

```
usage: crop_image-mask.py [-h] [-x WIDTH] [-y HEIGHT] [-n NUM] [-t TYPE] input_dir output_dir

positional arguments:
  input_dir   The base directory path of images and masks.
  output_dir  Storage location for cropped images and masks

optional arguments:
  -h, --help  show this help message and exit
  -x WIDTH    width of the cropped image. Default is 150.
  -y HEIGHT   height of the cropped image. Default is 150.
  -n NUM      Number of Crop Image Generations per Image. Default is 10.
  -t TYPE     train or val or test. Default is train.
```

基本的には`mask_to_coco.py`でのディレクトリ構造であることを仮定している．

It is assumed, in essence, that the directory structure follows that in `mask_to_coco.py`.

### `./tools/crop_image-mask_resize.py`
画像とマスク画像（各オブジェクトごとに色分けされたマスク画像）のディレクトリからそれぞれのクロップ画像を生成するコード． 生成されたクロップ画像は指定された画像サイズにリサイズされる． **花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

Code to generate individual cropped images from directories of images and mask images (mask images color-coded for each object). The generated cropped image is resized to the specified image size. **Specific processing is included to correspond to flower CT images, so modifications may be necessary when working with other data.**

```
usage: crop_image-mask_resize.py [-h] [--crop-x CROP_WIDTH] [--crop-y CROP_HEIGHT] [--resize-x RESIZE_WIDTH] [--resize-y RESIZE_HEIGHT] [-n NUM] [-t TYPE] input_dir output_dir

positional arguments:
  input_dir             The base directory path of images and masks.
  output_dir            Storage location for cropped images and masks

optional arguments:
  -h, --help            show this help message and exit
  --crop-x CROP_WIDTH   width of the cropped image. Default is 50.
  --crop-y CROP_HEIGHT  height of the cropped image. Default is 50.
  --resize-x RESIZE_WIDTH
                        width after resizing. Default is 150.
  --resize-y RESIZE_HEIGHT
                        height after resizing. Default is 150.
  -n NUM                Number of Crop Image Generations per Image. Default is 10.
  -t TYPE               train or val or test. Default is train.
```

基本的には`mask_to_coco.py`でのディレクトリ構造であることを仮定している．

It is assumed, in essence, that the directory structure follows that in `mask_to_coco.py`.

### `./tools/crop_rotate.py`

画像とマスク画像（各オブジェクトごとに色分けされたマスク画像）のディレクトリからそれぞれのクロップ画像を生成するコード． このコードは回転させながら横長の長方形のように画像を切り出す．前処理として，指定された回転の中心座標が画像中心となるように平行移動をしている． **花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

Code to generate individual cropped images from directories of images and mask images (mask images color-coded for each object). This code crops the image while rotating it to resemble a landscape-oriented rectangle. As a preprocessing step, it performs a translation to ensure that the specified rotation center coordinates align with the center of the image. **Specific processing is included to correspond to flower CT images, so modifications may be necessary when working with other data.**

```
usage: crop_rotate.py [-h] [--crop-w WIDTH] [--crop-h HEIGHT] [--step STEP] [--type TYPE] [--rotate-x CENTER_X] [--rotate-y CENTER_Y] input_dir output_dir

positional arguments:
  input_dir            The base directory path of images and masks.
  output_dir           Storage location for cropped images and masks

optional arguments:
  -h, --help           show this help message and exit
  --crop-w WIDTH       width of the cropped image. Default is 900.
  --crop-h HEIGHT      height of the cropped image. Default is 32.
  --step STEP          Number of Crop Image Generations per Image. Default is 10.
  --type TYPE          train or val or test. Default is train.
  --rotate-x CENTER_X  X-coordinate of the rotation center. Default is 421.
  --rotate-y CENTER_Y  Y-coordinate of the rotation center. Default is 435.
```

基本的には`mask_to_coco.py`でのディレクトリ構造であることを仮定している．

It is assumed, in essence, that the directory structure follows that in `mask_to_coco.py`.

### `./tools/dataset_mean-std_calc.py`

**PyTorchが使える環境が必要**

データセットの平均値と標準偏差を求めるコード．正確かどうかは未確認．

**PyTorch-enabled environment is required.** 

The code calculates the mean and standard deviation of the dataset. The accuracy of this code has not been verified.

```
usage: dataset_mean-std_calc.py [-h] dir

positional arguments:
  dir         Directory to store images of the dataset.

options:
  -h, --help  show this help message and exit
```

### `./tools/RandAugment.py`

データ拡張手法である[RandAugment](https://proceedings.neurips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html)のコード．**データ拡張方法や画像の取り扱いなど変更の必要がある．**

The code for the data augmentation technique, [RandAugment](https://proceedings.neurips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html). **Modifications are required for data augmentation methods and image handling.**

```
usage: RandAugment.py [-h] [-o OUTPUT_DIR] [-n N] [-m M] [-a NUM] input_dir

positional arguments:
  input_dir             Directory containing images and masks for data augmentation.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory of images and masks after data augmentation. Default is ./out
  -n N, --pram-n N      The parameter 'n' in RandAugment. Dafault is 4.
  -m M, --pram-m M      The parameter 'm' in RandAugment. Dafault is 10
  -a NUM, --aug-num NUM
                        Number of augmentations. Default is 100.
```

## Reference website
- [自前のMask画像からCOCO format jsonを作成](https://salt22g.hatenablog.jp/entry/2020/12/20/210419)
- [COCO Formatの作り方](https://qiita.com/harmegiddo/items/da131ae5bcddbbbde41f)
- [how to calculate “img_norm_cfg” for custom dataset #354](https://github.com/open-mmlab/mmdetection/issues/354)
- [voc2coco
/COCO_Image_Viewer.ipynb](https://github.com/Tony607/voc2coco/blob/master/COCO_Image_Viewer.ipynb)
- [ildoonet/pytorch-randaugment/RandAugment/augmentations.py](https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py)