# mask_to_coco

[For the English version of the README, click here.](./README_en.md)

## 概要
このリポジトリは，単一クラスのインスタンスセグメンテーション用にアノテーションされたマスク画像から，COCO形式のデータセット（JSONファイル）を生成することを目的としたリポジトリである．ここで扱うマスク画像において，背景を除いた画像内の各オブジェクトはそれぞれ異なる色のマスクが付けられている．

## ディレクトリ構造
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
ディレクトリやCOCOフォーマットなどの設定を格納

### `./utils`
コードに必要な関数を格納

### `./visualization`
データセットの可視化に関するコードを格納

### `./tools`
データセットに関わるその他のコードを格納

## 使い方

> [!IMPORTANT]
> - 花のCT画像に対応するため，特定の処理が入っているコードが存在する．他のデータを扱う際には変更の必要がある．
> - `mask_to_coco_parallel_segtype.py`を用いる場合はpycocotoolsのインストールが必要．

### `./mask_to_coco.py`
各オブジェクトごとに色分けされたインスタンスセグメンテーション（背景を除く1クラス）用のマスク画像から，COCO形式のデータセットを作成するコード．

`./config/coco_config.py`でCOCOフォーマットに必要な**info，licenses，categories**を編集する必要がある．

```bash
python mask_to_coco.py [-h] [-t TYPE] [-n NAME] dir

# positional arguments:
#   dir                   The base directory path of the dataset.[Dataset Name]

# optional arguments:
#   -h, --help            show this help message and exit
#   -t TYPE, --type TYPE  train or val or test. Default is train.
#   -n NAME, --name NAME  Specify the JSON file name. The default is '[TYPE]_annotations.json'.
```

データセットは以下のディレクトリ構造に設定することで変換することができる．ただし，masksの画像は全て**PNG**で格納し，imagesの画像と同じ名前にする必要がある．（imagesの画像もPNG推奨）

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
各オブジェクトごとに色分けされたインスタンスセグメンテーション（背景を除く1クラス）用のマスク画像から，並列処理でCOCO形式のデータセットを作成する． データセットの要件は`mask_to_coco.py`と同様．

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
**mask_to_coco.pyからの変更点：**　同じ色のマスクが分離している場合，それぞれを別のオブジェクトとして扱う．

### `./mask_to_coco_parallel_segtype.py`
各オブジェクトごとに色分けされたインスタンスセグメンテーション（背景を除く1クラス）用のマスク画像から，並列処理でCOCO形式のデータセットを作成する．セグメンテーションのフォーマットは，RLE (Run-Length Encoding)またはPolygonから選択できる．データセットの要件は`mask_to_coco.py`と同様．

**pycocotoolsのインストールが必要**

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
**mask_to_coco.pyからの変更点：**　同じ色のマスクが分離している場合，それぞれを別のオブジェクトとして扱う．


### `./visualization/visualization_mask.py`
COCOフォーマットのデータセットを可視化するためのコード．

```bash
python visualization_mask.py [-h] [-o OUT] dir

# positional arguments:
#   dir                       The base directory path of the dataset.

# optional arguments:
#   -h, --help                show this help message and exit
#   -o OUT, --output-dir OUT  The output directory. Default is ./out
```

mask_to_coco.pyでのディレクトリ構造（masksを除く）にする必要がある．出力は指定がなければ，`./out`以下に出力される．

### `./tools/crop_dataset.py`
COCOフォーマットのデータセットの画像からクロップ画像を作成し，新たなデータセットを作成するコード．**花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

```bash
python crop_dataset.py [-h] [-x WIDTH] [-y HEIGHT] [-n NUM] [-t TYPE]

# optional arguments:
#   -h, --help  show this help message and exit
#   -x WIDTH    width of the cropped image. Default is 150.
#   -y HEIGHT   height of the cropped image. Default is 150.
#   -n NUM      Number of Crop Image Generations per Image. Default is 10.
#   -t TYPE     train or val or test. Default is train.
```

入力するデータセットの格納先や出力先などは`./config/directory.py`で変更する必要がある．基本的には`mask_to_coco.py`でのディレクトリ構造（masksを除く）であることを仮定している．

### `./tools/crop_image-mask.py`
画像とマスク画像（各オブジェクトごとに色分けされたマスク画像）のディレクトリからそれぞれのクロップ画像を生成するコード． **花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

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

基本的には`mask_to_coco.py`でのディレクトリ構造であることを仮定している．

### `./tools/crop_image-mask_resize.py`
画像とマスク画像（各オブジェクトごとに色分けされたマスク画像）のディレクトリからそれぞれのクロップ画像を生成するコード． 生成されたクロップ画像は指定された画像サイズにリサイズされる． **花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

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

基本的には`mask_to_coco.py`でのディレクトリ構造であることを仮定している．

### `./tools/crop_rotate.py`

画像とマスク画像（各オブジェクトごとに色分けされたマスク画像）のディレクトリからそれぞれのクロップ画像を生成するコード． このコードは回転させながら横長の長方形のように画像を切り出す．前処理として，指定された回転の中心座標が画像中心となるように平行移動をしている． **花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

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

基本的には`mask_to_coco.py`でのディレクトリ構造であることを仮定している．

### `./tools/dataset_mean-std_calc.py`

データセットの平均値と標準偏差を求めるコード．

> [!NOTE]
> PyTorchが使える環境が必要である．また，正確かどうかの確認ができていないため注意が必要．

```bash
python dataset_mean-std_calc.py [-h] dir

# positional arguments:
#   dir         Directory to store images of the dataset.

# options:
#  -h, --help  show this help message and exit
```

### `./tools/RandAugment.py`

データ拡張手法である[RandAugment](https://proceedings.neurips.cc/paper/2020/hash/d85b63ef0ccb114d0a3bb7b7d808028f-Abstract.html)のコード．

> [!NOTE]
> データ拡張方法の選択や画像の取り扱いなど，変更の必要がある．


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

## 参考サイト
- [自前のMask画像からCOCO format jsonを作成](https://salt22g.hatenablog.jp/entry/2020/12/20/210419)
- [COCO Formatの作り方](https://qiita.com/harmegiddo/items/da131ae5bcddbbbde41f)
- [how to calculate “img_norm_cfg” for custom dataset #354](https://github.com/open-mmlab/mmdetection/issues/354)
- [voc2coco
/COCO_Image_Viewer.ipynb](https://github.com/Tony607/voc2coco/blob/master/COCO_Image_Viewer.ipynb)
- [ildoonet/pytorch-randaugment/RandAugment/augmentations.py](https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py)