# mask_to_coco

[For the English version of the README, click here.](/README_en.md)

## 概要
このリポジトリは，単一クラスのインスタンスセグメンテーション用にアノテーションされたマスク画像から，COCO形式のデータセット（JSONファイル）を生成することを目的としたリポジトリである．ここで扱うマスク画像において，背景を除いた画像内の各オブジェクトはそれぞれ異なる色のマスクが付けられている．

## 目次
- [ディレクトリ構造](#ディレクトリ構造)
- [使い方](#使い方)
  - [COCO形式のデータセット作成](#coco形式のデータセット作成)
  - [可視化](#可視化visualization)
  - [ツール](#ツールtools)
- [参考サイト](#参考サイト)

## ディレクトリ構造
```
mask_to_coco
├── config                  - ディレクトリやCOCOフォーマットなどの設定を格納
│   ├── __init__.py
│   ├── coco_config.py
│   └── directory.py
├── tools                   - データセットに関わるその他のコードを格納
│   ├── RandAugment.py
│   ├── crop_dataset.py
│   ├── crop_image-mask.py
│   ├── crop_image-mask_resize.py
│   ├── crop_rotate.py
│   └── dataset_mean-std_calc.py
├── utils                   - コードに必要な関数を格納
│   ├── __init__.py
│   └── tools.py
├── visualization           - データセットの可視化に関するコードを格納
│   ├── COCO_Image_Viewer.ipynb
│   └── visualization_mask.py
├── README.md
├── mask_to_coco.py
├── mask_to_coco_parallel_polygon.py
└── mask_to_coco_parallel_segtype.py    - このコードの使用を推奨
```

## 使い方

> [!IMPORTANT]
> - 花のCT画像に対応するため，特定の処理が入っているコードが存在する．他のデータを扱う際には変更の必要がある．
> - `mask_to_coco_parallel_segtype.py`を用いる場合はpycocotoolsのインストールが必要．

### COCO形式のデータセット作成
ここでは，単一クラスのインスタンスセグメンテーション用にアノテーションされたマスク画像から，COCO形式のデータセット（JSONファイル）を生成するコードについて説明している．**コードは3種類あるが基本的には，`./mask_to_coco_parallel_segtype.py`を使用することを推奨する．**

- [`mask_to_coco.py`](#mask_to_cocopy)
- [`mask_to_coco_parallel_polygon.py`](#mask_to_coco_parallel_polygonpy)
- [`mask_to_coco_parallel_segtype.py`**（推奨）**](#mask_to_coco_parallel_segtypepy推奨)

データセットは以下のディレクトリ構造に設定することでjsonファイルを生成することができる．ここで説明するコードを用いることで，`[Dataset Name]/`に`[TYPE]_annotation.json`が生成される．`[TYPE]`は指定したデータセットの種類（`train`，`val`，`test`）である．

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
> - `images/`の画像の拡張子は**PNGもしくはTIF**を推奨する．
> - `masks/`の画像は全て**PNG**で格納し，`images/`の画像と同じ名前にする必要がある．
> - `./config/coco_config.py`でCOCOフォーマットに必要な**info，licenses，categories**を編集する必要がある．

#### `./mask_to_coco.py`
各オブジェクトごとに色分けされたインスタンスセグメンテーション（背景を除く1クラス）用のマスク画像から，COCO形式のデータセットを作成するコード．

```bash
python mask_to_coco.py [-h] [-t TYPE] [-n NAME] dataset

# positional arguments:
#   dataset               The base directory path of the dataset.

# optional arguments:
#   -h, --help   show this help message and exit
#   --type TYPE  train or val or test. Default is train.
#   --name NAME  Specify the JSON file name. The default is '[TYPE]_annotations.json'.
```

#### `./mask_to_coco_parallel_polygon.py`
各オブジェクトごとに色分けされたインスタンスセグメンテーション（背景を除く1クラス）用のマスク画像から，並列処理でCOCO形式のデータセットを作成する． 

> [!NOTE]
> 画像の切り出しなどの処理により同じ色のマスクが分離している場合，それぞれを別のオブジェクトとして扱う．

```bash
python mask_to_coco_parallel_polygon.py [-h] [--type TYPE] [--name NAME] [--proc-num PROC_NUM]

# positional arguments:
#   datset               The base directory path of the dataset.

# optional arguments:
#   -h, --help           show this help message and exit
#   --type TYPE          The dataset type: train, val, or test. Defaults to 'train'.
#   --name NAME          The JSON file name for COCO annotations. Defaults to '[TYPE]_annotations.json'.
#   --proc-num PROC_NUM  The number of cores used for parallelization. Defaults to the system's physical core count.
```

#### `./mask_to_coco_parallel_segtype.py`**（推奨）**
各オブジェクトごとに色分けされたインスタンスセグメンテーション（背景を除く1クラス）用のマスク画像から，並列処理でCOCO形式のデータセットを作成する．セグメンテーションのフォーマットは，RLE (Run-Length Encoding)またはPolygonから選択できる．

> [!NOTE]
> - pycocotoolsのインストールが必要
>   ```bash
>   pip install pycocotools
>   ```
> - 画像の切り出しなどの処理により同じ色のマスクが分離している場合，それぞれを別のオブジェクトとして扱う．

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

### 可視化（`visualization/`）
ここではデータセットを可視化するコードについて説明する．
#### `./visualization/visualization_mask.py`
COCOフォーマットのデータセットを可視化するためのコード．

```bash
python visualization_mask.py [-h] [--type TYPE] [--output OUTPUT] dataset

# positional arguments:
#   dataset          The base directory path of the dataset.

# options:
#   -h, --help       show this help message and exit
#   --type TYPE      The dataset type: train, val, or test. Defaults to 'train'.
#   --output OUTPUT  The output directory. Default is ./out
```

[COCO形式のデータセット作成](#coco形式のデータセット作成)で説明したディレクトリ構造（`masks/`を除く）にする必要がある．出力は指定がなければ，`./out`以下に出力される．

### ツール（`tools/`）
ここではデータセットを作成する際に役に立つコードについて説明する．ここで説明される`dataset`ディレクトリも[COCO形式のデータセット作成](#coco形式のデータセット作成)で説明したディレクトリ構造（`masks/`を除く場合もある）にする必要がある．

#### `./tools/crop_dataset.py`
既存のCOCOフォーマットのデータセットからランダムに画像を切り出し，新たなデータセットを生成するコード．ここではjsonファイルの生成までは行わない．**花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

<img src="/assets/crop_dataset_ja.jpg" width="90%" />

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

出力は指定がなければ，`./out/crop_dataset/`以下に出力される．

#### `./tools/crop_image-mask.py`
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

#### `./tools/crop_image-mask_resize.py`
画像とマスク画像（各オブジェクトごとに色分けされたマスク画像）のディレクトリからそれぞれのクロップ画像を生成するコード． 生成されたクロップ画像は指定された画像サイズにリサイズされる． **花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

<img src="/assets/crop_image-mask_resize_ja.jpg" width="90%" />

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

画像とマスク画像（各オブジェクトごとに色分けされたマスク画像）のディレクトリからそれぞれのクロップ画像を生成するコード． このコードは回転させながら横長の長方形のように画像を切り出す．前処理として，指定された回転の中心座標が画像中心となるように平行移動をしている． **花のCT画像に対応するため，特定の処理が入っており，他のデータを扱う際には変更の必要がある．**

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
> デフォルトの設定は，オランダ紅のCT画像の切り出しに対応している．

#### `./tools/dataset_mean-std_calc.py`

データセットの平均値と標準偏差を求めるコード．

> [!WARNING]
> - PyTorchが使える環境が必要である．
> - 正確かどうかの確認ができていないため注意が必要．

```bash
python dataset_mean-std_calc.py [-h] dir

# positional arguments:
#   dir         Directory to store images of the dataset.

# options:
#  -h, --help  show this help message and exit
```

#### `./tools/RandAugment.py`

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