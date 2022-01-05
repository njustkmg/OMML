# 数据描述
PaddleMM 提供处理包括文本和图片的多模态数据，存放数据集的文件夹组织如下：
- images (存放数据集原始图片)
- img_feat.npy (经过 Faster-RCNN 提取的图像区域特征)
- img_box.npy (经过 Faster-RCNN 提取的图像区域位置信息)
- dataset.json (存放原始数据集相关信息，如文本、数据集划分、标签等，读取方式见 paddlemm/datasets/reader)

## MSCOCO 数据集
为得到工具包标准数据读取格式，需要对 MSCOCO 数据集经过以下处理
- Step 1. 下载 COCO2014 Tran/Val 原始图片和文本数据 [地址](https://cocodataset.org/#download) ，将原始的训练集图片和验证集图片合并成 images 文件夹；
- Step 2. 下载 Andrej Karpathy 的 COCO 处理和划分文件 [地址](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) ；
- Step 3. 下载经过 Faster-RCNN 提取的 COCO 区域特征和位置信息 [地址](https://storage.googleapis.com/up-down-attention/trainval_36.zip) ；
- Step 4. 分别使用 paddlemm/scripts/coco_region.py 和 paddlemm/scripts/coco_region.py 处理原始数据得到图片特征和标签。
