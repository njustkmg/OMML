# Data Description
PaddleMM Provides processing of multi-modal data including text and image. The folder for storing data sets are organized as follows:
- images (Store the original images of the dataset)
- img_feat.npy (Image region features extracted by Faster-RCNN)
- img_box.npy (The location information of the image area extracted by Faster-RCNN)
- dataset.json (Store the relevant information of the original data set, such as text, data set division, label, etc. See how to read from paddlemm/datasets/reader)

## MS-COCO Dataset
To obtain the standard data loading format of the toolkit, the MS-COCO dataset needs to be processed as follows:
- Step 1. Download COCO2014 Tran/Val images and captions [here](https://cocodataset.org/#download) , merge the training set images and validation set images into 'images' folder.
- Step 2. Download COCO processing and dividing files by Andrej Karpathy [here](http://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip) .
- Step 3. Download COCO regional features and location information extracted by Faster-RCNN [here](https://storage.googleapis.com/up-down-attention/trainval_36.zip) .
- Step 4. Use paddlemm/scripts/coco_region.py and paddlemm/scripts/coco_region.py to process the original data to get image features and labels.
