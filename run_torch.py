from torchmm import TorchMM

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TorchMM")
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--config', type=str, default='configs/fusion/early_add.yml',
                        help='Path to model configuration param file.')
    parser.add_argument('--data_root', type=str, default='../data/COCO',
                        help='Folder to dataset, for example COCO, include eg. dataset_coco.json, img_feat.npy, img_box.npy and label.npy.')
    parser.add_argument('--image_root', type=str, default='../data/COCO/images',
                        help='Folder to original image file.')
    # parser.add_argument('--data_root', type=str, default='/home/zcb/XWJ/PaddleMM/TomBERT-master/absa_data/twitter',
    #                     help='Folder to dataset, for example COCO, include eg. dataset_coco.json, img_feat.npy, img_box.npy and label.npy.')
    # parser.add_argument('--image_root', type=str, default='/home/zcb/XWJ/PaddleMM/IJCAI2019_data/twitter2017_images/',
    #                     help='Folder to original image file.')
    parser.add_argument('--out_root', type=str, default='experiment/early_add_torch',
                        help='Folder to save experiment data, include model and log.')
    config = parser.parse_args()

    runner = TorchMM(config=config.config,
                     data_root=config.data_root,
                     image_root=config.image_root,
                     out_root=config.out_root,
                     cuda=config.cuda)

    runner.train()
    runner.test()
