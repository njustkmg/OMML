from torchmm import TorchMM


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="TorcnMM")
    parser.add_argument('--cuda', type=int, default=1)
    parser.add_argument('--config', type=str, default='configs/late_max.yml',
                        help='Path to model configuration param file.')
    parser.add_argument('--data_root', type=str, default='/home/br/PDMM/COCO_new',
                        help='Folder to dataset, include eg. dataset_coco.json, img_feat.npy, img_box.npy and label.npy.')
    parser.add_argument('--image_root', type=str, default='/home/br/PDMM/COCO_new/images',
                        help='Folder to original image file.')
    parser.add_argument('--out_root', type=str, default='experiment/cmml',
                        help='Folder to save experiment data, include model and log.')
    config = parser.parse_args()

    runner = TorchMM(config=config.config,
                      data_root=config.data_root,
                      image_root=config.image_root,
                      out_root=config.out_root,
                      cuda=config.cuda)

    runner.train()
    runner.test()
