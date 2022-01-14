from paddlemm import PaddleMM


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(description="PaddleMM")
    parser.add_argument('--gpu', type=int, default=1)
    parser.add_argument('--config', type=str, default='configs/retrieval/cmml.yml',
                        help='Path to model configuration param file.')
    parser.add_argument('--data_root', type=str, default='data/COCO',
                        help='Folder to dataset, include eg. dataset_coco.json, img_feat.npy, img_box.npy and label.npy.')
    parser.add_argument('--image_root', type=str, default='data/COCO/images',
                        help='Folder to original image file.')
    parser.add_argument('--out_root', type=str, default='experiment/cmml',
                        help='Folder to save experiment data, include model and log.')
    config = parser.parse_args()

    runner = PaddleMM(config=config.config,
                      data_root=config.data_root,
                      image_root=config.image_root,
                      out_root=config.out_root,
                      gpu=config.gpu)

    runner.train()
    runner.test()
