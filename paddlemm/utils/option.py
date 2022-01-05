from easydict import EasyDict
import yaml


def get_option(config, data_root, image_root, out_root, gpu):

    with open(config, 'r') as f:
        opt = EasyDict(yaml.safe_load(f))

    opt.data_root = data_root
    opt.image_root = image_root
    opt.out_root = out_root
    opt.gpu = gpu

    return opt