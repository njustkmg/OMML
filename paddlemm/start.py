from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import os

from paddlemm.engines.fusion_trainer import FusionTrainer
from paddlemm.engines.caption_trainer import CaptionTrainer
from paddlemm.engines.retrieval_trainer import RetrievalTrainer
from paddlemm.engines.multitask_trainer import MultitaskTrainer

from paddlemm.utils.logger import get_logger
from paddlemm.utils.option import get_option

TrainerMap = {
    'fusion': FusionTrainer,
    'caption': CaptionTrainer,
    'retrieval': RetrievalTrainer,
    'multi_task': MultitaskTrainer
}


class PaddleMM(object):

    def __init__(self, config, data_root, image_root, out_root, gpu):

        opt = get_option(config, data_root, image_root, out_root, gpu)
        if not os.path.exists(opt.out_root):
            os.mkdir(opt.out_root)

        logger = get_logger(opt.out_root)

        for k, v in opt.items():
            logger.info(f"{str(k)} : {str(v)}")

        opt.logger = logger
        paddle.set_device(f'gpu:{str(opt.gpu)}')

        if opt.task in TrainerMap:
            self.trainer = TrainerMap[opt.task](opt)
        else:
            raise ValueError("Please specify task from [fusion, caption, retrieval]!")

    def train(self):
        self.trainer.train()

    def test(self):
        self.trainer.test()