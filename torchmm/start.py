from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import os

from torchmm.engines.fusion_trainer import FusionTrainer
from torchmm.engines.caption_trainer import CaptionTrainer
from torchmm.engines.retrieval_trainer import RetrievalTrainer
from torchmm.engines.multitask_trainer import MultitaskTrainer

from torchmm.utils.option import get_option
from torchmm.utils.logger import get_logger

TrainerMap = {
    'fusion': FusionTrainer,
    'caption': CaptionTrainer,
    'retrieval': RetrievalTrainer,
    'multi_task': MultitaskTrainer
}


class TorchMM(object):

    def __init__(self, config, data_root, image_root, out_root, cuda):

        opt = get_option(config, data_root, image_root, out_root, cuda)
        if not os.path.exists(opt.out_root):
            os.mkdir(opt.out_root)

        logger = get_logger(opt.out_root)

        for k, v in opt.items():
            logger.info(f"{str(k)} : {str(v)}")

        opt.logger = logger

        if torch.cuda.is_available():
            torch.cuda.set_device(opt.cuda)

        if opt.task in TrainerMap:
            self.trainer = TrainerMap[opt.task](opt)
        else:
            raise ValueError("Please specify task from [fusion, caption, retrieval]")

    def train(self):
        self.trainer.train()

    def test(self):
        self.trainer.test()