from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from tqdm import tqdm
import os

import paddle
from paddle.io import DataLoader

from paddlemm.metrics import score_retrieval
from .base_trainer import BaseTrainer
from paddlemm.models import SCAN, SGRAF, VSEPP, IMRAM

FunMap = {
    'scan': SCAN,
    'sgraf': SGRAF,
    'vsepp': VSEPP,
    'imram': IMRAM
}


class RetrievalTrainer(BaseTrainer):

    def __init__(self, opt):
        super().__init__(opt)

        self.opt = opt
        self.best_score = 0.0
        self.best_loss = float(np.inf)

        # embed_size for calculate similarity between image and text
        self.embed_size = opt.embed_size
        self.max_len = self.dataset.max_len

        # for coco dataset, Cross-validate according to the settings in the paper
        self.fold5 = opt.get('fold5', True)

    def encode_data(self, data_loader, split='valid'):
        self.model.eval()

        length = 5000 if split == 'valid' else 25000
        if self.opt.image_type == 'region':
            img_embs = np.zeros((length, 36, self.embed_size), dtype=np.float32)
            cap_embs = np.zeros((length, self.max_len+2, self.embed_size), dtype=np.float32)
        else:
            img_embs = np.zeros((length, self.embed_size), dtype=np.float32)
            cap_embs = np.zeros((length, self.embed_size), dtype=np.float32)

        cap_lens = np.array([0] * length)

        for idx, batch in enumerate(tqdm(data_loader(), ncols=80)):
            # only use 5000 samples for evaluation
            if idx >= 50 and split == 'valid':
                break
            img_emb, cap_emb, cap_len = self.model.forward_emb(batch)

            # cache embeddings
            start = idx * 100
            end = min(len(self.dataset), (idx + 1) * 100)
            img_embs[start: end] = np.array(img_emb)
            cap_embs[start: end, :, :] = np.array(cap_emb)
            cap_lens[start: end] = np.array(cap_len.squeeze())

        return img_embs, cap_embs, cap_lens

    def evaluate(self):
        self.model.eval()

        valid_loader = DataLoader(self.dataset.valid_(),
                                  shuffle=False,
                                  batch_size=100,
                                  num_workers=self.num_workers)
        img_embs, cap_embs, cap_lens = self.encode_data(valid_loader)
        img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])
        sims = FunMap[self.model_name].cal_sim(self.model, img_embs, cap_embs, cap_lens, **self.opt)
        val_res = score_retrieval(sims, npts=100)
        print(val_res)
        return val_res

    def test(self):

        checkpoint = paddle.load(os.path.join(self.opt.out_root, 'best_model.pdparams'))
        self.model.set_state_dict(checkpoint)
        self.model.eval()
        test_loader = DataLoader(self.dataset.test_(),
                                 shuffle=False,
                                 batch_size=100,
                                 num_workers=self.num_workers)
        img_embs, cap_embs, cap_lens = self.encode_data(test_loader, split='test')

        if not self.fold5:
            # no cross-validation, full evaluation
            img_embs = np.array([img_embs[i] for i in range(0, len(img_embs), 5)])

            sims = FunMap[self.model_name].cal_sim(self.model, img_embs, cap_embs, cap_lens, **self.opt)
            result = score_retrieval(sims, npts=1000)

        else:
            # 5fold cross-validation, only for MSCOCO
            all_results = []
            for i in range(5):
                img_embs_shard = img_embs[i * 5000:(i + 1) * 5000:5]
                cap_embs_shard = cap_embs[i * 5000:(i + 1) * 5000]
                cap_lens_shard = cap_lens[i * 5000:(i + 1) * 5000]
                sims = FunMap[self.model_name].cal_sim(self.model, img_embs_shard, cap_embs_shard, cap_lens_shard, **self.opt)
                all_results.append(score_retrieval(sims, npts=1000))

            result = {}
            for res in all_results:
                for k, v in res.items():
                    if k in result:
                        result[k] += v
                    else:
                        result[k] = v
            result = {k: v/5 for k, v in result.items()}

        for k, v in result.items():
            self.logger.info(f"{k}: {str(v)}")
