from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from paddlenlp.transformers.bert.tokenizer import BertTokenizer


class PreTokenizer(object):

    def __init__(self,
                 max_len,
                 bert_model='bert-base-uncased',
                 add_special_token=True):
        """
                A class convert raw text into numbers by bert pretrained tokenizer
                :param max_len: max length of a text, manually or automatically set the longest sentence in the dataset
                :param add_special_token: whether to add special tokens
                """

        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(bert_model)
        self.add_special_token = add_special_token

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def mask_token(self):
        return self.tokenizer.convert_tokens_to_ids('[MASK]')

    @property
    def pad_token(self):
        return self.tokenizer.convert_tokens_to_ids('[PAD]')

    def __call__(self, text):

        if not isinstance(text, list):
            batch_token = self.tokenizer(text,
                                         max_seq_len=self.max_len,
                                         return_special_tokens_mask=self.add_special_token,
                                         return_attention_mask=True)

            input_ids = batch_token['input_ids']
            input_ids += [self.pad_token] * (self.max_len - len(input_ids))
            attn_mask = batch_token['attention_mask']
            attn_mask += [0] * (self.max_len - len(attn_mask))

            input_ids = np.array(input_ids, dtype=int)
            attn_mask = np.array(attn_mask, dtype=int)
            num = attn_mask.sum()

        else:
            # for multi sentence process
            batch_token = self.tokenizer(text,
                                         max_seq_len=self.max_len,
                                         return_special_tokens_mask=self.add_special_token,
                                         return_attention_mask=True)

            input_ids = [b['input_ids'] for b in batch_token]
            input_ids = [ip + [self.pad_token] * (self.max_len - len(ip)) for ip in input_ids]
            attn_mask = [b['attention_mask'] for b in batch_token]
            attn_mask = [ip + [0] * (self.max_len - len(ip)) for ip in attn_mask]

            input_ids = np.array(input_ids, dtype=int)
            attn_mask = np.array(attn_mask, dtype=int)
            num = attn_mask.sum(1)

        # return {'tokens': tokens, 'length': num, 'mask': mask}
        return input_ids, num, attn_mask