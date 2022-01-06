from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from transformers import BertTokenizer


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
        return self.tokenizer.mask_token_id

    @property
    def pad_token(self):
        return self.tokenizer.pad_token_id

    def __call__(self, text):

        if not isinstance(text, list):
            batch_token = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=[text],
                add_special_tokens=True,
                truncation='only_first',
                max_length=self.max_len,
                padding=True,
            )
            input_ids = batch_token['input_ids'][0]
            input_ids += [self.pad_token]*(self.max_len-len(input_ids))
            attn_mask = batch_token['attention_mask'][0]
            attn_mask += [0]*(self.max_len-len(attn_mask))

            input_ids = np.array(input_ids, dtype=int)
            attn_mask = np.array(attn_mask, dtype=int)
            num = attn_mask.sum()

        else:
            # for multi sentence process
            batch_token = self.tokenizer.batch_encode_plus(
                batch_text_or_text_pairs=text,
                add_special_tokens=True,
                truncation='only_first',
                max_length=self.max_len,
                padding=True,
            )
            input_ids = batch_token['input_ids']
            temp_len = len(input_ids[0])
            input_ids = [ip+[self.pad_token]*(self.max_len-temp_len) for ip in input_ids]
            attn_mask = batch_token['attention_mask']
            attn_mask = [ip+[0]*(self.max_len-temp_len) for ip in attn_mask]

            input_ids = np.array(input_ids, dtype=int)
            attn_mask = np.array(attn_mask, dtype=int)
            num = attn_mask.sum(1)

        # return {'tokens': tokens, 'length': num, 'mask': mask}
        return input_ids, num, attn_mask