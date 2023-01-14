from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import torch
import csv
import logging
from torchvision import transforms
from PIL import Image
from torchmm.datasets.reader.tokenization import BertTokenizer
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import Dataset

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


def image_process(image_path, transform):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    return image


class MMInputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b, img_id, label=None):
        """Constructs a InputExample.

        self:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.img_id = img_id
        self.label = label


class MMInputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, added_input_mask, segment_ids, s2_input_ids, s2_input_mask, \
                 s2_segment_ids, img_feat, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.added_input_mask = added_input_mask
        self.segment_ids = segment_ids
        self.s2_input_ids = s2_input_ids
        self.s2_input_mask = s2_input_mask
        self.s2_segment_ids = s2_segment_ids
        self.img_feat = img_feat
        self.label_id = label_id


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_root):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_root):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_root):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_tsv(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with open(input_file, "r", encoding='utf-8') as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            for line in reader:
                lines.append(line)
            return lines


class AbmsaProcessor(DataProcessor):
    """Processor for the MRPC data set (GLUE version)."""

    def get_train_examples(self, data_root):
        """See base class."""
        logger.info("LOOKING AT {}".format(os.path.join(data_root, "train.tsv")))
        return self._create_examples(
            self._read_tsv(os.path.join(data_root, "train.tsv")), "train")

    def get_dev_examples(self, data_root):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_root, "dev.tsv")), "dev")

    def get_test_examples(self, data_root):
        """See base class."""
        return self._create_examples(
            self._read_tsv(os.path.join(data_root, "test.tsv")), "test")

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2"]

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            if i == 0:
                continue
            guid = "%s-%s" % (set_type, i)
            text_a = line[3].lower()
            text_b = line[4].lower()
            img_id = line[2]
            label = line[1]
            examples.append(
                MMInputExample(guid=guid, text_a=text_a, text_b=text_b, img_id=img_id, label=label))
        return examples


def convert_mm_examples_to_features(examples, label_list, max_seq_length, max_entity_length, tokenizer, crop_size,
                                    path_img):
    """Loads a data file into a list of `InputBatch`s."""

    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    count = 0

    transform = transforms.Compose([
        transforms.RandomCrop(crop_size),  # self.crop_size, by default it is set to be 224
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    sent_length_a = 0
    entity_length_b = 0
    total_length = 0
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = tokenizer.tokenize(example.text_b)
        if len(tokens_b) >= entity_length_b:
            entity_length_b = len(tokens_b)
        if len(tokens_a) >= sent_length_a:
            sent_length_a = len(tokens_a)

        if len(tokens_b) > max_entity_length - 2:
            s2_tokens = tokens_b[:(max_entity_length - 2)]
        else:
            s2_tokens = tokens_b
        s2_tokens = ["[CLS]"] + s2_tokens + ["[SEP]"]
        s2_segment_ids = [0] * len(s2_tokens)
        s2_input_ids = tokenizer.convert_tokens_to_ids(s2_tokens)
        s2_input_mask = [1] * len(s2_input_ids)

        # Zero-pad up to the sequence length.
        s2_padding = [0] * (max_entity_length - len(s2_input_ids))
        s2_input_ids += s2_padding
        s2_input_mask += s2_padding
        s2_segment_ids += s2_padding

        # Modifies `tokens_a` and `tokens_b` in place so that the total
        # length is less than the specified length.
        # Account for [CLS], [SEP], [SEP] with "- 3"
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambigiously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        if len(tokens) >= total_length:
            total_length = len(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        added_input_mask = [1] * (len(input_ids) + 49)  # 1 or 49 is for encoding regional image representations

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        added_input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        label_id = label_map[example.label]

        image_name = example.img_id
        image_path = os.path.join(path_img, image_name)

        if not os.path.exists(image_path):
            print(image_path)
        try:
            image = image_process(image_path, transform)
        except:
            count += 1
            # print('image has problem!')
            image_path_fail = os.path.join(path_img, '17_06_4705.jpg')
            image = image_process(image_path_fail, transform)

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: %s" % (example.guid))
            logger.info("tokens: %s" % " ".join(
                [str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("label: %s (id = %d)" % (example.label, label_id))

        features.append(
            MMInputFeatures(input_ids=input_ids, input_mask=input_mask, added_input_mask=added_input_mask,
                            segment_ids=segment_ids,
                            s2_input_ids=s2_input_ids, s2_input_mask=s2_input_mask, s2_segment_ids=s2_segment_ids,
                            img_feat=image,
                            label_id=label_id))

    print('the number of problematic samples: ' + str(count))
    print('the max length of sentence a: ' + str(sent_length_a + 2) + ' entity b: ' + str(entity_length_b + 2) + \
          ' total length: ' + str(total_length + 3))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


class TwitterDataset(Dataset):

    def __init__(self,
                 data_root,
                 image_root,
                 text_type,
                 image_type,
                 **kwself):
        super(TwitterDataset, self).__init__()

        self.data_root = data_root
        self.image_root = image_root
        self.text_type = text_type
        self.image_type = image_type
        self.bert_model = "bert-large-uncased"
        self.task_name = kwself.get('dataset', "twitter").lower()
        self.max_seq_length = kwself.get('max_len', -1)
        self.max_entity_length = 16
        self.do_lower_case = False
        self.batch_size = kwself.get('batch_size', 64)
        self.crop_size = 224

        self.processors = {
            "twitter2015": AbmsaProcessor,  # our twitter-2015 dataset
            "twitter": AbmsaProcessor  # our twitter-2017 dataset
        }
        self.num_labels_task = {
            "twitter2015": 3,  # our twitter-2015 dataset
            "twitter": 3  # our twitter-2017 dataset
        }

        if self.task_name not in self.processors:
            raise ValueError("Task not found: %s" % (self.task_name))
        processor = self.processors[self.task_name]()
        label_list = processor.get_labels()
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_model, do_lower_case=self.do_lower_case)
        train_examples = processor.get_train_examples(self.data_root)

        train_features = convert_mm_examples_to_features(
            train_examples, label_list, self.max_seq_length, self.max_entity_length, self.tokenizer, self.crop_size, self.image_root)

        self.train_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        # all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        # all_added_input_mask = torch.tensor([f.added_input_mask for f in train_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        # all_s2_input_ids = torch.tensor([f.s2_input_ids for f in train_features], dtype=torch.long)
        # all_s2_input_mask = torch.tensor([f.s2_input_mask for f in train_features], dtype=torch.long)
        # all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in train_features], dtype=torch.long)
        self.train_img_feats = torch.stack([f.img_feat for f in train_features])
        self.train_label_ids = torch.tensor([f.label_id for f in train_features], dtype=torch.long)
        self.train_data = TensorDataset(self.train_input_ids, self.train_img_feats, self.train_label_ids)
        self.train_onehot_lb = []
        for i in self.train_label_ids:
            if i == 0:
                self.train_onehot_lb.append([1, 0, 0])
            elif i == 1:
                self.train_onehot_lb.append([0, 1, 0])
            else:
                self.train_onehot_lb.append([0, 0, 1])
        # self.train_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, \
        #                            all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids,
        #                            all_img_feats, all_label_ids)
        # self.train_sampler = RandomSampler(self.train_data)
        # train_dataloader = DataLoader(self.train_data, sampler=self.train_sampler, batch_size=self.batch_size)

        eval_examples = processor.get_dev_examples(self.data_root)
        eval_features = convert_mm_examples_to_features(
            eval_examples, label_list, self.max_seq_length, self.max_entity_length, self.tokenizer, self.crop_size,
            self.image_root)

        self.eval_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        # all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        # all_added_input_mask = torch.tensor([f.added_input_mask for f in eval_features], dtype=torch.long)
        # all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        # all_s2_input_ids = torch.tensor([f.s2_input_ids for f in eval_features], dtype=torch.long)
        # all_s2_input_mask = torch.tensor([f.s2_input_mask for f in eval_features], dtype=torch.long)
        # all_s2_segment_ids = torch.tensor([f.s2_segment_ids for f in eval_features], dtype=torch.long)
        self.eval_img_feats = torch.stack([f.img_feat for f in eval_features])
        self.eval_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        self.eval_data = TensorDataset(self.eval_input_ids, self.eval_img_feats, self.eval_label_ids)
        self.eval_onehot_lb = []
        for i in self.eval_label_ids:
            if i == 0:
                self.eval_onehot_lb.append([1, 0, 0])
            elif i == 1:
                self.eval_onehot_lb.append([0, 1, 0])
            else:
                self.eval_onehot_lb.append([0, 0, 1])
        # self.eval_data = TensorDataset(all_input_ids, all_input_mask, all_added_input_mask, all_segment_ids, \
        #                           all_s2_input_ids, all_s2_input_mask, all_s2_segment_ids, \
        #                           all_img_feats, all_label_ids)
        # self.eval_sampler = SequentialSampler(self.eval_data)
        # eval_dataloader = DataLoader(self.eval_data, sampler=self.eval_sampler, batch_size=self.batch_size)

    def train_(self):
        self.input_ids = self.train_input_ids
        self.img_feats = self.train_img_feats
        self.label_ids = self.train_label_ids
        self.onehot_lb = self.train_onehot_lb
        return self

    def valid_(self):
        self.input_ids = self.eval_input_ids
        self.img_feats = self.eval_img_feats
        self.label_ids = self.eval_label_ids
        self.onehot_lb = self.eval_onehot_lb
        return self

    def test_(self):
        self.input_ids = self.eval_input_ids
        self.img_feats = self.eval_img_feats
        self.label_ids = self.eval_label_ids
        self.onehot_lb = self.eval_onehot_lb
        return self

    def __getitem__(self, idx):
        return {
            'image_feat': torch.FloatTensor(self.img_feats[idx]).to('cuda'),
            'text_token': torch.LongTensor(self.input_ids[idx]).to('cuda'),
            'label': torch.FloatTensor(self.onehot_lb[idx]).to('cuda'),
            'label_': self.label_ids[idx]
        }

    def __len__(self):
        return len(self.label_ids)

    @property
    def vocab_size(self):
        return self.tokenizer.vocab_size

    @property
    def word2idx(self):
        return self.tokenizer.vocab

    # @property
    # def num_labels(self):
    #     return self.reader.num_labels
    #
    # @property
    # def idx2word(self):
    #     return self.tokenizer.idx2word
    #
    # @property
    # def max_len(self):
    #     return self.tokenizer.max_len
