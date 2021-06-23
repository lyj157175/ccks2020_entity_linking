import pandas as pd
import torch
from transformers import (
    DataProcessor,
    InputExample,
    BertTokenizer,
    glue_convert_examples_to_features
)


class EntityLinkingDataLoader(DataProcessor):
    """实体链指dataloader
    """
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)

    def get_train_examples(self, file_path, set_type='train'):
        return self._create_examples(self._read_tsv(file_path), set_type)

    def get_dev_examples(self, file_path, set_type='dev'):
        return self._create_examples(self._read_tsv(file_path), set_type)

    def get_test_examples(self, file_path, set_type='test'):
        return self._create_examples(self._read_tsv(file_path), set_type)

    def get_labels(self):
        return ['0', '1']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f'{set_type}-{i}'
            text_a = line[1] + ' ' + line[3]  # 'entity  short_text'
            text_b = line[5]                  # kb_text
            label = line[-1]                  # predict
            examples.append(InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label))
        return examples

    def create_feature(self, examples):
        pickle_file = 'link_loader_' + examples[0].guid.split('-')[0] + '.pkl'
        features = glue_convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=self.get_labels(),
            max_length=self.args.el_max_len,
            output_mode='classification')
        pd.to_pickle(features, self.args.feature_path + pickle_file)


    def generate_feature(self):
        train_examples = self.get_train_examples(self.args.tsv_path + 'link_train.tsv', 'train')
        dev_examples = self.get_dev_examples(self.args.tsv_path + 'link_dev.tsv', 'dev')
        test_examples = self.get_test_examples(self.args.tsv_path + 'link_test.tsv', 'test')

        for examples in [train_examples, dev_examples, test_examples]:
            self.create_feature(examples=examples)



class EntityTypingDataLoader(DataProcessor):
    """实体链接数据处理"""
    def __init__(self, args):
        self.args = args
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)

    def get_train_examples(self, file_path, set_type='train'):
        return self._create_examples(self._read_tsv(file_path), set_type)

    def get_dev_examples(self, file_path, set_type='dev'):
        return self._create_examples(self._read_tsv(file_path), set_type)

    def get_test_examples(self, file_path, set_type='test'):
        return self._create_examples(self._read_tsv(file_path), set_type)


    def get_labels(self):
        return self.args.pickle_data['idx_to_type']

    def _create_examples(self, lines, set_type):
        examples = []
        for i, line in enumerate(lines):
            if i == 0:
                continue
            guid = f'{set_type}-{i}'
            text_a = line[1]
            text_b = line[3]
            label = line[-1]
            examples.append(InputExample(
                guid=guid,
                text_a=text_a,
                text_b=text_b,
                label=label))
        return examples

    def create_feature(self, examples):
        pickle_file = 'type_loader_' + examples[0].guid.split('-')[0] + '.pkl'
        features = glue_convert_examples_to_features(
            examples,
            self.tokenizer,
            label_list=self.get_labels(),
            max_length=self.args.et_max_len,
            output_mode='classification')
        pd.to_pickle(features, self.args.feature_path + pickle_file)

    def generate_feature(self):
        train_examples = self.get_train_examples(self.args.tsv_path + 'type_train.tsv', 'train')
        dev_examples = self.get_dev_examples(self.args.tsv_path + 'type_dev.tsv', 'dev')
        test_examples = self.get_test_examples(self.args.tsv_path + 'type_test.tsv', 'test')

        for examples in [train_examples, dev_examples, test_examples]:
            self.create_feature(examples=examples)



def feature_to_loader(features, shuffle=False):
    dataset = torch.utils.data.TensorDataset(
        torch.LongTensor([f.input_ids for f in features]),
        torch.LongTensor([f.attention_mask for f in features]),
        torch.LongTensor([f.token_type_ids for f in features]),
        torch.LongTensor([f.label for f in features]))

    dataloader = torch.utils.data.DataLoader(
        dataset,
        shuffle=shuffle,
        batch_size=32,
        num_workers=2)
    return dataloader