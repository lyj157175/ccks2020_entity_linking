import torch
import torch.nn as nn
from transformers import (
    BertConfig,
    BertTokenizer,
    BertForSequenceClassification)



class EntityTyping(nn.Module):
    """实体类型推断模型"""

    def __init__(self, args):
        super(EntityTyping, self).__init__()
        self.tokenizer = BertTokenizer.from_pretrained(args.pretrain_path)
        self.config = BertConfig.from_json_file(args.pretrain_path + 'config.json')
        self.config.num_labels = len(args.pickle_data['idx_to_type'])
        self.bert = BertForSequenceClassification.from_pretrained(args.pretrain_path + 'pytorch_model.bin', config=self.config)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.parameters(), lr=2e-5, eps=1e-8)

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.bert(input_ids=input_ids,
                         attention_mask=attention_mask,
                         token_type_ids=token_type_ids)[0]


