import torch
import torch.nn as nn
# from pytorch_pretrained_bert import BertModel, BertTokenizer
from transformers import BertModel, BertTokenizer

class BertForTHUCNews(nn.Module):

    def __init__(self, config):
        super(BertForTHUCNews, self).__init__()
        self.bert = BertModel.from_pretrained(config.path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_classes)

    def forward(self, input_ids, attention_mask):
        _, pooled = self.bert(input_ids, attention_mask=attention_mask,
                              return_dict=False)
        out = self.fc(pooled)
        return out