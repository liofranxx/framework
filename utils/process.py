'''
Descripttion: 
Author: Lyc
Date: 2022-11-08 09:27:20
LastEditors: Lyc
LastEditTime: 2022-11-09 13:57:07
'''
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class_list = ['finance','realty','stocks','education','science','society','politics','sports','game','entertainment']
id2label = {idx: label for idx, label in enumerate(class_list)}
label2id = {label: idx for idx, label in enumerate(class_list)}


def load_data(path):
    file = open(path, 'r', encoding='utf-8')
    text = []
    label = []
    for line in file:
        data = line.strip().split("\t")
        assert len(data) == 2, "数据格式错误"
        text.append(data[0])
        label.append(int(data[1]))
    # print(data)
    return text, label


class DatasetForBert(Dataset):
    def __init__(self, path, config):
        self.texts, self.labels = load_data(path)
        self.max_len = 32
        assert len(self.texts) == len(self.labels), "数据读取错误"
        self.tokenizer = AutoTokenizer.from_pretrained(config.path)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        item = {}

        encodings = self.tokenizer(text, truncation=True,
                                   padding='max_length', max_length=self.max_len)

        input_ids = torch.LongTensor(encodings['input_ids'])
        attention_mask = torch.LongTensor(encodings['attention_mask'])
        label = torch.tensor(label)

        item['input_ids'] = input_ids
        item['attention_mask'] = attention_mask

        return item, label

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    file_path = '/home/liyongchun/project/experiment/Bert_THUNCNews/THUCNews/data/train.txt'
    text, label = load_data(file_path)
    print(len(text))
    # config = Config('THUCNews')
    # train_data = DatasetForBert(file_path, config)
    # item, label = train_data.__getitem__(2)
    # print(item['input_ids'])
