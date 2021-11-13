# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : dataset.py
@Project  : NLP-Beginner
@Time     : 2021/11/3 2:15 下午
@Author   : Zhiheng Xi
@Contact_1: 1018940210@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/11/3 2:15 下午        1.0             None

准备数据集
"""

import torch
from torch.utils.data import Dataset, DataLoader
import os
import re
from utils import read_tsv, word_extraction
from word_sequence import ws
from config import batch_size, debug, test_data_path, train_data_path


class SentimentDataset(Dataset):
    def __init__(self, mode, debug):
        super(SentimentDataset, self).__init__()
        self.mode = mode
        self.data = read_tsv(train_data_path, ["Phrase", "Sentiment"])
        if mode != "test":
            self.data = read_tsv(train_data_path, ["Phrase", "Sentiment"])
            self.label = self.data["Sentiment"]
            self.text = self.data["Phrase"]
            self.length = len(self.label)
            train_len = self.length * 0.7
            if mode == "train":
                self.label = self.label[:int(train_len)]
                self.text  = self.text[:int(train_len)]
                self.debug = debug
                if (self.debug == 1):
                    self.label = self.label[:1000]
                    self.text = self.text[:1000]
            elif mode=="eval":
                self.label = self.label[int(train_len)+1:]
                self.text = self.text[int(train_len)+1:]
                self.debug = debug
                if (self.debug == 1):
                    self.label = self.label[:1000]
                    self.text = self.text[:1000]
        else:
            self.data = read_tsv(test_data_path, ["Phrase"])
            self.text = self.data["Phrase"]
            self.debug = debug
            if (self.debug == 1):
                self.text = self.text[:1000]

    def __getitem__(self, idx):
        if self.mode == "train" :
            return self.label[idx], self.text[idx].strip().split(" ")
        elif self.mode=="eval":
            # 因为分训练集和测试集的时候，label分好了，分成了[(index,value)]列表，因此在获取的时候需要加上length。
            # 比如我获取idx为0的时候，其实index不是0，而是idx+int(self.length*0.7
            i = int(idx + int(self.length * 0.7))+1
            y = self.label[i]
            x = self.text[i].strip().split(" ")
            return y, x
        else:
            return self.text[idx].strip().split(" ")

    def __len__(self):
        return len(self.text)


def collate_fn_train(batch):
    batch = list(zip(*batch))
    labels = torch.tensor(batch[0], dtype=torch.long)
    texts = batch[1]
    texts = torch.tensor([ws.transform(i, 500) for i in texts])
    length = len(texts)

    del batch
    return labels, texts


def collate_fn_test(batch):
    texts = torch.tensor([ws.transform(i, 500) for i in batch])
    del batch
    return texts


# 实例化
train_dataset = SentimentDataset(mode="train", debug=debug)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)

eval_dataset = SentimentDataset(mode="eval", debug=debug)
eval_dataloader = DataLoader(dataset=eval_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn_train)


test_dataset = SentimentDataset(mode="test", debug=debug)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False
                            , collate_fn=collate_fn_test)


for idx, (label, text) in enumerate(train_dataloader):
    print("idx:", idx)
    print("text:", text)
    print("label:", label)
    break

for idx, (text) in enumerate(test_dataloader):
    print("idx:", idx)
    print("text:", text)
    break
