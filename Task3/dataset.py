# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : dataset.py
@Project  : NLP-Beginner
@Time     : 2021/11/10 3:20 下午
@Author   : Zhiheng Xi

1. 将jsonl文件中的gold_label(0),sentence1(5),sentence2(6)取出来，存到csv里面
"""
import math

import jsonlines
import csv
import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import re
import torch
import Task3.config as config
from Task3.utils import read_csv


def label_trans(label):
    if label == "entailment":
        return 2
    if label == "contradiction":
        return 0
    if label == "neutral":
        return 1


def transform_txt_to_csv(src_filepath, tgt_filepath):
    data = [("label", "sentence1", "sentence2")]
    with jsonlines.open(src_filepath) as reader:
        for obj in reader:
            if obj["gold_label"] is not None and obj["sentence1"] is not None and obj["sentence2"] is not None:
                new_obj = (
                    label_trans(obj["gold_label"].strip()),
                    obj["sentence1"].strip(),
                    obj["sentence2"].strip()
                )
                data.append(new_obj)
                print(obj)
                # break
    with open(tgt_filepath, 'w') as f:
        writer = csv.writer(f)
        for i in data:
            writer.writerow(i)


# tmp = transform_txt_to_csv("./data/snli_1.0_test.jsonl","./data/test.csv")
# tmp = transform_txt_to_csv("./data/snli_1.0_train.jsonl","./data/train.csv")
# tmp2 = read_csv("./data/train.csv")
# print(tmp2)

class SnliDataset(Dataset):
    def __init__(self, mode, debug):
        super(SnliDataset, self).__init__()
        self.mode = mode
        if self.mode=="train":

            self.data = read_csv(config.train_data_path)
        else:
            self.data = read_csv(config.test_data_path)
        self.label = self.data["label"]
        self.sentence1 = self.data["sentence1"]
        self.sentence2 = self.data["sentence2"]
        self.length = len(self.label)
        self.debug = config.debug
        if self.debug:
            self.label = self.label[:min(config.debug_dataset_size,len(self.label))]
            self.sentence1 = self.sentence1[:min(config.debug_dataset_size,len(self.label))]
            self.sentence2 = self.sentence2[:min(config.debug_dataset_size,len(self.label))]

    def __getitem__(self, idx):
        return self.label[idx], str(self.sentence1[idx]).strip().split(" "), str(self.sentence2[idx]).strip().split(" ")

    def __len__(self):
        return len(self.label)


def collate_fn(batch):
    batch = list(zip(*batch))
    try:

        labels = torch.tensor(batch[0], dtype=torch.long)
    except ValueError:
        print("出现了nan")
        for i in range(len(batch[0])):
            if math.isnan(batch[0][i]):
                # batch[0][i] = 0s
                tmp = list(batch[0])
                tmp[i] = 0
                batch[0] = tuple(tmp)

    l = list(batch[0])
    labels = torch.tensor(l, dtype=torch.long)
    # print(labels)
    sentence1s = batch[1]
    sentence2s = batch[2]
    print(sentence1s)
    print(sentence2s)
    sentence1s = torch.tensor([config.ws.transform(i, config.seq_len) for i in sentence1s])
    sentence2s = torch.tensor([config.ws.transform(i, config.seq_len) for i in sentence2s])

    del batch
    return labels, sentence1s, sentence2s


# 实例化
train_dataset = SnliDataset(mode="train", debug=config.debug)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size
                              , shuffle=True, collate_fn=collate_fn)

test_dataset = SnliDataset(mode="test", debug=config.debug)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=128
                              , shuffle=True, collate_fn=collate_fn)

# for idx, (labels, sentences1, sentences2) in enumerate(train_dataloader):
#     print("idx:", idx)
#     print("sentences1:", sentences1)
#     print("sentence2s:", sentences2)
#     print("labels:", labels)
#     break
ten = torch.Tensor([math.nan])
print(ten)
ten[0] = 0
print(ten)
ten = torch.tensor(ten,dtype=torch.long)
print(ten)