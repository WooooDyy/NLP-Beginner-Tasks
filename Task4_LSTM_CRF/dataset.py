# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : dataset.py
@Project  : NLP-Beginner
@Time     : 2021/11/18 3:11 下午
@Author   : Zhiheng Xi
"""
import math
import pickle

import jsonlines
import csv
import tqdm
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import re
import torch
import Task4_LSTM_CRF.utils as utils
import Task4_LSTM_CRF.config as config

from Task4_LSTM_CRF.word_sequence import Word2Sequence


ws = pickle.load(open("./models/ws.pkl", "rb"))

class CoNLLDataset(Dataset):
    def __init__(self,mode,debug):
        super(CoNLLDataset, self).__init__()
        self.mode = mode
        if self.mode=="train":
            self.data = utils.read_csv(config.train_data_path)
        else:
            self.data = utils.read_csv(config.test_data_path)
        self.tag = self.data["tags"]
        self.sentence = self.data["sentence"]
        self.length = len(self.tag)
        self.debug = config.debug
        if self.debug:
            self.tag = self.tag[:min(config.debug_dataset_size, len(self.tag))]
            self.sentence = self.sentence[:min(config.debug_dataset_size, len(self.tag))]
    def __getitem__(self, idx):
        return str(self.tag[idx]).strip().split(" "),str(self.sentence[idx]).strip().split(" ")
    def __len__(self):
        return len(self.tag)


def collate_fn(batch):
    batch = list(zip(*batch))
    tags = list(batch[0])[0]
    # TODO
    try:
        tags = [int(i) if i!='None' and i is not None else 4 for i in tags]
    except ValueError:
        return
    sentences = list(batch[1])
    sentences = [ws.transform(i) for i in sentences] # 要不要加[0],不要，因为还是要保留batch这个维度
    sentences = torch.tensor(sentences)
    del batch
    return tags,sentences

# 实例化
train_dataset = CoNLLDataset(mode="train", debug=config.debug)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=config.batch_size
                              , shuffle=True, collate_fn=collate_fn)

# for idx,(tags,sentences) in enumerate(train_dataloader):
#     print(idx)
#     print(tags)
#     print(sentences)