# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : dataset.py
@Project  : NLP-Beginner
@Time     : 2021/11/22 4:14 下午
@Author   : Zhiheng Xi
"""
from torch.utils.data import Dataset, DataLoader
import os
import re
import torch
import pickle
import Task5.utils as utils
import Task5.config as config
from Task5.char2sequence import Char2Sequence
class PoetryDataset(Dataset):
    def __init__(self,mode,debug):
        super(PoetryDataset, self).__init__()
        self.mode = mode
        if self.mode=="train":
            self.data = utils.read_csv(config.train_path)
        else:
            self.data = utils.read_csv(config.test_path)

        self.data = self.data["sentence"]
        self.length = len(self.data)
        self.debug= debug
        if self.debug:
            self.data = self.data[:min(config.debug_dataset_size, len(self.data))]
        print("共{}首唐诗".format(self.length))

    def __getitem__(self, idx):
        item = self.data[idx]
        return item

    def __len__(self):
        return len(self.data)

char2seq = pickle.load(open("./models/char2sequence.pkl", "rb"))

def collate_fn(batch):
    sentences = batch

    sentences = [char2seq.transform(i,config.seq_len) for i in sentences]
    sentences = torch.tensor(sentences)
    del batch
    return sentences


train_dataset = PoetryDataset(mode="train",debug=config.debug)
train_dataloader = DataLoader(dataset=train_dataset,batch_size=config.batch_size,shuffle=True,collate_fn=collate_fn)
# for idx,(sentence) in enumerate(train_dataloader):
#     print(sentence)


