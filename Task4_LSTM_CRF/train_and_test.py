# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : train_and_test.py
@Project  : NLP-Beginner
@Time     : 2021/11/19 1:21 上午
@Author   : Zhiheng Xi
"""
import pickle
from Task4_LSTM_CRF.word_sequence import Word2Sequence
from torch import optim
import torch
import torch.nn.functional as F
import Task4_LSTM_CRF.config as config
from Task4_LSTM_CRF.dataset import train_dataloader
from lstm_crf_model import BiLSTM_CRF

# TODO 与其他任务训练过程的区别：loss损失函数的选取与构建、梯度回传

ws = pickle.load(open("./models/ws.pkl", "rb"))


def train():
    model = BiLSTM_CRF()
    optimizer = optim.SGD(model.parameters(), lr=config.learning_reate)

    # 载入模型
    # optimizer.load_state_dict(torch.load(config.lstm_crf_optimizer_state_dict_path))
    # model.load_state_dict(torch.load(config.lstm_crf_model_state_dict_path))

    model.train(mode=True)
    for idx,(tags,words) in enumerate(train_dataloader):
        print(idx)
        # print(words)
        model.zero_grad()
        loss = model.neg_log_liklihood(words,tags)# 前向求出负对数似然(loss); 然后回传梯度
        loss.backward()
        optimizer.step()
        if idx % 50 == 0:
            torch.save(model.state_dict(), config.lstm_crf_model_state_dict_path)
            torch.save(optimizer.state_dict(), config.lstm_crf_optimizer_state_dict_path)
            torch.save(model, config.lstm_crf_model_path)
            print("saved")
    # todo 跑出来全是4 4 4
    with torch.no_grad():
        for idx,(tags,words) in enumerate(train_dataloader):
            print("predict")
            print(model(words))
            print("true")
            print(tags)

train()