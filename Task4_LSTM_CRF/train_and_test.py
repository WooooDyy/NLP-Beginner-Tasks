# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : train_and_test.py
@Project  : NLP-Beginner
@Time     : 2021/11/19 1:21 上午
@Author   : Zhiheng Xi
"""
import pickle

from tqdm import tqdm

from Task4_LSTM_CRF.word_sequence import Word2Sequence
from torch import optim
import torch
import torch.nn.functional as F
import Task4_LSTM_CRF.config as config
from Task4_LSTM_CRF.dataset import train_dataloader
from lstm_crf_model import BiLSTM_CRF

# TODO 与其他任务训练过程的区别：loss损失函数的选取与构建、梯度回传

ws = pickle.load(open("./models/ws.pkl", "rb"))

model = BiLSTM_CRF()
optimizer = optim.SGD(model.parameters(), lr=config.learning_reate, weight_decay=1e-4)
# optimizer.load_state_dict(torch.load(config.lstm_crf_optimizer_state_dict_path))
# model.load_state_dict(torch.load(config.lstm_crf_model_state_dict_path))
def train_batch(batch_data,batch_size):
    model.train(mode=True)
    model.zero_grad()
    for tags,words in batch_data:
        loss = model.neg_log_liklihood(words,tags)/batch_size
        loss.backward()

    print('    loss = %.6lf' % loss)
    optimizer.step()


def train():
    #载入模型
    # optimizer.load_state_dict(torch.load(config.lstm_crf_optimizer_state_dict_path))
    # model.load_state_dict(torch.load(config.lstm_crf_model_state_dict_path))
    model.train(mode=True)
    batch_data = []
    for idx,(tags,words) in enumerate(train_dataloader):
        # print(idx)
        # print(words)
        # model.zero_grad()
        # loss = model.neg_log_liklihood(words,tags)# 前向求出负对数似然(loss); 然后回传梯度
        # loss.backward()
        # optimizer.step()
        batch_data.append([tags,words])
        if idx % 100 == 0:
            train_batch(batch_data,batch_size=len(batch_data))
            batch_data=[]
            torch.save(model.state_dict(), config.lstm_crf_model_state_dict_path)
            torch.save(optimizer.state_dict(), config.lstm_crf_optimizer_state_dict_path)
            torch.save(model, config.lstm_crf_model_path)
            print("saved")


def match(batch_data):
    acc = 0
    all_len = 0
    with torch.no_grad():
        for tags, words in batch_data:
            all_len+=len(words[0])
            ans = model(words)
            ans = ans[1]
            for i in range(len(words[0])):
                try:
                    if ans[i]==tags[i]:
                        acc+=1
                except IndexError:
                    continue
    print('acc = %.6lf%%' % (acc / all_len * 100))
    return acc/all_len * 100

def test():

    acc_all_len = 0
    batch_idx=0
    with torch.no_grad():
        batch_data = []
        for idx,(tags,words) in enumerate(train_dataloader):
            batch_data.append([tags, words])
            if idx%100==0:
                acc_all_len += match(batch_data)
                batch_data = []
                batch_idx+=1

    print("acc_all_len= %.6lf%%" %(acc_all_len/batch_idx))

for i in tqdm(range(50)):
    print("--------------------------------------------------------")
    print("epoch="+str(i))

    train()
    # todo 跑出来全是4 4 4
    test()

