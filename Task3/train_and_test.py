# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : train_and_test.py
@Project  : NLP-Beginner
@Time     : 2021/11/13 1:11 上午
@Author   : Zhiheng Xi
"""
from torch import optim
from dataset import train_dataloader,test_dataloader
from esim_model import ESIMModel
import Task3.config as config
import torch
import torch.nn.functional as F

# model = torch.load(config.esim_model_path)
# model = ESIMModel(config.ws, config.embedding_dim, config.hidden_size, config.num_of_class, config.dropout)
def train_esim(epoch):
    model = ESIMModel(config.ws, config.embedding_dim, config.hidden_size, config.num_of_class, config.dropout)
    optimizer = optim.Adam(model.parameters(), lr=0.2)
    # 载入模型
    optimizer.load_state_dict(torch.load(config.esim_optimizer_state_dict_path))
    model.load_state_dict(torch.load(config.esim_model_state_dict_path))
    model.train(mode=True)
    for idx, (labels, sentence1s, sentence2s) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(sentence1s, sentence2s)
        # b = torch.argmax(output,dim=1)
        # 为了不让loss为负数，需要在模型最后一步做log softmax，即在softmax之后做log，而非softmax
        loss = F.nll_loss(output, labels)
        loss.backward()
        optimizer.step()

        pred = torch.max(output, dim=-1, keepdim=False)[-1]
        correct = pred.eq(labels.data).sum()
        if idx % 1 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, idx * len(sentence1s), len(train_dataloader.dataset),
                   100. * idx / len(train_dataloader), loss.item()))

        if idx % 5 == 0:
            torch.save(model.state_dict(), config.esim_model_state_dict_path)
            torch.save(optimizer.state_dict(), config.esim_optimizer_state_dict_path)
            torch.save(model, config.esim_model_path)
            print("saved")

def eval_esim():
    """
    test
    :return:
    """
    # model = ESIMModel(config.ws, config.embedding_dim, config.hidden_size, config.num_of_class, config.dropout)
    # model.load_state_dict(torch.load(config.esim_model_state_dict_path))
    model = torch.load(config.esim_model_path)
    # dataloader = test_dataloader
    dataloader = train_dataloader
    with torch.no_grad():
        print(dataloader)
        for idx, (labels, sentence1s, sentence2s) in enumerate(dataloader):
            output = model(sentence1s, sentence2s)
            test_loss = F.nll_loss(output, labels, reduction="mean")
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct = pred.eq(labels.data).sum()
            acc = 100. * pred.eq(labels.data).cpu().numpy().mean()
            print('idx: {} Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'
                  .format(idx, test_loss,correct,labels.size(0), acc))

# train_esim()
def train(epoch):
    for i in range(epoch):
        train_esim(i)
