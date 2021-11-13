# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : main.py
@Project  : NLP-Beginner
@Time     : 2021/11/12 4:34 下午
@Author   : Zhiheng Xi
"""
from torch import optim
from dataset import train_dataloader
from esim_model import ESIMModel
import Task3.config  as config
import torch
import torch.nn.functional as F
from Task3.train_and_test import train_esim,eval_esim,train
if __name__ == '__main__':
    train(epoch=3)
    # eval_esim()



