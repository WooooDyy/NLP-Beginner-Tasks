# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : config.py
@Project  : NLP-Beginner
@Time     : 2021/11/22 4:14 下午
@Author   : Zhiheng Xi
"""

train_path = "./data/train.csv"
test_path = "./data/test.csv"
model_state_dict_path = "./data/model_state_dict.model"
optimizer_dict_path = "./data/optimizer_dict.model"
debug_dataset_size = 64
debug = False
batch_size = 32
seq_len = 100

hidden_size = 128
output_size = 32
embedding_dim = 128
dropout = 0.2
learning_rate = 0.01