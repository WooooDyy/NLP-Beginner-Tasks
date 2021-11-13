# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : config.py
@Project  : NLP-Beginner
@Time     : 2021/11/10 4:03 下午
@Author   : Zhiheng Xi
@Contact_1: 1018940210@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/11/10 4:03 下午        1.0             None
"""
import pickle

import torch

train_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"
debug = False
from Task3.word_sequence import fit_save_word_sequence
ws = pickle.load(open("./models/ws.pkl", "rb"))


batch_size = 1
num_layer = 2
hidden_size = 128
embedding_dim = 300
bidirectional = True
dropout = 0.5
weights = None
num_of_class = 3
seq_len = 50
debug_dataset_size = 10000

esim_model_state_dict_path = "./models/esim_model_state_dict.model"
esim_model_path = "./models/esim_model.model"
esim_optimizer_state_dict_path = "./models/esim_optimizer_state_dict.model"
esim_optimizer_path = "./models/esim_optimizer.model"
# esim_model = torch.load(esim_model_path)


learning_reate = 0.0004
