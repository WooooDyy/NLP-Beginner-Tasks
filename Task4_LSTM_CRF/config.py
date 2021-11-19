# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : config.py
@Project  : NLP-Beginner
@Time     : 2021/11/18 3:11 下午
@Author   : Zhiheng Xi
"""
import pickle

train_data_path = "./data/train.csv"
test_data_path = "./data/test.csv"
debug = True
from Task4_LSTM_CRF.word_sequence import Word2Sequence
# ws = pickle.load(open("./models/ws.pkl", "rb"))


batch_size = 1
num_layer = 2
hidden_size = 16
embedding_dim = 5
bidirectional = True
dropout = 0.5
weights = None
num_of_tag = 7
seq_len = 50
debug_dataset_size = 1000

learning_reate = 0.00001
# print(ws)
lstm_crf_model_state_dict_path="./models/lstm_crf_model_state_dict.model"
lstm_crf_optimizer_state_dict_path="./models/lstm_crf_optimizer_state_dict.model"
lstm_crf_model_path="./models/lstm_crf_model.model"