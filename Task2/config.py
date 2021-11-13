# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : config.py
@Project  : NLP-Beginner
@Time     : 2021/11/3 6:36 下午
@Author   : Zhiheng Xi
@Contact_1: 1018940210@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/11/3 6:36 下午        1.0             None
"""

test_data_path = "./data/test.tsv"
train_data_path = "./data/train.tsv"
lstm_model_state_dict_path = "./models/text_classify_lstm_state_dict_model.pkl"
lstm_optimizer__state_dict_path = "./models/text_classify_lstm_state_dict_optimizer.pkl"

lstm_model_path = "./models/text_classify_lstm_model.model"
lstm_optimizer__path = "./models/text_classify_lstm_optimizer.model"

batch_size = 10
debug = 0
num_layer = 2
hidden_size = 128
embedding_dim = 200
bidirectional = True
dropout = 0.5
weights = None
num_of_class = 5
from dataset import train_dataloader,test_dataloader,eval_dataloader
from word_sequence import ws
