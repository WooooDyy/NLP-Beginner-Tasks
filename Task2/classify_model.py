# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : classify_model.py
@Project  : NLP-Beginner
@Time     : 2021/11/3 3:11 下午
@Author   : Zhiheng Xi
@Contact_1: 1018940210@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/11/3 3:11 下午        1.0             None
"""

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
from config import ws,batch_size,debug,num_layer,hidden_size,embedding_dim,bidirectional,dropout,weights,num_of_class,train_dataloader,test_dataloader

class ClassifyModelLSTM(nn.Module):
    def __init__(self, max_len, ws=ws, num_of_class=num_of_class,num_layer=num_layer, hidden_size=hidden_size, embedding_dim=embedding_dim, bidirectional=bidirectional, dropout=dropout,weights=weights):
        super(ClassifyModelLSTM, self).__init__()
        self.ws = ws
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.bi_num = 2 if self.bidirectional else 1
        self.num_of_class = num_of_class

        self.embedding = nn.Embedding(len(self.ws),self.embedding_dim,padding_idx=self.ws.PAD) #[vocab size,200]
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_size,self.num_layer,bidirectional=self.bidirectional,
                            dropout=self.dropout)

        ## 使用全连接层，中间使用relu激活函数
        # 第一个全连接层的输入形状为hidden_size * bi_num
        # 也就是说lstm结束之后，取前向后向的最后一个hidden state拼接起来
        # 不确定这里要不要乘以self.num_layer,不用乘以layer，
        # 因为多层的lstm，只需要最后一层的结果就行了。lstm的hidden state输出是这样的：
        # 第一层正向的最终hidden state，第一层反向的最终hiddenstate，第二层正向最终的hidden state。。。
        self.fc = nn.Linear(self.hidden_size * self.bi_num,self.num_of_class)


    def forward(self,x):
        x = self.embedding(x)
        x = x.permute(1,0,2) # 进行轴交换
        h_0,c_0 = self.init_hidden_state(x.size(1))# x的第一个维度大小为batch size
        _,(h_n,c_n) = self.lstm(x,(h_0,c_0))
        # print(h_n.shape)
        # 只要最后一个timestep的结果，这里去掉多余的hidden state,注意要看是不是双向的
        if self.bidirectional:
            out = torch.cat([h_n[-2,:,:],h_n[-1,:,:]],dim=-1) # 注意，需要dim=-1，把两个拼接到最后一个维度上，所以最后一个维度大小为self.hidden_size * self.bi_num
        else:
            # out = torch.cat([h_n[-2, :, :], h_n[-1, :, :]])
            # out = h_n[-1,:,:]
            out = torch.cat([h_n[-1,:,:]],dim=-1)

        out = self.fc(out)
        res = F.log_softmax(out,dim=-1)
        return res



    def init_hidden_state(self,batch_size):
        h_0 = torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        c_0 = torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        return h_0,c_0



