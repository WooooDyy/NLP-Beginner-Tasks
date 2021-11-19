# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : esim_model.py
@Project  : NLP-Beginner
@Time     : 2021/11/10 7:11 下午
@Author   : Zhiheng Xi
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.optim as optim
import Task3.config as config


class InputEncoder(nn.Module):
    def __init__(self, ws=config.ws, num_layer=1,
                 hidden_size=config.hidden_size,
                 embedding_dim=config.embedding_dim, bidirectional=True,
                 dropout=config.dropout):
        super(InputEncoder, self).__init__()
        self.ws = ws
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.bi_num = 2 if self.bidirectional else 1

        # [vocab size,300]
        self.embedding = nn.Embedding(len(self.ws), self.embedding_dim, padding_idx=self.ws.PAD)
        # self.embedding.weight.data.copy_(self.embedding)
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_size,
                            self.num_layer,bidirectional=self.bidirectional
                            ,dropout=self.dropout,batch_first=True)
    def forward(self, input):
        tmp_input = input
        tmp_embedding = self.embedding

        input = self.embedding(input)

        if(torch.isnan(input).int().sum()>0):
            non_zero = (torch.isnan(input).int()).nonzero()
            print("embedding后出现nan")
        # x = x.permute(1,0,2)
        h_0,c_0 = self.init_hidden_state(input.size(0))# x的第一个维度大小为batch size
        out,(h_n,c_n) = self.lstm(input, (h_0, c_0))
        # [batch_size,seq_len(time_step),hidden_size*bi_num]
        return out

    # 输入给lstm的时候不能把batch 放在first
    """
    根据运行结果来看，设置batch first为true，只有输入input和输出output的batch会在第一维，
    hn和cn是不会变的。使用的时候要注意，会很容易弄混。
    还有就是，这里并没有提供h0和c0，如果需要提供h0和c0，也需要注意shape,batch不在first。
    """
    def init_hidden_state(self,batch_size):
        h_0 = torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        c_0 = torch.rand(self.num_layer*self.bi_num,batch_size,self.hidden_size)
        return h_0,c_0

class LocalInferenceModel(nn.Module):
    def __init__(self):
        super(LocalInferenceModel, self).__init__()
        self.softmax1 = nn.Softmax(dim=1)
        self.softmax2 = nn.Softmax(dim=2)

    def forward(self,premise,hypothesis):
        """
        :param premise: [batch_size,len_p,hidden_size*bi_num]
        :param hypothesis: [batch_size,len_h,hidden_size*bi_num]
        :return:
        """
        # [batch_size,len_p,len_h]
        e = torch.matmul(premise,hypothesis.transpose(1,2))

        # 公式12、公式13 ，

        # 1. 计算attention分布 ，注意顺序交错
        # [batch_size,len_p,len_h]
        p_score,h_score = self.softmax2(e),self.softmax1(e)

        # 2. attention加权平均
        # batch matrix multiply
        # (x,a,b).bmm((x,b,c)) = (x,a,c)
        # [batch_size,len_p,len_h] 乘以 [batch_size,len_h,hidden_size*2]
        p_ = p_score.bmm(hypothesis) # [batch_size,len_p,hidden_size * 2]
        # 把h_score变成与p_score一样的形状
        h_ = h_score.transpose(1,2).bmm(premise) # [batch_size, len_h,hidden_size * 2]

        # 公式14，15
        # [batch_size, len_p, hidden_size * 8 ]
        m_p = torch.cat((premise,p_,premise-p_,premise*p_),dim=2)
        m_h = torch.cat((hypothesis,h_,hypothesis-h_,hypothesis*h_),dim=2)
        return m_p,m_h

class CompositionLayer(nn.Module):
    def __init__(self,input_size,hidden_size, output_size,dropout):
        super(CompositionLayer,self).__init__()

        self.hidden_layer = nn.Linear(input_size,output_size)
        # 此时的embedding dim就是前面一个隐藏层输出的output_size
        self.lstm = nn.LSTM(output_size,hidden_size,bidirectional=True,batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        """

        :param x: [batch_size, len_p, hidden_size * 8 ]
        :return:
        """
        x = self.hidden_layer(x) # [batch_size, len, hidden_size  ]
        x = self.dropout(x)
        x,_ = self.lstm(x)
        return x # [batch_size, len, hidden_size*2(因为是双向的)]
class PoolingLayer(nn.Module):
    """
    max pooling，sum pooling
    """
    def __init__(self):
        super(PoolingLayer, self).__init__()
    def forward(self,x):
        """
        :param x:# [batch_size, len, hidden_size*2(因为是双向的)]
        :return:
        """
        # 在len这个维度上求平均，这个维度就消失了，相当于在这个维度垂直做映射
        v_ave = x.sum(1)/x.shape[1] # [batch_size, hidden_size * 2]

        v_max = x.max(1)[0]# [batch_size, hidden_size * 2]

        return torch.cat((v_ave,v_max),dim=-1) # # [batch_size, hidden_size * 4]
class InferenceCompositionLayer(nn.Module):
    def __init__(self,input_size, output_size, hidden_size,dropout=0.0):
        super(InferenceCompositionLayer, self).__init__()
        self.composition = CompositionLayer(input_size,output_size,hidden_size,dropout)
        self.pooling = PoolingLayer()

    def forward(self,m_p,m_h):
        """
        :param m_p: [batch_size, len_p, hidden_size * 8 ]
        :param m_h:
        :return:
        """
        # attention and some operations
        v_p, v_h = self.composition(m_p), self.composition(m_h)

        # pooling
        v_p, v_h = self.pooling(v_p), self.pooling(v_h)
        return torch.cat((v_p,v_h),dim=1) # [batch_size, hidden_size * 8]

class OutputLayer(nn.Module):
    def __init__(self,input_size,output_size,num_of_class,dropout=0.0):
        super(OutputLayer, self).__init__()

        # mlp layer
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size,output_size,nn.Tanh),
            nn.Linear(output_size,num_of_class)
        )

    def forward(self,x):
        """
        :param x: [batch_size, hidden_size * 8]
        :return:
        """
        return self.mlp(x)

class ESIMModel(nn.Module):
    def __init__(self,ws,embedding_dim,hidden_size,num_of_class,dropout):
        super(ESIMModel, self).__init__()
        self.input_encoder = InputEncoder(ws,1,hidden_size,embedding_dim,bidirectional=True,dropout=dropout)
        self.local_inference = LocalInferenceModel()
        self.inference_composition = InferenceCompositionLayer(input_size=hidden_size*8,output_size=hidden_size,hidden_size=hidden_size,dropout=dropout)
        self.output = OutputLayer(input_size=hidden_size*8,output_size=hidden_size,num_of_class=num_of_class,dropout=dropout)

    def forward(self,p,h):
        p_encoded = self.input_encoder(p)
        h_encoded = self.input_encoder(h)

        m_p,m_h = self.local_inference(p_encoded,h_encoded)

        v = self.inference_composition(m_p,m_h)
        out = self.output(v)
        return F.log_softmax(out,dim=1)








a = torch.randint(1,9,[3,3])
v = a.sum(0)/a.shape[0]
# print(v)