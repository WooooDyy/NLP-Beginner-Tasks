# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : lstm_crf_model.py
@Project  : NLP-Beginner
@Time     : 2021/11/18 3:11 下午
@Author   : Zhiheng Xi
"""

import torch
import torch.nn as nn
import torch.optim as optim
from Task4_LSTM_CRF.word_sequence import Word2Sequence
import Task4_LSTM_CRF.config as config
import pickle
import Task4_LSTM_CRF.utils as utils

ws = pickle.load(open("./models/ws.pkl", "rb"))


def log_sum_exp(state_matrix):
    """
    计算的时候有数值技巧，防止exp(999)溢出
    :param state_matrix:
    :return: 计算log sum exp，主要用来计算 到达当前frame时，每个状态的score，结果以向量形式呈现
    数值计算技巧解释见 https://gist.github.com/koyo922/9300e5afbec83cbb63ad104d6a224cf4#file-bilstm_crf-py-L12
    主要就是exp()中的加减，加上了exp之后，就变成了乘除，可以提取公因子vmax
    """
    vmax = state_matrix.max(dim=0, keepdim=True).values  # 找到每一列的最大值
    return (state_matrix - vmax).exp().sum(axis=0, keepdim=True).log() + vmax


class BiLSTM_CRF(nn.Module):
    def __init__(self, ws=ws, num_layer=1, hidden_size=config.hidden_size,
                 embedding_dim=config.embedding_dim, bidirectional=True,
                 dropout=config.dropout):
        """
        :param ws:
        :param num_layer:
        :param hidden_size: 期望的LSTM输出维度
        :param embedding_dim: 给BILSTM的词向量维度
        :param bidirectional:
        :param dropout:
        """
        super(BiLSTM_CRF, self).__init__()
        self.ws = ws
        self.num_layer = num_layer
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.bi_num = 2 if self.bidirectional else 1

        self.embedding = nn.Embedding(len(self.ws), self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_size, self.num_layer,
                            bidirectional=self.bidirectional,  batch_first=True)
        # 将输出降维到tag维度的空间
        self.hidden2tag = nn.Linear(hidden_size * 2, config.num_of_tag)
        # num_of_tag * num_of_tag的转移矩阵
        #  tag间的转移score矩阵，即CRF层参数; 注意这里的定义是未转置过的，即"i到j"的分数(而非"i来自j")
        self.transitions = nn.Parameter(torch.randn(config.num_of_tag, config.num_of_tag))

        # "BEGIN_TAG来自于?" 和 "?来自于END_TAG" 都是无意义的
        self.transitions.data[:, utils.tag_to_idx("<BEGIN>")] = self.transitions.data[utils.idx_to_tag("<END>"),
                                                                :] = -10000

    def neg_log_liklihood(self, words, tags):
        """
        负对数似然函数，作为loss
        :param words:
        :param tags:
        :return:
        """
        frames = self._get_lstm_emission_score(words) # [seq_len,num_of_tag]
        gold_score = self.score_sentence(frames, tags)  # 获取正确路径的分数，也就是分子，已经做过log了
        forward_score = self._forward_alg(frames) # 所有路径score之和
        # -(正确路径的分数 - 所有路径的分数和）;注意取负号 -log(a/b) = -[log(a) - log(b)] = log(b) - log(a)
        return forward_score-gold_score

    """
    根据运行结果来看，设置batch first为true，只有输入input和输出output的batch会在第一维，
    hn和cn是不会变的。使用的时候要注意，会很容易弄混。
    还有就是，这里并没有提供h0和c0，如果需要提供h0和c0，也需要注意shape,batch不在first。
    """

    def init_hidden_state(self, batch_size):
        h_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size)
        c_0 = torch.rand(self.num_layer * self.bi_num, batch_size, self.hidden_size)
        return h_0, c_0

    def _get_lstm_emission_score(self, words):
        """
        获得每一帧对应的发射分数，先喂给lstm，然后转换为num_of_tag长度
        :param words:输入单词
        :return:
        """

        input = self.embedding(torch.tensor(words,dtype=torch.long))
        # [1(即batch_size),seq_len,embedding_dim]
        h_0, c_0 = self.init_hidden_state(input.size(0))  # x的第一个维度大小为batch size

        out, (h_n, c_n) = self.lstm(input, (h_0, c_0))  # out: [batch_size(1),seq_len,hidden_size*bi_num]

        out = self.hidden2tag(out.squeeze(0)) # 把LSTM输出的隐状态张量去掉batch维(batch first,所以第一维度)，然后降维到tag空间. [seq_len,num_of_tag]
        # out = out.squeeze(0)
        return out

    def tags_to_tensor(self,tags):
        """
        将tag序列（已经是数字了）转为tensor
        :param tags:
        :return:
        """
        return torch.tensor(tags,dtype=torch.long)


    def score_sentence(self,frames,tags):
        """
        求取正确路径的分值

        index:      0   1   2   3   4   5   6
        frames:     F0  F1  F2  F3  F4
        tags:  <s>  Y0  Y1  Y2  Y3  Y4  <e>

        :param frames: [seq_len,num_of_tag] 整个序列的emit score
        :param tags: 正确的tag
        :return:
        """
        tags_tensor = self.tags_to_tensor([utils.tag_to_idx("<BEGIN>")]+tags) # 注意不要加<END>，结尾会处理
        score = torch.zeros(1)
        for idx,frame in enumerate(frames): # 沿途累加每一帧的转移和发射,frame长度比tags_tensor短1，所以idx+1
            try:
                score += self.transitions[tags_tensor[idx],tags_tensor[idx+1]]+frame[tags_tensor[idx+1]]
            except IndexError:
                continue
        return score+self.transitions[tags_tensor[-1],utils.tag_to_idx("<END>")] #加上到end的转移分数

    def _forward_alg(self,frames):
        """
        给定每一帧的emission score;
        按照当前的CRF层参数算出所有可能序列的分值和，用作概率归一化分母，即Z
        解释参考https://zhuanlan.zhihu.com/p/97676647
        :return:
        """
        # alpha[i][j]：到第i个位置，状态为j的分数，使用动态规划来求，另外，这里用了一些压缩动态规划技巧，减少空间消耗
        alpha = torch.full((1,config.num_of_tag),-10000.0)

        # TODO 下面这行不确定
        alpha[0][utils.tag_to_idx("<BEGIN>")] = 0.# 初始化分值分布. <BEGIN>是log(1)=0, 其他都是很小的值 "-10000",也就是只能到<BEGIN>
        for frame in frames:
            log_sum_exp(alpha.T + frame.unsqueeze(0)+self.transitions)
        # 最后转到EOS，发射分值为0，转移分值为列向量 self.transitions[:, [self.tag2ix[END_TAG]]]
        return log_sum_exp(alpha.T + 0 + self.transitions[:, [utils.tag_to_idx("<END>")]]).flatten()

    def _viterbi_decode(self,frames):
        """
        求最佳路径
        :param frames:
        :return:
        todo 计算最大分数的时候是否还要log sum exp？
        """
        backtrace = [] # 回溯路径，backtrace[i][j] := 第i帧到达j状态的所有路径中, 得分最高的那条在i-1帧是在哪里
        alpha = torch.full((1,config.num_of_tag),-10000.0)
        alpha[0][utils.tag_to_idx("<START>")] = 0
        for frame in frames:
            # 这里跟 _forward_alg()稍有不同: 需要求最优路径（而非一个总体分值）, 所以还要对smat求column_max
            state_matrix = alpha.T+frame.unsqueeze(0)+self.transitions
            # state_matrix = alpha.T+self.transitions
            backtrace.append(state_matrix.argmax(0)) # 每列的最大值，到达下一个时刻每个位置的最优来源
            alpha = log_sum_exp(state_matrix)  # 转移规律跟 _forward_alg()一样; 只不过转移之前拿smat求了一下回溯路径
            # alpha = state_matrix.argmax(0)  # 转移规律跟 _forward_alg()一样; 只不过转移之前拿smat求了一下回溯路径

        # 回溯路径
        state_matrix = alpha.T+0+self.transitions[:, [utils.tag_to_idx("<END>")]]
        # state_matrix = alpha.T+self.transitions[:, [utils.tag_to_idx("<END>")]]
        best_tag_id = state_matrix.flatten().argmax().item()
        best_path = [best_tag_id]
        for bptrs_t in reversed(backtrace[1:]):  # 从[1:]开始，去掉开头的 START_TAG
            best_tag_id = bptrs_t[best_tag_id].item() # 到达下一个位置的best_tag_id的最有可能的位置
            best_path.append(best_tag_id)

        return log_sum_exp(state_matrix).item(),best_path[::-1]

    def forward(self,words):
        """
        推断
        :param words:
        :return:
        """
        emission_scores = self._get_lstm_emission_score(words)  # 求出每一帧的发射矩阵
        return self._viterbi_decode(emission_scores)  # 采用已经训好的CRF层, 做维特比解码, 得到最优路径及其分数

