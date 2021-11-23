# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : generate_model.py
@Project  : NLP-Beginner
@Time     : 2021/11/22 6:10 下午
@Author   : Zhiheng Xi
"""
import pickle

import torch.nn as nn
import torch
import Task5.config as config
from Task5.char2sequence import Char2Sequence
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence

char2Seqence =  pickle.load(open("./models/char2sequence.pkl", "rb"))
class GenerateModel(nn.Module):
    def __init__(self,char2Seq,embedding_dim = config.embedding_dim,hidden_size = config.hidden_size,output_size = config.output_size,
                 dropout = config.dropout):
        super(GenerateModel, self).__init__()
        vocab_size = len(char2Seq)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = vocab_size
        # self.output_size = output_size
        self.embedding_dim = embedding_dim
        self.dropout= dropout
        self.ws = char2Seq


        self.embedding = nn.Embedding(vocab_size,self.embedding_dim,padding_idx=self.ws.PAD)
        # lstm
        self.num_layers = 1
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_size,num_layers=self.num_layers,batch_first=True)
        self.linear = nn.Linear(self.hidden_size,self.vocab_size)


        # gru
        # self.gru = nn.GRU(self.embedding_dim,self.hidden_size,dropout=self.dropout,batch_first=True)
        # self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self,input,seq_lengths,hidden=None):
        # lstm
        batch_size,sequence_length = input.size()
        if hidden is None:
            h_0 = input.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
            c_0 = input.data.new(self.num_layers, batch_size, self.hidden_size).fill_(0).float()
        else:
            h_0,c_0 = hidden

        # embeds shape(batch_size,sequence_length,embedding_dim)
        embeds = self.embedding(input)
        lengths = torch.tensor([])
        packed_embed = pack_padded_sequence(embeds,batch_first=True,enforce_sorted=False,lengths=seq_lengths)

        packed_output,hidden = self.lstm(packed_embed,(h_0,c_0))
        output,lengths = pad_packed_sequence(packed_output,batch_first=True)

        # #output shape(batch_size,sequence_length,hidden_dim(2 if bidirectional else 1))
        # output,hidden = self.lstm(embeds,(h_0,c_0))
        #
        # output shape (sequence_length *betch, vocab size) 把word压平了，因此是所有的word，每个word有embedding
        output = self.linear(output.reshape(sequence_length * batch_size, -1))
        return output,hidden



        # gru
        # # input [batch_size,seq_len]
        # input = self.embedding(input) # [batch_size,seq_len,embedding_size]
        #
        # h_0,_ = self.init_hidden_state(input.size(0))  # x的第一个维度大小为batch size  h_0 [1,batch_size,hidden_size]
        # out,h_n = self.gru(input,h_0) # out[batch_size,seq_len,hidden_size]?todo
        # output = self.out(out)#[batch_size,seq_len,out_size] 注意，返回的是所有的output，而不仅仅是h_n
        # return output,h_n



    # def generate(self,x,sent_num=4,max_len=15):
    #     """
    #     gru 生成 todo
    #     :param x: todo 形状是1*1吗
    #     :param sent_num:
    #     :param max_len:
    #     :return:
    #     """
    #     init_hidden = torch.zeros(1,1,self.hidden_size) #[1,1,hidden_size]
    #     output=[]
    #     hn = init_hidden
    #
    #     x = self.embedding(x)
    #     for i in range(sent_num):
    #         seq_out,hn = self.gru(x,hn)
    #         seq_out = seq_out[:,-1,:].unsqueeze(0) # 选取seq的最后一个输出,也就是最后一个timestep
    #         output.append(x[:,i,:].unsqueeze(1)) # 选取下一句的开头
    #
    #         for j in range(max_len):
    #             # 上一个time step的输出 找到概率最大的
    #             _,topi = self.out(seq_out).data.topk(1)
    #             topi = topi.item()
    #             xi_from_output = torch.zeros([1,1],dtype=int)
    #             # todo xi_from_output做embedding处理
    #             xi_from_output[0][0] = int(topi)# 最大概率的位置设为1
    #             xi_from_output = self.embedding(xi_from_output) # [1,1,embeding dim]
    #             output.append(xi_from_output)
    #             # 生成新的output和hn todo
    #             seq_out,hn = self.gru(xi_from_output,hn)
    #             if topi==self.ws.to_index("。"):
    #                 break
    #     return output

    def generate(self,start_words,char2seq:Char2Sequence=char2Seqence,prefix_words=None,max_generate_length=64):
        """
        给定开头几个词生成一首诗歌
        :param start_words: 给定开头词
        :param char2seq:
        :param prefix_words: 不是诗歌组成部分，用来生成诗歌意境
        :return:
        """
        result = list(start_words)
        start_words_length = len(start_words)
        # 第一个词为<S>
        input = torch.LongTensor([char2seq.to_index("<S>")]).view(1,1)

        hidden=None

        # 用来生成意境
        if prefix_words:
            # 第一个input: <S>,然后是后面用来生成意境的前缀词
            # 第一个hidden state是None，后面会自动生成
            for char in prefix_words:
                output,hidden = self.forward(input,seq_lengths=torch.Tensor([1]),hidden=hidden)
                # 生成新的输入，便于下一循环
                input = input.data.new([char2seq.to_index(char)]).view(1,1)

        # 生成完意境之后，生成诗词
        # 如果之前没有前缀，那么input就是<"S">,hidden是None
        # 如果之前有前缀生成意境了，那么input是前缀的最后一个词对应的index？  hidden是forward生成出来的了
        for i in range(max_generate_length):
            output,hidden = self.forward(input,seq_lengths=torch.Tensor([1]),hidden=hidden)
            if i<start_words_length: # 开头还没用完呢
                char = result[i]
                input = input.data.new([char2seq.to_index(char)]).view(1,1)

            else:
                # 开头词用完了
                # 找到可能性最大的次
                # output[1,vocab_size]
                top_index = output.data[0].topk(1) # keys(例如0.32),values（目标char位置）
                top_index = top_index[1][0].item()# todo
                char = char2seq.to_char(top_index)
                result.append(char)
                #生成新的输入
                input = input.data.new([top_index]).view(1,1)

            #  END
            if char=="<E>":
                del result[-1]
                break
        return "".join(result)


    def init_hidden_state(self, batch_size):
        h_0 = torch.zeros(1, batch_size, self.hidden_size)
        c_0 = torch.zeros(1, batch_size, self.hidden_size)
        return h_0, c_0




from torch.optim import Adam
import Task5.config as config
from Task5.dataset import train_dataloader

def train_one_epoch(epoch):
    model = GenerateModel(char2Seq=char2Seqence)
    optimizer = Adam(model.parameters(),lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    # model_path = "./models/generate_model.model"

    total_loss = 0
    model.train()
    for i,batch in enumerate(train_dataloader):
        seq_length = []
        for j in range(0,config.batch_size):
            """
            epoch0,idx:483,loss:4.03963
epoch0,idx:484,loss:3.89282
Traceback (most recent call last):
  File "/Users/woooodyy/Documents/DL/NLP-Beginner/Task5/generate_model.py", line 234, in <module>
    train(3)
  File "/Users/woooodyy/Documents/DL/NLP-Beginner/Task5/generate_model.py", line 231, in train
    train_one_epoch(i)
  File "/Users/woooodyy/Documents/DL/NLP-Beginner/Task5/generate_model.py", line 188, in train_one_epoch
    a = list(batch[j].data).index(torch.tensor(3))
IndexError: index 2 is out of bounds for dimension 0 with size 2
"""
            a = list(batch[j].data).index(torch.tensor(3))
            seq_length.append(a+1-1)# x要qu掉1
        seq_length = torch.tensor(seq_length)

        optimizer.zero_grad()
        sentence = batch
        x = sentence[:,:-1] #todo 最后一个不会生成y，，只作为输出，不作为输入
        y = sentence[:,1:]# 生成的y是从1开始的seqlen

        output,_ = model(x,seq_length) #[batch_size * seq_len-1,vocab size]
        # output = output.reshape(-1,output.shape[-1]) # [batch_size*(seq_len-1),vocab_size]
        y = y.flatten()
        loss = criterion(output,y)
        loss.backward()
        optimizer.step()
        total_loss+=loss.item()
        print("epoch{},idx:{},loss:{:.5f}".format(epoch,i,loss))
        if(i%10==0):
            torch.save(model.state_dict(), config.model_state_dict_path)
            torch.save(optimizer.state_dict(), config.optimizer_dict_path)
            print("saved")
    print("epoch{},total loss:{:.5f}".format(epoch,total_loss))


def test():
    model = GenerateModel(char2Seq=char2Seqence)
    optimizer = Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()
    # 载入模型
    optimizer.load_state_dict(torch.load(config.optimizer_dict_path))
    model.load_state_dict(torch.load(config.model_state_dict_path))
    model.eval()

    x = "三层阁上"
    # x = [char2Seqence.to_index(i) for i in list(x)]
    # x = [x]
    # x = torch.tensor(x)
    output= model.generate(x)
    print(output)


def train(epoch):
    for i in range(epoch):
        train_one_epoch(i)


# train(3)
test()