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


    def forward(self,input,seq_lengths,hidden=None):
        # lstm
        batch_size,sequence_length = input.size() # sequence_length用来确保生成和计算loss的时候，不同的长度，在pad_packed_sequence里面同样用
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
        output,lengths = pad_packed_sequence(packed_output,batch_first=True,total_length=sequence_length)

        # #output shape(batch_size,sequence_length,hidden_dim(2 if bidirectional else 1))
        # output,hidden = self.lstm(embeds,(h_0,c_0))
        #
        # output shape (sequence_length *betch, vocab size) 把word压平了，因此是所有的word，每个word有embedding
        output = self.linear(output.reshape(sequence_length * batch_size, -1))
        return output,hidden

    def generate(self,start_words,char2seq:Char2Sequence=char2Seqence,prefix_words=None,max_generate_length=24):
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

def train_one_epoch(epoch,model,optimizer):
    criterion = nn.CrossEntropyLoss()

    # model_path = "./models/generate_model.model"

    total_loss = 0
    model.train()
    for i,batch in enumerate(train_dataloader):
        seq_length = []
        for j in range(0,batch.shape[0]):
            try:
                a = list(batch[j].data).index(torch.tensor(3))
            except IndexError:
                print("index error")
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

    x = "空山新雨后，"
    # x = [char2Seqence.to_index(i) for i in list(x)]
    # x = [x]
    # x = torch.tensor(x)
    output= model.generate(x,prefix_words="秋冬",max_generate_length=24)
    print(output)


def train(epoch,load_model=True):
    model = GenerateModel(char2Seq=char2Seqence)
    optimizer = Adam(model.parameters(),lr=config.learning_rate)
    # 载入模型
    if load_model:
        optimizer.load_state_dict(torch.load(config.optimizer_dict_path))
        model.load_state_dict(torch.load(config.model_state_dict_path))
    for i in range(epoch):
        train_one_epoch(i,model,optimizer)


# train(1,load_model=False)

# train(30,load_model=True)
test()