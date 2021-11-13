# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : train_and_test.py
@Project  : NLP-Beginner
@Time     : 2021/11/3 7:00 下午
@Author   : Zhiheng Xi
@Contact_1: 1018940210@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/11/3 7:00 下午        1.0             None
"""
import pickle

from torch import optim
import  numpy as np

from classify_model import ClassifyModelLSTM
from config import num_of_class, hidden_size, lstm_model_state_dict_path, lstm_optimizer__state_dict_path, \
    test_dataloader, lstm_model_path
# from dataset import train_dataloader,test_dataloader
# from word_sequence import ws
from dataset import train_dataloader, eval_dataloader
from word_sequence import ws
import torch.nn.functional as F
import torch

def train_lstm():
    model = ClassifyModelLSTM(max_len=len(ws), ws=ws, num_of_class=num_of_class, hidden_size=hidden_size)
    model.train(mode=True)
    optimizer = optim.Adam(model.parameters())
    for idx, (target, input) in enumerate(train_dataloader):
        optimizer.zero_grad()
        output = model(input)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        # print(idx)
        if idx % 2 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                1, idx * len(input), len(train_dataloader.dataset),
                   100. * idx / len(train_dataloader), loss.item()))
        # if idx > 1000:
        #     break
        torch.save(model.state_dict(), lstm_model_state_dict_path)
        torch.save(optimizer.state_dict(), lstm_optimizer__state_dict_path)
        if idx > 30:
            break

    torch.save(model,lstm_model_path)

def eval_lstm():
    """
    todo 测试结果
    :return:
    """
    test_loss = 0
    correct = 0
    mode = False
    # 载入模型
    model = torch.load(lstm_model_path)
    dataloader = eval_dataloader
    with torch.no_grad():
        print(eval_dataloader)
        for idx, (target, input) in enumerate(eval_dataloader):
            output = model(input)
            test_loss = F.nll_loss(output, target, reduction="mean")
            pred = torch.max(output, dim=-1, keepdim=False)[-1]
            correct = pred.eq(target.data).sum()
            acc = 100. * pred.eq(target.data).cpu().numpy().mean()
            print('idx: {} Test set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(idx, test_loss, correct,
                                                                                            target.size(0), acc))
            # _, y_pre = torch.max(out, -1)
            # acc = torch.mean((torch.tensor(y_pre == target, dtype=torch.float)))
            # val_accs.append(acc)
            # if idx>500:
            #     break




if __name__ == '__main__':
    train_lstm()