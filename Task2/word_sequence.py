# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : word_sequence.py
@Project  : NLP-Beginner
@Time     : 2021/11/3 2:38 下午
@Author   : Zhiheng Xi
@Contact_1: 1018940210@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/11/3 2:38 下午        1.0             None
构建词典，实现方法把桔子转化为数字序列和把数字序列转化为句子

"""
import numpy as np
from tqdm import tqdm


class Word2Sequence():
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    UNK = 0
    PAD = 1

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD
        }
        self.fitted = False
    def to_index(self,word):
        """
        word => index
        :param word:
        :return:
        """
        assert self.fitted
        return self.dict.get(word,self.UNK)
    def to_word(self,index):
        """
        index => word
        :param index:
        :return:
        """
        assert self.fitted
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    def __len__(self):
        return len(self.dict)

    def fit(self,sentences,min_count = 1,max_count=None,max_feature=None):
        """

        :param sentences: [[word1,word2,word3...],[word1,word2,word3...]...]
        :param min_count: 最小出现的次数
        :param max_count: 最大出现的次数
        :param max_feature: 总词语的最大数量
        :return:
        """

        # 词频统计
        count = {}
        for sentence in sentences:
            for word in sentence:
                if word not in count:
                    count[word] = 0
                count[word]+=1

        # 比最小的数量大，比最大的数量小，这些才能需要
        if min_count is not None:
            count = {word:value for word,value in count.items() if value>min_count}
        if max_count is not None:
            count = {word: value for word, value in count.items() if value < max_count}

        # 限制词典大小
        if isinstance(max_feature,int):
            count = sorted(list(count.items()),lambda x:x[1])
            if max_feature is not None and len(count)>max_feature:
                count = count[-int(max_feature):]
            for w,_ in count:
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)
        # 完成构建词典
        self.fitted = True

        self.inversed_dict = dict(zip(self.dict.values(),self.dict.keys()))

    def transform(self,sentence,max_len=None):
        """
        把句子转化为向量
        :return:
        """
        assert self.fitted
        if max_len is not None:
            r = [self.PAD]*max_len
        else:
            r = [self.PAD]*len(sentence)
        if max_len is not None and len(sentence)>max_len:
            sentence = sentence[:max_len]
        for index,word in enumerate(sentence):
            r[index] = self.to_index(word)
        return np.array(r,dtype=np.int64)
    def inverse_transform(self,indices):
        sentence = [self.PAD_TAG]*len(indices)
        for index,num in enumerate(indices):
            sentence[index] = self.to_word(num)
        return sentence

from utils import read_tsv
def fit_save_word_sequence():
    ws = Word2Sequence()
    raw_sentences = read_tsv("./data/train.tsv", ["Phrase"])
    raw_sentences = raw_sentences["Phrase"]
    sentences = []
    for sentence in tqdm(raw_sentences):
        x = sentence.strip().split(" ")
        sentences.append(x)
    ws.fit(sentences)
    return ws




ws = fit_save_word_sequence()
# ws.fit([
#     ["你", "好", "么"],
#     ["你", "好", "哦"]])
# print(ws.dict)
# print(ws.fitted)
# print(ws.transform(["你", "好", "嘛"]))
# print(ws.transform(["你", "好", "嘛"], max_len=10))
# print(ws.inverse_transform([2, 3, 2, 3]))
