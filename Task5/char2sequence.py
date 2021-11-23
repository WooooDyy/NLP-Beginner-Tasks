# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : char2sequence.py
@Project  : NLP-Beginner
@Time     : 2021/11/22 5:00 下午
@Author   : Zhiheng Xi
"""
import pickle
import numpy as np
from tqdm import tqdm


class Char2Sequence():
    UNK_TAG = "UNK"
    PAD_TAG = "PAD"
    UNK = 0
    PAD = 1
    BEGIN_TAG = "<S>"
    END_TAG = "<E>"
    BEGIN = 2
    END = 3

    def __init__(self):
        self.dict = {
            self.UNK_TAG: self.UNK,
            self.PAD_TAG: self.PAD,
            self.BEGIN_TAG: self.BEGIN,
            self.END_TAG: self.END
        }
        self.fitted = False

    def to_index(self, char):
        """
        char => index
        :param char:
        :return:
        """
        assert self.fitted
        return self.dict.get(char, self.UNK)

    def to_char(self, index):
        """
        index => char
        :param index:
        :return:
        """
        assert self.fitted
        if index in self.inversed_dict:
            return self.inversed_dict[index]
        return self.UNK_TAG

    def __len__(self):
        return len(self.dict)

    def fit(self, sentences, min_count=1, max_count=None, max_feature=None):
        """
        这里是char级别
        :param sentences: [[word1,word2,word3...],[word1,word2,word3...]...]
        :param min_count: 最小出现的次数
        :param max_count: 最大出现的次数
        :param max_feature: 总词语的最大数量
        :return:
        """

        # 词频统计
        count = {}
        for sentence in sentences:
            for char in sentence:
                if char not in count:
                    count[char] = 0
                count[char] += 1

        # 比最小的数量大，比最大的数量小，这些才能需要
        if min_count is not None:
            count = {char: value for char, value in count.items() if value > min_count}
        if max_count is not None:
            count = {char: value for char, value in count.items() if value < max_count}

        # 限制词典大小
        if isinstance(max_feature, int):
            count = sorted(list(count.items()), lambda x: x[1])
            if max_feature is not None and len(count) > max_feature:
                count = count[-int(max_feature):]
            for w, _ in count:
                self.dict[w] = len(self.dict)
        else:
            for w in sorted(count.keys()):
                self.dict[w] = len(self.dict)
        # 完成构建词典
        self.fitted = True

        self.inversed_dict = dict(zip(self.dict.values(), self.dict.keys()))

    def transform(self, sentence, max_len):
        """
        把句子转化为向量
        注意要在开头加上2（start），结尾加上3（end）
        :return:
        """
        assert self.fitted
        r = [self.PAD] * max_len
        r[0] = 2 # 第一个是<S>
        if len(sentence)+2 > max_len:
            sentence = sentence[:max_len-2]
        try:
            r[len(sentence)+1] = 3 # 最后一个是<E>
        except IndexError:
            print()

        for index, char in enumerate(sentence):
            r[index+1] = self.to_index(char)
        return np.array(r, dtype=np.int64)

    def inverse_transform(self, indices):
        sentence = [self.PAD_TAG] * len(indices)
        for index, num in enumerate(indices):
            sentence[index] = self.to_char(num)
        return sentence


from Task5.utils import read_csv


def fit_save_word_sequence():
    ws = Char2Sequence()
    # train data
    data = read_csv("./data/train.csv", ["sentence"])
    raw_sentences = data["sentence"].values.tolist()
    sentences = []
    raw_sentences = raw_sentences

    # # test data
    # data = read_csv("./data/test.csv", ["sentence"])
    # raw_sentences_test = data["sentence"].values.tolist()
    # raw_sentences = raw_sentences+raw_sentences_test

    for sentence in tqdm(raw_sentences):
        if isinstance(sentence, str):
            x = list(sentence)
            x = ['<S>'] + x + ['<E>']
            sentences.append(x)
    ws.fit(sentences)
    pickle.dump(ws, open("./models/char2sequence.pkl", "wb"))
    return ws


fit_save_word_sequence()
