# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : utils.py
@Project  : NLP-Beginner
@Time     : 2021/11/22 4:14 下午
@Author   : Zhiheng Xi
"""
import os
import pandas as pd


def transform_txt_to_csv(raw_path, target_path):
    with open(raw_path, encoding='utf-8') as f:
        lines = f.read().split("\n")

    lines = list(map(lambda x: x.replace("\n", ""), lines))
    df = pd.DataFrame()
    df['sentence'] = lines
    df.to_csv(target_path, index=False, encoding='utf_8_sig')

    print(lines)


def read_csv(file_path, col_list=None):
    if col_list is None:
        df = pd.read_csv(file_path, sep=',')
        return df
    else:
        df = pd.read_csv(file_path, sep=',')
        return df[col_list]


transform_txt_to_csv("./data/poetryFromTang.txt", "./data/train.csv")
print(read_csv("./data/train.csv"))