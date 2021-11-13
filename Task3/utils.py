# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : utils.py
@Project  : NLP-Beginner
@Time     : 2021/11/10 3:24 下午
@Author   : Zhiheng Xi
@Contact_1: 1018940210@qq.com
@Software : PyCharm
@Last Modify Time      @Version     @Desciption
--------------------       --------        -----------
2021/11/10 3:24 下午        1.0             None
"""

import pandas as pd


def read_csv(file_path,col_list=None):
    if col_list is None:
        df = pd.read_csv(file_path)
        return df
    else:
        df = pd.read_csv(file_path, sep=',')
        return df[col_list]
