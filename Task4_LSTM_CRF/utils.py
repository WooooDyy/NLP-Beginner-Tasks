# !/usr/bin/env python3
# _*_ coding:utf-8 _*_
"""
@File     : utils.py
@Project  : NLP-Beginner
@Time     : 2021/11/18 3:11 下午
@Author   : Zhiheng Xi
"""

import pandas as pd
import csv

def read_csv(file_path, col_list=None):
    if col_list is None:
        df = pd.read_csv(file_path, sep=',')
        return df
    else:
        df = pd.read_csv(file_path, sep=',')
        return df[col_list]

def tag_to_idx(tag):
    if tag=="I-LOC":
        return 0
    if tag== "I-MISC":
        return 1
    if tag=="I-ORG":
        return 2
    if tag=="I-PER":
        return 3
    if tag=="O":
        return 4
    if tag=="<BEGIN>":
        return 5
    if tag=="<END>":
        return 6

def idx_to_tag(idx):
    if idx == 0:
        return "I-LOC"
    if idx==1:
        return "I-MISC"
    if idx==2:
        return "I-ORG"
    if idx==3:
        return "I-PER"
    if idx==4:
        return "O"
    if idx==5:
        return "<BEGIN>"
    if idx==6:
        return "<END>"



# 需要将原始数据变成sentence,tags的格式
def process_raw_data(raw_path="./data/eng.train", target_path="./data/train.csv"):
    # raw = read_csv(raw_path)
    processed_data = []
    with open(raw_path, encoding='utf-8')as f:
        reader = csv.reader(f)
        words = []
        tags = []
        for row in reader:
            if len(row) == 0 and len(words) > 0:
                # 空行
                sentence_str = " ".join(words)
                tags_str = " ".join(tags)
                processed_data.append(
                    {
                        "sentence": sentence_str,
                        "tags": tags_str,
                    }
                )
                words = []
                tags = []
            else:
                if len(row)==0 :
                    continue
                else:
                    row = row[0].split(' ')
                    if row[0] == '-DOCSTART-':
                        # 第一行
                        continue
                    if len(row)<4:
                        continue
                    else:
                        words.append(row[0])
                        tags.append(str(tag_to_idx(row[3])))
    print(processed_data)

    with open(target_path, 'w') as csvfile:
        fieldnames = ['sentence', 'tags']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for pair in processed_data:
            writer.writerow(pair)


#
process_raw_data("./data/eng.train","./data/train.csv")
process_raw_data("./data/eng.testa","./data/test.csv")
