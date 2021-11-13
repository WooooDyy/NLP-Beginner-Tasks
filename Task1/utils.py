
import pandas as pd


def read_tsv(file_path, col_list:list):
    df = pd.read_csv(file_path, sep='\t')
    return df[col_list]

def word_extraction(sentence):
    words = sentence.strip().split(" ")
    return words