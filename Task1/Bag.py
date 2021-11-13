
import numpy as np
from utils import word_extraction



class Bag:
    def __init__(self,do_lower_case=False):
        self.vocabulary = {} # 每个单词对应自己的位置
        self.do_lower_case = do_lower_case



    def generate_vocabulary(self,sentences):
        for sentence in sentences:
            if self.do_lower_case:
                sentence = sentence.lower()
            words = word_extraction(sentence=sentence)
            for word in words:
                if word not in self.vocabulary:
                    self.vocabulary[word] = len(self.vocabulary)

    def generate_bow(self,sentences):
        self.generate_vocabulary(sentences)
        vocab_size = len(self.vocabulary)
        bow = np.zeros((len(sentences),vocab_size))
        for idx in range(0,len(sentences)):
            sentence = sentences[idx]
            if self.do_lower_case:
                sentence = sentence.lower()
            words = word_extraction(sentence)
            for word in words:
                bow[idx][self.vocabulary[word]]+=1
        return bow

