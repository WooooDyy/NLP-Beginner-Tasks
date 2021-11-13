
import numpy as np
from utils import word_extraction
class NGram:
    def __init__(self, ngram, do_lowe_case=False):
        self.ngram = ngram
        self.feature_voacbulary = {}
        self.do_lower_case = do_lowe_case

    def generate_vocabulary(self,sentences):
        for gram in self.ngram:
            for sentence in sentences:
                if self.do_lower_case:
                    sentence = sentence.lower()
                sentence = word_extraction(sentence)
                for i in range(len(sentence)-gram+1):
                    feature = "_".join(sentence[i:i+gram])
                    if feature not in self.feature_voacbulary:
                        self.feature_voacbulary[feature] = len(self.feature_voacbulary)

    def generate_ngram(self,sentences):
        self.generate_vocabulary(sentences)
        vocab_size = len(self.feature_voacbulary)
        ngram_bow = np.zeros((len(sentences),vocab_size))
        for idx in range(len(sentences)):
            sentence = sentences[idx]
            if self.do_lower_case:
                sentence = sentence.lower()
            sentence = word_extraction(sentence)
            for word in sentence:
                ngram_bow[idx][self.feature_voacbulary[word]]+=1
        return ngram_bow


