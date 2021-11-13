from sklearn.model_selection import train_test_split

from utils import read_tsv
from Bag import Bag
from NGram import NGram
from softmax_regression import SoftmaxRegression
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    debug = 1

    # prepare data
    data = read_tsv("./data/train.tsv",["Phrase","Sentiment"])
    X_train = data["Phrase"]
    y_train = data["Sentiment"]

    if debug==1:
        X_train = X_train[:1000]
        y_train = y_train[:1000]

    y = np.array(y_train).reshape(len(y_train),1)

    # bag of word
    bow_object = Bag(do_lower_case=True)
    X_Bow = bow_object.generate_bow(X_train)

    print("Bow is : ", X_Bow.shape)

    # 交叉验证
    X_train_Bow, X_test_Bow, y_train_Bow, y_test_Bow = train_test_split(X_Bow, y, test_size=0.2, random_state=42, stratify=y)

    # train
    epoc = 100
    bow_learning_rate = 1

    # 梯度下降
    bow_model = SoftmaxRegression()
    history = bow_model.do_fit(X_train_Bow, y_train_Bow, epoch=epoc, learning_rate=bow_learning_rate, print_loss_steps=epoc // 10, update_strategy="stochastic")
    plt.plot(np.arange(len(history)),np.array(history))
    plt.show()
    print("Bow train {} test {}".format(bow_model.score(X_train_Bow, y_train_Bow), bow_model.score(X_test_Bow, y_test_Bow)))

    # ngram
    ngram_object = NGram((1, 2), True)
    X_NGram = ngram_object.generate_ngram(X_train)
    print("NGram is: ",X_NGram.shape)
    X_train_NGram, X_test_NGram, y_train_NGram, y_test_NGram = train_test_split(X_NGram, y, test_size=0.2, random_state=42, stratify=y)
    epoc=100
    ngram_learning_rate = 1
    ngram_model = SoftmaxRegression()
    history = ngram_model.do_fit(X_train_NGram, y_train_NGram, epoch=epoc, learning_rate=bow_learning_rate, print_loss_steps=epoc // 10, update_strategy="stochastic")
    plt.plot(np.arange(len(history)),np.array(history))
    plt.show()
    print("NGram train {} test {}".format(ngram_model.score(X_train_NGram, y_train_NGram), ngram_model.score(X_test_NGram, y_test_NGram)))
