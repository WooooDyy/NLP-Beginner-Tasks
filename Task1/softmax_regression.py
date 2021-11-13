import numpy as np


def softmax(z):
    # z有多行，这里的目标是对每一行做softmax
    z -= np.max(z, axis=1, keepdims=True)
    z = np.exp(z)
    z /= np.sum(z, axis=1, keepdims=True)
    return z


class SoftmaxRegression:
    def __init__(self):
        self.num_of_class = None
        self.n = None  # 数据的个数
        self.m = None  # 数据维度,即有多少种数据
        self.weight = None  # 模型权重 shape(类别数，数据维度) 一行代表1个类别的各个维度的权重
        self.learning_rate = None

    def do_fit(self, X, y, learning_rate=0.01, epoch=10, num_of_class=5, print_loss_steps=-1, update_strategy="batch"):
        self.n, self.m = X.shape
        self.num_of_class = num_of_class
        # 先随机各个类别的数据的权重
        self.weight = np.random.randn(self.num_of_class,self.m)
        self.learning_rate = learning_rate

        # 将y换为独热码矩阵，每一行独热码表示一个label，仅在该label上为1
        y_one_hot = np.zeros((self.n,self.num_of_class))
        for i in range(self.n):
            y_one_hot[i][y[i]] = 1

        loss_history = []

        for e in range(epoch):
            # 每吃进一个X样本，就去改动weight
            # X(n, m)  weight(C, m)
            loss = 0
            # 随机梯度下降
            if update_strategy == "stochastic":
                # 打乱吃进样本的顺序
                rand_index = np.arange(len(X))
                np.random.shuffle(rand_index)
                for index in list(rand_index):
                    # 取出一个样本，变成一个行向量
                    Xi = X[index].reshape(1,-1)
                    # 行向量 点乘 weight的转置(m*class_num), 结果是一个长度为class_num 的行向量，代表了符合各个class的probability
                    prob = Xi.dot(self.weight.T)
                    prob = softmax(prob).flatten()
                    # y[index]代表应得的sentiment值，prob[y[index]]代表预测的该sentiment值的可能性，即y_hat
                    # 而 true_prob[y[index]]为1，所以略去
                    loss += -np.log(prob[y[index]])
                    # 梯度下降 x[i].T 点乘 (y-y_hat) ，得到的结果再转置
                    self.weight += (Xi.reshape(1, self.m).T.dot((y_one_hot[index]-prob).reshape(1,self.num_of_class))).T
            elif update_strategy=="batch":
                # 直接求softmax
                prob = X.dot(self.weight.T)
                prob = softmax(prob)

                weight_update = np.zeros_like(self.weight)
                for i in range(self.n):
                    weight_update += X[i].reshape(1,self.m).T.dot((y_one_hot[i]-prob[i]).reshape(1,self.num_of_class)).T
                self.weight += weight_update*self.learning_rate/self.n

            loss /= self.n
            loss_history.append(loss)
            if print_loss_steps != -1 and e % print_loss_steps == 0:
                print("epoch {} loss {}".format(e, loss))
        return loss_history

    def predict(self,X):
        prob = softmax(X.dot(self.weight.T))
        # argmax(axis=1): 根据第2维切开，比如原来矩阵为(800,5),切开之后就是一维的，800个数字，表示原始矩阵第二维度每个最大的值的位置
        return prob.argmax(axis=1)

    def score(self, X, y):
        pred = self.predict(X)
        # np.sum(pred.reshape(y.shape) == y)得出判断正确的y的个数
        return np.sum(pred.reshape(y.shape) == y) / y.shape[0]







