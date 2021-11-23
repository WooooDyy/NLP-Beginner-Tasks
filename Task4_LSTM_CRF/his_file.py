import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import random_split
import pandas as pd
import numpy as np
import random

torch.manual_seed(1)
data = []

f = open('./data/eng.train', 'r', encoding='utf-8')
f.readline()
line = f.readline()
phrase = []
token = []
while line:
    if line == '\n':
        if len(token) > 0:
            data.append([phrase, token])
            phrase = []
            token = []
    else:
        phrase.append(line.split()[0])
        token.append(line.split()[-1])
    line = f.readline()
data_len = len(data)  # 14986

word_to_ix = {}  # 给每个词分配index
ix_to_word = {}
label_to_ix = {}
ix_to_label = {}
word_set = set()
label_set = set()
for sent, toke in data:
    for word in sent:
        if word not in word_to_ix:
            ix_to_word[len(word_to_ix)] = word
            word_to_ix[word] = len(word_to_ix)
            word_set.add(word)
    for tokens in toke:
        if tokens not in label_to_ix:
            ix_to_label[len(label_to_ix)] = tokens
            label_to_ix[tokens] = len(label_to_ix)
            label_set.add(tokens)

unk = '<unk>'
ix_to_word[len(word_to_ix)] = unk
word_to_ix[unk] = len(word_to_ix)
word_set.add(unk)

START_TAG = "<START>"
STOP_TAG = "<STOP>"
ix_to_label[len(label_to_ix)] = START_TAG
label_to_ix[START_TAG] = len(label_to_ix)
label_set.add(START_TAG)
ix_to_label[len(label_to_ix)] = STOP_TAG
label_to_ix[STOP_TAG] = len(label_to_ix)
label_set.add(STOP_TAG)

train_len = int(0.8 * data_len)
test_len = data_len - train_len
train_data, test_data = random_split(data, [train_len, test_len])  # 分割数据集
# print(type(train_data))  # torch.utils.data.dataset.Subset
train_data = list(train_data)
test_data = list(test_data)

# 参数字典，方便成为调参侠
args = {
    'vocab_size': len(word_to_ix),  # 有多少词，embedding需要以此来生成词向量
    'embedding_size': 50,  # 每个词向量有几维（几个特征）
    'hidden_size': 16,
    'type_num': 5,  # 分类个数
    'train_batch_size': 100,  # int(train_len / 10),
    'dropout': 0.1
}

f = open('./data/glove.6B.50d.txt', 'r', encoding='utf-8')
line = f.readline()
glove_word2vec = {}
pretrained_vec = []
while line:
    line = line.split()
    word = line[0]
    if word in word_set:
        glove_word2vec[word] = [float(v) for v in line[1:]]
    line = f.readline()

unk_num = 0
for i in range(args['vocab_size']):
    if ix_to_word[i] in glove_word2vec:
        pretrained_vec.append(glove_word2vec[ix_to_word[i]])
    else:
        pretrained_vec.append(list(torch.randn(args['embedding_size'])))
        unk_num += 1

print(unk_num, args['vocab_size'])
print(len(label_set))
pretrained_vec = np.array(pretrained_vec)

train_len = int(int(train_len / args['train_batch_size']) * args['train_batch_size'])


def argmax(vec):
    _, idx = torch.max(vec, 1)
    return idx.item()


def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    # equals torch.log(torch.sum(torch.exp(vec))), avoid overflow?
    return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))


class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.tagset_size = len(tag_to_ix)

        self.word_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.word_embedding.weight.data.copy_(torch.from_numpy(pretrained_vec))
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

        self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

        # transitioning *to* i *from* j.
        self.transitions = nn.Parameter(
            torch.randn(self.tagset_size, self.tagset_size)
        )

        # These two statements enforce the constraint that we never transfer
        # to the start tag and we never transfer from the stop tag
        self.transitions.data[tag_to_ix[START_TAG], :] = -10000
        self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

        self.hidden = self.init_hidden()

    def init_hidden(self):
        # h_0 and c_0
        return (torch.randn(2, 1, self.hidden_dim // 2),
                torch.randn(2, 1, self.hidden_dim // 2))  # [num_layer * direction, batch_size, hidden_dim]

    def _forword_alg(self, feats):
        # compute the best route
        init_alphas = torch.full((1, self.tagset_size), -10000.)
        init_alphas[0][self.tag_to_ix[START_TAG]] = 0  # make the first iteration choose START_TAG

        forward_var = init_alphas

        for feat in feats:
            alphas_t = []
            feat = torch.squeeze(feat)
            for next_tag in range(self.tagset_size):
                # [1, tagset_size] from emission matrix, no need of expand?
                emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
                trans_score = self.transitions[next_tag].view(1, -1)  # [1, tagset_size], from transition matrix
                next_tag_var = forward_var + trans_score + emit_score
                alphas_t.append(log_sum_exp(next_tag_var).view(1))

            forward_var = torch.cat(alphas_t).view(1, -1)  # [1, tagset_size], t_column of the viterbi map

        # START_TAG and STOP_TAG are not in the emission matrix
        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)  # a single score number
        return alpha

    def _get_lstm_features(self, sentence):
        # BiLSTM + full connection layer
        self.hidden = self.init_hidden()
        embeds = self.word_embedding(sentence).view(len(sentence[0]), 1, -1)
        lstm_out, self.hidden = self.lstm(embeds, self.hidden)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats

    def _score_sentence(self, feats, tags):
        # gold score
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long).view(1, -1), tags], dim=1)
        tags = torch.squeeze(tags)
        for i, feat in enumerate(feats):
            feat = torch.squeeze(feat)
            score = score + self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
        return score

    def _viterbi_decode(self, feats):
        backpointers = []  # find route

        # equals to init_alphas
        init_vvars = torch.full((1, self.tagset_size), -10000.)
        init_vvars[0][self.tag_to_ix[START_TAG]] = 0

        forward_var = init_vvars

        for feat in feats:
            bptrs_t = []
            viterbivars_t = []  # equals to alphas_t

            for next_tag in range(self.tagset_size):
                next_tag_var = forward_var + self.transitions[next_tag]
                best_tag_id = argmax(next_tag_var)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))

            forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
            backpointers.append(bptrs_t)

        terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
        best_tag_id = argmax(terminal_var)
        path_score = terminal_var[0][best_tag_id]

        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id)

        start = best_path.pop()
        assert start == self.tag_to_ix[START_TAG]
        best_path.reverse()
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # negative log likelihood
        feats = self._get_lstm_features(sentence)
        forward_score = self._forword_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def forward(self, sentence):
        lstm_feats = self._get_lstm_features(sentence)
        score, tag_seq = self._viterbi_decode(lstm_feats)
        return score, tag_seq


model = BiLSTM_CRF(len(word_to_ix), label_to_ix, args['embedding_size'], args['hidden_size'])
loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.002)


def match(test_batch):
    acc = 0
    all_len = 0
    with torch.no_grad():
        for instance, label in test_batch:
            all_len += len(instance)
            phrase = [word_to_ix[word] for word in instance]
            token = [label_to_ix[word] for word in label]
            phrase = torch.LongTensor(phrase).view(1, -1)
            ans = model(phrase)

            ans = ans[1]
            for i in range(len(instance)):
                if ans[i] == token[i]:
                    acc += 1

    print('acc = %.6lf%%' % (acc / all_len * 100))


def train(batch_data, batch_size):
    model.zero_grad()
    for instance, label in batch_data:
        phrase = [word_to_ix[word] for word in instance]  # 要先把每个词转换为其对应的index
        token = [label_to_ix[word] for word in label]
        phrase = torch.LongTensor(phrase).view(1, -1)
        token = torch.LongTensor(token).view(1, -1)

        loss = model.neg_log_likelihood(phrase, token) / batch_size
        loss.backward()
    print('    loss = %.6lf' % loss)
    optimizer.step()


# match(test_data)
random.seed(6)
for epoch in range(10):
    print('now in epoch %d...' % epoch)
    random.shuffle(train_data)
    for i in range(0, train_len, args['train_batch_size']):
        train(train_data[i: i + args['train_batch_size']], args['train_batch_size'])
    match(test_data)

# for epoch in range(10):
#     print('now in epoch %d...' % epoch)
#     random.shuffle(train_data)
#     train(train_data, train_len)
#     match(test_data)

# accs = [1.9031, 82.6398, 84.1167, 86.1422, 89.0837, 90.8937, 92.1624, 93.1470, 93.9112, 94.5382, 94.9276]