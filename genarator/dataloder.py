from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch.nn as nn
import torch
from torch.optim.adam import Adam

all_letters = string.ascii_letters + " .,;'-"
LETTRR = len(all_letters) + 1


def findfiles(pathname):
    return glob.glob(pathname)


def read_line(file):
    lines = []
    with open(file, "r") as f:
        for line in f:
            lines.append(unicodeToAscii(line.strip()))

    return lines


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


def read_data():
    all_categories = []
    categorys_data = {}
    files = findfiles("/Users/genyanjin1/PycharmProjects/NLP-learning/genarator/data/names/*txt")

    for file in files:
        category = os.path.splitext(os.path.basename(file))[0]
        print(category)
        all_categories.append(category)
        lines = read_line(file)
        categorys_data[category] = lines
    print(len(categorys_data))
    return categorys_data, all_categories


categorys_data, all_categories = read_data()
NCATEGORY = len(categorys_data)


class RNN(nn.Module):
    def __init__(self, hidens_size, input_size, output_size, categories_size):
        super(RNN, self).__init__()
        self.hidden_size = hidens_size
        self.emdding_category = nn.Embedding(NCATEGORY, categories_size)
        self.emdding_input = nn.Embedding(LETTRR, input_size, padding_idx=LETTRR - 1)
        self.i2h = nn.Linear(input_size + categories_size + hidens_size, hidens_size)
        self.i2o = nn.Linear(input_size + categories_size + hidens_size, output_size)
        self.o2o = nn.Linear(hidens_size + output_size, output_size)
        self.drop = nn.Dropout(0.5)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        category = self.emdding_category(category)
        input = self.emdding_input(input)

        input_conbined = torch.cat((input, hidden), dim=1)
        # input_conbined=self.drop(input_conbined)
        hidden = self.i2h(input_conbined)
        output = self.i2o(input_conbined)
        # output=self.o2o(torch.cat((hidden,output),dim=1))
        # output=self.drop(output)
        output = self.softmax(output)
        return output, hidden

    def Hidden(self):
        return torch.zeros([1, self.hidden_size])


import random


def randomchoice(k):
    return k[random.randint(0, len(k) - 1)]


def randompair():
    category = randomchoice(all_categories)
    line = randomchoice(categorys_data[category])
    return category, line


print(randompair())


def line2tensor():
    category, line = randompair()
    category2tentor = torch.LongTensor([all_categories.index(category)])
    input2tensor = torch.LongTensor([all_letters.find(line[i]) for i in range(0, len(line))])
    input2tensor = input2tensor.view(-1, 1)
    letter2indexs = [all_letters.find(line[i]) for i in range(1, len(line))]

    letter2indexs.append(LETTRR - 1)
    target2tensor = torch.LongTensor(letter2indexs)
    target2tensor = target2tensor.view(-1, 1)
    return category2tentor, input2tensor, target2tensor


print(line2tensor())
rnn = RNN(128, 128, NCATEGORY, 0)
print(rnn.state_dict())
opetimizor = Adam(rnn.parameters(), lr=0.0001)
criterion = nn.NLLLoss()
import time


def save_model(model,path):
    torch.save(model.state_dict(),path)


def train(N_iter):
    acccount = 0
    bestacc = 0
    for iter in range(N_iter):
        hidden = rnn.Hidden()
        outout = torch.zeros(1, NCATEGORY)
        loss = 0
        category2tentor, input2tensor, target2tensor = line2tensor()
        starttime = time.time()

        for i in range(len(input2tensor)):
            input = input2tensor[i]
            target = target2tensor[i]
            output, hidden = rnn(category2tentor, input, hidden)
            outout += output

        L = criterion(output, category2tentor)
        loss = L
        opetimizor.zero_grad()
        loss.backward()
        opetimizor.step()
        output = torch.max(output, 1)[1]
        if output == category2tentor:
            acccount += 1

        if iter % 5000 == 0:
            if iter != 0:
                acc = acccount / 5000.0
                print("loss:{},time:{},iter:{},acc:{} ".format(loss / len(input2tensor), time.time() - starttime, iter,
                                                               acccount / 5000.0))
                if acc >= bestacc:
                    save_model(rnn,"rnn.pt")
                    bestacc=acc

                acccount = 0


train(1000000)

def evaluate():
    pass
