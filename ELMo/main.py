import torch
from typing import Optional, Tuple, List
import argparse


def config():
    argparses = argparse.ArgumentParser()
    argparses.add_argument()
    argparses.add_argument()
    _arg = argparses.parse_args()


class ELMo(torch.nn.Module):
    def __init__(self, numlayar):
        super(ELMo, self).__init__()
        # self.embeding=torch.nn.Embedding(50,64)
        # self.lstmmodels = torch.nn.ModuleList(torch.nn.LSTM() for _ in range(numlayar*2))
        # self.linear=torch.nn.Linear()
        self.numlay = numlayar

    def forward(self, input, mask):
        inputs = self.linear(input)
        for i in range(self.numlay):
            forwards, (ho, co) = self.lstmmodels[0](inputs)
            backwards, (hb, cb) = self.lstmmodels[0 + self.numlay](inputs)

    @classmethod
    def from_pretrained(cls, path):
        model = cls()
        model.load_state_dict(torch.load(path))

    @property
    def arg(self):
        return self.numlay

    @arg.setter
    def arg(self, value):
        self.numlay = value


elmo = ELMo(3)
elmo.arg = 30
print(elmo.numlay)
