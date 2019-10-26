import torch
import tqdm
import re
from torch.utils.data import DataLoader, Dataset
from utils.fileprocess import read_json, write_json

label2id = {"B": 0, "I": 1, "o": 2}
start = "START"
end = "END"
label2id[start] = len(label2id)
label2id[end] = len(label2id)


class Dataloder(object):
    def __init__(self, *arg, **kwargs):
        pass

    @classmethod
    def data_process(cls):
        data = []
        y = []
        source_data = read_json("NER/data/feiai_total.json")
        for key, value in (enumerate(source_data)):
            text = value["text"]
            label = len(text) * ["o"]
            assert len(label) == len(text)
            # print(len(label))

            for position in value["annotation"]:
                start = position["start"]
                end = position["end"]

                if end <= len(label) or start <= len(label):
                    label[start:end] = ["I"] * (end - start)
                    label[start] = "B"
                    assert len(label) == len(text)

            texts = re.split(r',|，|。|\?|!|！|？', text)
            lenghts = list(map(len, texts))
            len_start = 0
            assert len(label) == len(text)
            if (len(lenghts) - 1 + sum(lenghts)) != len(label):
                print(len(lenghts), sum(lenghts), len(label))

            for i, lenght in enumerate(lenghts):
                if lenght != 0:
                    data.append(texts[i])
                    y.append(label[len_start:len_start + lenght])
                    len_start = len_start + lenght + 1
        for key, value in zip(data, y):
            if "B" in value:
                #print(key, value)
                pass
        return data,y

def decode_answer(predicton):
    for text, label in zip(*predicton):
        answers=[]
        answer=[]
        i=0
        while i <=(len(label)):
            while i<(len(label)) and label[i]=="O":
                i+=1
            if  i<(len(label)) and label[i]=="B" :
                begin=i
                i += 1
                while i<(len(label)) and label[i]=="I"  :
                    i += 1
                    end=i

                print(text[begin:end])
            i=i+1

        #print(text,label)




    pass

class Dataset(Dataset):
    pass


if __name__ == "__main__":
    predicton=Dataloder.data_process()
    decode_answer(predicton)