# -*- coding: utf-8 -*-
import utils.fileprocess as fileprocess

total = fileprocess.read_json("/Users/genyanjin1/PycharmProjects/NLP-learning/processor/total.json")
errolabels = fileprocess.read_file("/Users/genyanjin1/PycharmProjects/NLP-learning/processor/scratch.json")
print(len(errolabels))
errolabels=set(errolabels)
print(len(errolabels))

print(len(total))

def filter_function(datas):
    if not datas["weizhi"]:
        data = datas["miaoshu"]
        for errolabel in errolabels:

            if data.find(errolabel)!=-1:
                return False
        return True

    else:
        return True


def filter_erro(datas):
    if not datas["weizhi"]:
        data = datas["miaoshu"]
        for errolabel in errolabels:
            if data.find(errolabel)!=-1:
                return True
        return False

    else:
        return False

def datasplit(datas,rite=[8,1,1],name=["train.json","dev.json","test.json"]):
    assert len(rite)== len(name)
    data=[]
    l=0
    for i in range(len(rite)):
        data=datas[l:int(len(datas)*sum(rite[0:i+1])/sum(rite))]
        l=l+len(data)
        fileprocess.write_json(data,name[i])
        print(len(data))

new_data = [i for i in filter(filter_function, total)]



print(len(new_data))
fileprocess.write_json(new_data, "new_data.json")

erro_data = [i for i in filter(filter_erro, total)]
print(len(erro_data))
fileprocess.write_json(erro_data, "erro_data.json")
datasplit(new_data)
