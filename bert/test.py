from transformers import BertConfig,BertTokenizer,BertModel
import torch
config= BertConfig()
tokenizer=BertTokenizer(vocab_file="/Users/genyanjin1/PycharmProjects/NLP-learning/bert/data/vocab_file")
bertModel= BertModel(config)
data=torch.tensor([tokenizer.encode("[CLS]金根杨ing,read[SEP]")]*2)
out=bertModel(data)
print(tokenizer.tokenize("[CLS]金,根杨[SEP]"))