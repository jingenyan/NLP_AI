import transformers
import torch
tokenizer = transformers.BertTokenizer(vocab_file="/Users/genyanjin1/PycharmProjects/NLP-learning/bert/data/vocab_file")
model = transformers.BertModel.from_pretrained('bert-base-uncased')
input_ids = torch.tensor(tokenizer.encode("金根炎")).unsqueeze(0)  # Batch size 1
#outputs = model(input_ids)
#last_hidden_states = outputs[0]
device=torch.device("cuda")
model=torch.nn.DataParallel(model)
model.to(device)
model.load_state_dict(torch.load(vocab_file,  map_location=lambda storage, loc: storage.cuda(1)))