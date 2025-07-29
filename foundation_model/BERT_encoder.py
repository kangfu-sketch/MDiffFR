import numpy as np
from transformers import BertTokenizer, BertModel, BertConfig
import torch

from utils import read_text

# load BERT model
bert_model_load = '/foundation_model/bert/'
config_bert = BertConfig.from_pretrained(bert_model_load, output_hidden_states=True)
bert_model = BertModel.from_pretrained(bert_model_load, config=config_bert)
tokenizer = BertTokenizer.from_pretrained(bert_model_load)

dataname = 'KU' # KU, Food, Dance, Movie

if dataname in {'KU', 'Food', 'Dance', 'Movie'}:
    # text
    text_path = '../data/' + dataname + '/' + f"{dataname}_item.csv"
    item_text_column, CN_text_column, EN_text_column = read_text(text_path)
    texts = [" "]*len(item_text_column)
    for i, item in enumerate(item_text_column):
        if isinstance(EN_text_column[i], str):
            if len(EN_text_column[i]) > 1:
                texts[item] = EN_text_column[i]

elif dataname in {'Citeulike'}:# if you use other datasets, please modify there.
    pass

encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')

with torch.no_grad():
    outputs = bert_model(**encoded_inputs)

cls_embeddings = outputs.last_hidden_state[:, 0, :]  # get [CLS] token

print(cls_embeddings.shape)
np.save(f'../data/{dataname}/item_features_768.npy', cls_embeddings)       # default dimension = 768


