##############################################################
#
# Custom_Dataset_Class.py
# This file contains the code to load and prepare the dataset
# for use by BERT.
# It does BERT features extraction
#
##############################################################

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, AdamW, BertConfig
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
import transformers
# get_linear_schedule_with_warmup
from transformers import RobertaTokenizer, BertTokenizer, RobertaModel, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time


class CustomDataset(Dataset):
    """ Make preprocecing, tokenization and transform
    dataset into pytorch DataLoader instance.

    """

    def __init__(self, tokenizer, max_len, chunk_len=200, approach="all", max_size_dataset=None, data=None, min_len=249):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.chunk_len = chunk_len
        self.approach = approach
        self.min_len = min_len
        self.max_size_dataset = max_size_dataset
        self.data, self.label = data['text'].values, data['labels'].values

    def __getitem__(self, idx):
        """  Return a single tokenized sample at a given positon [idx] from data"""
        input_ids_list = []
        attention_mask_list = []
        token_type_ids_list = []
        targets_list = []
#        print("idx:", idx)
        notes = self.data[idx]
 #       print(notes)
 #       print('# notes: ', len(notes))
        for text in notes:
            text = str(text)
#            print(self.label[idx])
            targets = np.array(self.label[idx], dtype=int)
            data = self.tokenizer.encode_plus(
                text,
                max_length=self.chunk_len,
                pad_to_max_length=True,
                add_special_tokens=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt')

            input_ids = data["input_ids"].reshape(-1)
            attention_mask = data["attention_mask"].reshape(-1)
            token_type_ids = data["token_type_ids"].reshape(-1)
            targets = torch.tensor(targets, dtype=torch.int)

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            token_type_ids_list.append(token_type_ids)
            targets_list.append(targets)
#        print('#notes: ', len(input_ids_list[0]))
        return ({
            'ids': input_ids_list,  # torch.tensor(ids, dtype=torch.long),
            # torch.tensor(mask, dtype=torch.long),
            'mask': attention_mask_list,
            # torch.tensor(token_type_ids, dtype=torch.long),
            'token_type_ids': token_type_ids_list,
            'targets': targets_list,
            'len': [torch.tensor(len(targets_list), dtype=torch.long)]
        })

    def __len__(self):
        """ Return data length """
        return self.label.shape[0]
