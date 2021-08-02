##############################################################
#
# utils.py
# This file contains various functions that are applied in
# the training loops.
# They convert batch data into tensors, feed them to the models,
# compute the loss and propagate it.
#
##############################################################

import torch
from torch.autograd import Variable
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
from transformers import BertTokenizer, BertModel, AdamW
from transformers import get_linear_schedule_with_warmup
import time
from tqdm import tqdm

def my_collate1(batches):
    # return batches
    return [{key: torch.stack(value) for key, value in batch.items()} for batch in batches]


def loss_fun(outputs, targets):
    loss = nn.BCEWithLogitsLoss()
    return loss(outputs, targets.float())
    # return nn.BCEWithLogitsLoss()(outputs, targets)


def evaluate(target, predicted):
    true_label_mask = [1 if (np.argmax(x)-target[i]) ==
                       0 else 0 for i, x in enumerate(predicted)]
    nb_prediction = len(true_label_mask)
    true_prediction = sum(true_label_mask)
    false_prediction = nb_prediction-true_prediction
    accuracy = true_prediction/nb_prediction
    return{
        "accuracy": accuracy,
        "nb exemple": len(target),
        "true_prediction": true_prediction,
        "false_prediction": false_prediction,
    }


def train_loop_fun1(data, indices, model, optimizer, device, scheduler=None):
    model.train()
    t0 = time.time()
    losses = []

    train_indices = indices
    MINI_BATCH_SIZE = 16
    # 16, 16, ..
    # if > len(batch) ..
    for adm in range(len(train_indices)):
#        print("adm:", adm)
        batch = data[adm]
#        print("batch:", batch)
        batch_size = batch['len'][0].item()
        num_minibatch = (batch_size - 1)//MINI_BATCH_SIZE + 1

        hx = Variable(torch.zeros(1, 1, 100)) # [num_layers*num_directions, batch, hidden]
        cx = Variable(torch.zeros(1, 1, 100))

        for b in range(num_minibatch):
            flag = True # begin of adm, reset cell state
            if b == num_minibatch - 1:
#                minibatch = [b*MINI_BATCH_SIZE:]
#                print(batch["ids"][b*MINI_BATCH_SIZE:])
                ids = batch["ids"][b*MINI_BATCH_SIZE:]
                mask = batch["mask"][b*MINI_BATCH_SIZE:]
                token_type_ids = batch["token_type_ids"][b*MINI_BATCH_SIZE:]
                targets = batch["targets"][b*MINI_BATCH_SIZE:]
                lengt = batch['len'][b*MINI_BATCH_SIZE:]

            else:
                flag = False
#                minibatch = batch[b*MINI_BATCH_SIZE, (b+1)*MINI_BATCH_SIZE]
                ids = batch["ids"][b*MINI_BATCH_SIZE:(b+1)*MINI_BATCH_SIZE] # 16
                mask = batch["mask"][b*MINI_BATCH_SIZE:(b+1)*MINI_BATCH_SIZE]
                token_type_ids = batch["token_type_ids"][b*MINI_BATCH_SIZE:(b+1)*MINI_BATCH_SIZE]
                targets = batch["targets"][b*MINI_BATCH_SIZE:(b+1)*MINI_BATCH_SIZE]
                lengt = batch['len'][b*MINI_BATCH_SIZE:(b+1)*MINI_BATCH_SIZE]

            ids = torch.stack(ids, dim=0)
#            print(type(ids))
#            print(ids.size())
            mask = torch.stack(mask, dim=0)
            token_type_ids = torch.stack(token_type_ids, dim=0)
            targets = torch.stack(targets, dim=0)
#            lengt = torch.stack(lengt, dim=0)

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)
#            print("tar:", targets.size())

            optimizer.zero_grad()

            if not flag:
                hx = Variable(hx.detach())
                cx = Variable(cx.detach())
            cell_states = (hx, cx)

            outputs, (hx, cx) = model(ids=ids, mask=mask, token_type_ids=token_type_ids, cell_states=cell_states)
            loss = loss_fun(outputs, targets)
            loss.backward()
            model.float()
            optimizer.step()
            if scheduler:
                scheduler.step()
            losses.append(loss.item())
        if adm % 250 == 0:
            print(
                f"___ batch index = {adm} / {len(train_indices)} ({100*adm / len(train_indices):.2f}%), loss = {np.mean(losses[-10:]):.4f}, time = {time.time()-t0:.2f} secondes ___")
            t0 = time.time()
    return losses


def eval_loop_fun1(data, indices, model, device):
    model.eval()
    fin_targets = []
    fin_outputs = []
    losses = []

    eval_indices = indices
    MINI_BATCH_SIZE = 16

    for adm in range(len(eval_indices)):
#        print("adm:", adm)
        batch = data[adm]
#        print("batch:", batch)
        batch_size = batch['len'][0].item()
        num_minibatch = (batch_size - 1) // MINI_BATCH_SIZE + 1
        for b in range(num_minibatch):
            if b == num_minibatch - 1:
#                minibatch = [b*MINI_BATCH_SIZE:]
#                print(batch["ids"][b*MINI_BATCH_SIZE:])
                ids = batch["ids"][b * MINI_BATCH_SIZE:]
                mask = batch["mask"][b * MINI_BATCH_SIZE:]
                token_type_ids = batch["token_type_ids"][b * MINI_BATCH_SIZE:]
                targets = batch["targets"][b * MINI_BATCH_SIZE:]
                lengt = batch['len'][b * MINI_BATCH_SIZE:]

            else:
#                minibatch = batch[b*MINI_BATCH_SIZE, (b+1)*MINI_BATCH_SIZE]
                ids = batch["ids"][b * MINI_BATCH_SIZE:(b + 1) * MINI_BATCH_SIZE]  # 16
                mask = batch["mask"][b * MINI_BATCH_SIZE:(b + 1) * MINI_BATCH_SIZE]
                token_type_ids = batch["token_type_ids"][b * MINI_BATCH_SIZE:(b + 1) * MINI_BATCH_SIZE]
                targets = batch["targets"][b * MINI_BATCH_SIZE:(b + 1) * MINI_BATCH_SIZE]
                lengt = batch['len'][b * MINI_BATCH_SIZE:(b + 1) * MINI_BATCH_SIZE]

            ids = torch.stack(ids, dim=0)
#            print(type(ids))
#            print(ids.size())
            mask = torch.stack(mask, dim=0)
            token_type_ids = torch.stack(token_type_ids, dim=0)
            targets = torch.stack(targets, dim=0)
#            lengt = torch.stack(lengt, dim=0)

            ids = ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.long)

            with torch.no_grad():
                outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
                loss = loss_fun(outputs, targets)
                losses.append(loss.item())

            fin_targets.append(targets.cpu().detach().numpy())
            fin_outputs.append(torch.softmax(
                outputs, dim=1).cpu().detach().numpy())
    return np.concatenate(fin_outputs), np.concatenate(fin_targets), losses


