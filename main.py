# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 20:04:08 2020



@author: hoang
"""

import time
import torch
import torchtext
import math
from newton_scraping import newton_scraping
from torchtext.data.utils import get_tokenizer
from batching import batchify, get_batch, get_txt
from model_v2 import transformer_model
import torch.nn as nn

# Hardware for torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Scraping
df = newton_scraping()

# Dataset split
train_size = int(0.6*len(df))
valid_size = int(0.2*len(df))
test_size = len(df) - train_size - valid_size
df_train, df_valid, df_test = torch.utils.data.random_split(df, [train_size, valid_size, test_size])

# Batching params
bptt = 35   # batch seqeunce length
batch_size = 16
eval_batch_size = 8


# Create torch data objects
train_txt = get_txt(df_train['full_text'])
val_txt = get_txt(df_valid['full_test'])
test_txt = get_txt(df_test['full_text'])

TEXT = torchtext.data.Field(tokenize=get_tokenizer("spacy"),
                            init_token="<sos>",
                            eos_token="<eos>",
                            lower=True,
                            include_lengths=True)
TEXT.build_vocab(train_txt)

# Create batches
train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)


# Model hyperparams
n_tokens = len(TEXT.vocab.stoi)
emb_size = 200          # embedding dimensions
nhid = 200          # dimensions of ff network
n_layers = 2            # number of encoder layers
n_head = 2          # number of attention heads
dropout = 0.4

model = transformer_model(n_tokens, emb_size, nhid, n_layers, n_head, dropout)

# Training hyperparams
criterion = nn.CrossEntropyLoss()
lr = 2
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.95)

# Training loop
def train():
    # initialize learning
    model.train()           # training mode on
    total_loss = 0
    start_time = time.time()
    n_tokens = len(TEXT.vocab.stoi)
    
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        # 1 batch pass
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, n_tokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        
        # update loss
        total_loss += loss.item()
        log_interval = 200
        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f}'.format(
                      epoch, batch, len(train_data) // bptt, scheduler.get_lr()[0],
                      elapsed * 1000 / log_interval,
                      cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

# Evaluation
def evaluate(eval_model, data_source):
    eval_model.eval()           # eval mode on
    total_loss = 0
    n_tokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, n_tokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)

# Train model
# saves model at best eval loss
best_val_loss = float('inf')
epochs = 5
best_model = None

for epoch in range(1, epochs + 1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(model, val_data)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
          'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                     val_loss, math.exp(val_loss)))
    print('-' * 89)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model = model
        
    scheduler.step()
