# -*- coding: utf-8 -*-
"""
Created on Sat Feb 22 15:24:32 2020

Preprocessing for model
Embedding to frequency and batching (target is 1 word ahead)

@author: hoang
"""
import torch
import torchtext
from torchtext.data.utils import get_tokenizer

def get_txt(raw_txt):
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("spacy"),
                        init_token="<sos>",
                        eos_token="<eos>",
                        lower=True,
                        include_lengths=True)
    TEXT.build_vocab(raw_txt)

    return TEXT

def batchify(text, batch_size):
    text = TEXT.numericalize([text.examples[0].text])
    nbatch = text.size(0) // batch_size
    text = text.narrow(0, 0, nbatch*batch_size)
    text = text.view(batch_size, -1).t().contiguous()
    
    return text.to(device)

def get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    
    return data, target

