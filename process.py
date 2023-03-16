import numpy as np 
import pandas as pd
from datasets import load_dataset
from keras_preprocessing.sequence import pad_sequences
from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert import BertModel
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.nn.utils import clip_grad_norm_
from IPython.display import clear_output


def tokenize(text, tokenizer):
    # Add [CLS] and [SEP] tokens to each text and truncate to 510 tokens
    tokens = list(map(lambda t: ["[CLS]"] + tokenizer.tokenize(t)[:510] + ["[SEP]"], text))
    # Convert tokens to token ids
    tokens_ids = list(map(tokenizer.convert_tokens_to_ids, tokens))
    # Pad the token ids to a length of 512
    tokens_ids = pad_sequences(tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int")
    return tokens_ids



def mask(token_ids):
    
    return [[i > 0 for i in ii] for ii in token_ids]



def context(train, val, test, sizeOfTheContext=-1):
    
    train_modified = train[['Dialogue_ID', 'Utterance', 'Label']].copy()
    train_modified['Label'] = train_modified['Label'].apply(lambda x: [x])
    train_modified = train_modified.groupby('Dialogue_ID').agg(sum).reset_index()
    
    val_modified = val[['Dialogue_ID', 'Utterance', 'Label']].copy()
    val_modified['Label'] = val_modified['Label'].apply(lambda x: [x])
    val_modified = val_modified.groupby('Dialogue_ID').agg(sum).reset_index()

    test_modified = test[['Dialogue_ID', 'Utterance', 'Label']].copy()
    test_modified['Label'] = test_modified['Label'].apply(lambda x: [x])
    test_modified = test_modified.groupby('Dialogue_ID').agg(sum).reset_index()

    if sizeOfTheContext != -1:
        # Filter the context
        train_modified['len'] = train_modified['Label'].apply(len)
        test_modified['len'] = test_modified['Label'].apply(len)
        val_modified['len'] = val_modified['Label'].apply(len)
        
        train_modified = train_modified[train_modified['len'] < sizeOfTheContext]
        test_modified = test_modified[test_modified['len'] < sizeOfTheContext]
        val_modified = val_modified[val_modified['len'] < sizeOfTheContext]
        
        train_modified = train_modified.drop(['len'], axis=1)
        test_modified = test_modified.drop(['len'], axis=1)
        val_modified = val_modified.drop(['len'], axis=1)

    if sizeOfTheContext == -1:
        sizeOfTheContext = max(
            np.max(val_modified['Label'].apply(len)),
            np.max(train_modified['Label'].apply(len)),
            np.max(test_modified['Label'].apply(len))
        )

    pad = list(pad_sequences(train_modified['Label'], maxlen=sizeOfTheContext, truncating='post', padding='post', dtype='int', value=-1))
    pad = [[j for j in i] for i in pad]
    train_modified['Label'] = pad

    pad = list(pad_sequences(test_modified['Label'], maxlen=sizeOfTheContext, truncating='post', padding='post', dtype='int', value=-1))
    pad = [[j for j in i] for i in pad]
    test_modified['Label'] = pad
    
    pad = list(pad_sequences(val_modified['Label'], maxlen=sizeOfTheContext, truncating='post', padding='post', dtype='int', value=-1))
    pad = [[j for j in i] for i in pad]
    val_modified['Label'] = pad
    
    return train_modified, val_modified, test_modified
