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


def tokenize(text, tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)):
    
    tokens = list(map(lambda t: ["[CLS]"] + tokenizer.tokenize(t)[:510] + ["[SEP]"] , text)) #The max size of bert input is 512
    tokens_ids = list(map(tokenizer.convert_tokens_to_ids, tokens))
    tokens_ids = pad_sequences(tokens_ids, maxlen=512, truncating="post", padding="post", dtype="int",value=-1) #Pad to get 512
    
    return tokens_ids

def mask(token_ids):
    
    return [[int(i >= 0) for i in ii] for ii in token_ids]





def context(train, test, sizeOfTheContext = -1):
    
    train_modified = train[['Dialogue_ID','Utterance','Label']]
    train_modified['Label'] = train_modified['Label'].apply(lambda x:[x])
    train_modified = train_modified.groupby('Dialogue_ID').agg(sum).reset_index()

    test_modified = test[['Dialogue_ID','Utterance','Label']]
    test_modified['Label'] = test_modified['Label'].apply(lambda x:[x])
    test_modified = test_modified.groupby('Dialogue_ID').agg(sum).reset_index()

    if(sizeOfTheContext != -1):
        #Filter the context
        train_modified['len'] = train_modified.apply(lambda x: len(x['Label']),axis=1)
        test_modified['len'] = test_modified.apply(lambda x: len(x['Label']),axis=1)
        
        train_modified['filter']  = (train_modified['len'] < sizeOfTheContext)*1
        test_modified['filter']  = (test_modified['len'] < sizeOfTheContext)*1
        train_modified = train_modified[(train_modified['filter'] == 1)]
        test_modified = test_modified[(test_modified['filter'] == 1)]
        
        train_modified = train_modified.drop(['len','filter'],axis=1)
        test_modified = test_modified.drop(['len','filter'],axis=1)

    if(sizeOfTheContext == -1):
        sizeOfTheContext = max(np.max(train_modified.apply(lambda x: len(x['Label']),axis=1)), np.max(test_modified.apply(lambda x: len(x['Label']),axis=1))) 

        
    pad = list(pad_sequences(train_modified['Label'], maxlen=sizeOfTheContext, truncating="post", padding="post", dtype="int", value = -1))
    pad = [list(i) for i in pad]
    train_modified['Label'] = pad

    pad = list(pad_sequences(test_modified['Label'], maxlen=sizeOfTheContext, truncating="post", padding="post", dtype="int", value = -1))
    pad = [list(i) for i in pad]
    test_modified['Label'] = pad
    
    return train_modified, test_modified