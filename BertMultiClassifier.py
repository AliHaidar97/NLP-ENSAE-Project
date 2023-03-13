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
from tqdm import tqdm

#Create an encoder using Bert and decoder as linear layer
class BertMultiClassifier(nn.Module):
    
    def __init__(self, n_classes, dropout=0.1):
        super(BertMultiClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.n_classes = n_classes
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(768, n_classes)
        self.softmax = nn.Softmax()
    
    def forward(self, tokens, masks=None):
        _, pooled_output = self.bert(tokens, attention_mask=masks, output_all_encoded_layers=False)
        dropout_output = self.dropout(pooled_output)
        linear_output = self.linear(dropout_output)
        proba = self.softmax(linear_output)
        return proba
    
def train(model_clf , train_tokens_ids, train_masks, y_train,  BATCH_SIZE = 8, EPOCHS = 10):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    model_clf = model_clf.to(device)

    train_tokens_tensor = torch.tensor(train_tokens_ids)
    train_y_tensor = torch.tensor(y_train.reshape(-1, 1))
    train_masks_tensor = torch.tensor(train_masks)
    
    train_dataset = TensorDataset(train_tokens_tensor, train_masks_tensor, train_y_tensor)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=BATCH_SIZE)

    param_optimizer = model_clf.linear.parameters()

    optimizer = Adam(param_optimizer, lr=3e-6)
    
    for epoch_num in tqdm(range(EPOCHS)):
        model_clf.train()
        train_loss = 0
        for step_num, batch_data in enumerate(train_dataloader):
            token_ids, masks, labels = tuple(t.to(device) for t in batch_data)
        
            logits = model_clf(token_ids, masks)
            
            loss_func = nn.CrossEntropyLoss()
            
            batch_loss = loss_func(logits, labels.squeeze())
            
            train_loss += batch_loss.item()
    
            model_clf.zero_grad()
            batch_loss.backward()

            clip_grad_norm_(parameters=param_optimizer, max_norm=1.0)
            optimizer.step()
            
            clear_output(wait=True)
            print('Epoch: ', epoch_num + 1)
            print("\r" + "{0}/{1} loss: {2} ".format(step_num, len(train_tokens_ids) / BATCH_SIZE, train_loss / (step_num + 1)))
            
def test(model_clf, test_tokens_ids, test_masks, y_test, BATCH_SIZE = 8):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   
    test_tokens_tensor = torch.tensor(test_tokens_ids)
    test_y_tensor = torch.tensor(y_test.reshape(-1, 1))
    
    test_masks_tensor = torch.tensor(test_masks)
    
    test_dataset = TensorDataset(test_tokens_tensor, test_masks_tensor, test_y_tensor)
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=BATCH_SIZE)
    
    model_clf.eval()
    bert_predicted = []
    loss_pred = 0
    loss_all = 0
    with torch.no_grad():

        for step_num, batch_data in tqdm(enumerate(test_dataloader)):

            token_ids, masks, labels = tuple(t.to(device) for t in batch_data)

            logits = model_clf(token_ids, masks)
            loss_func = nn.CrossEntropyLoss()
            loss = loss_func(logits, labels.squeeze())

            loss_all+= loss.item()
        
            numpy_logits = logits.cpu().detach().numpy()    
            pred = np.argmax(numpy_logits,axis = 1)
            loss_pred = np.sum(pred != labels.cpu().detach().numpy() )
            bert_predicted += list(pred)

   
    bert_predicted = np.array(bert_predicted)
    acc = (np.sum(bert_predicted == y_test)/len(y_test)) *100
    return acc, bert_predicted, labels

    