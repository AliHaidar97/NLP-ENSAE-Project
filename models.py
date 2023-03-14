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


from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, add, concatenate,Flatten
from keras.layers import CuDNNLSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM
from keras.preprocessing import text, sequence
from gensim.models import KeyedVectors
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import PolynomialDecay


class BertMLP1Layer:

  def __init__(self,  dropout = 0.2):

    self.dropout = 0.2

  
  def build_model(self, embedding_matrix, num_aux_targets):
      words = Input(shape=(512))
      x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False, mask_zero = True)(words)
      x = SpatialDropout1D(self.dropout)(x)
      x = Flatten()(x)
      aux_result = Dense(num_aux_targets, activation='softmax')(x)
      model = Model(inputs=words, outputs=aux_result)
      return model


class BertMLP2Layers:

  def __init__(self, HIDDEN = 128, dropout = 0.2):

    self.HIDDEN = HIDDEN
    self.dropout = 0.2

  
  def build_model(self, embedding_matrix, num_aux_targets):
      words = Input(shape=(512,))
      x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
      x = SpatialDropout1D(self.dropout)(x)
      x = Flatten()(x)
      x = Dense(self.HIDDEN, activation='relu')(x)
      aux_result = Dense(num_aux_targets, activation='softmax')(x)
      
      model = Model(inputs=words, outputs=aux_result)
   
      return model
    
class BertLstm:

  def __init__(self,  LSTM_UNITS = 64, dropout = 0.2):
 
    self.LSTM_UNITS = LSTM_UNITS
    self.dropout = 0.2
    self.DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

        
  
  def build_model(self, embedding_matrix, num_aux_targets):
      words = Input(shape=(None,))
      x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
      x = SpatialDropout1D(self.dropout)(x)
      x = Bidirectional(CuDNNLSTM(self.LSTM_UNITS , return_sequences=True))(x)
      x = Bidirectional(CuDNNLSTM(self.LSTM_UNITS , return_sequences=True))(x)
      hidden = concatenate([
          GlobalMaxPooling1D()(x),
          GlobalAveragePooling1D()(x),
      ])
      hidden = add([hidden, Dense(self.DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
      aux_result = Dense(num_aux_targets, activation='softmax')(hidden)
      
      model = Model(inputs=words, outputs=aux_result)
    
      return model
  
  
class BertDoubleLstm:

  def __init__(self, LSTM_UNITS = 64, dropout = 0.2):
      
      self.LSTM_UNITS = LSTM_UNITS
      self.dropout = 0.2
      self.DENSE_HIDDEN_UNITS = 4 * LSTM_UNITS

  def build_model(self,embedding_matrix, num_aux_targets):
      words = Input(shape=(None,))
      x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False)(words)
      x = SpatialDropout1D(self.dropout)(x)
      x = Bidirectional(CuDNNLSTM(self.LSTM_UNITS, return_sequences=True))(x)
      x = Bidirectional(CuDNNLSTM(self.LSTM_UNITS, return_sequences=True))(x)
      hidden = concatenate([
          GlobalMaxPooling1D()(x),
          GlobalAveragePooling1D()(x),
      ])
      hidden = add([hidden, Dense(self.DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
      hidden = add([hidden, Dense(self.DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
      aux_result = Dense(num_aux_targets, activation='softmax')(hidden)
      
      model = Model(inputs=words, outputs=aux_result)
 
      return model