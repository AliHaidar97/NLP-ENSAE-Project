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
from keras.layers import CuDNNLSTM, LSTM, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, LSTM, Lambda,MaxPooling1D, AveragePooling1D,GRU, Reshape
from keras.preprocessing import text, sequence
from gensim.models import KeyedVectors
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import PolynomialDecay



class BertMLP1Layer:
  
    def __init__(self, dropout=0.2):
        # Constructor method that sets the value of the dropout rate.
        self.dropout = dropout
  
    def build_model(self, embedding_matrix, labels, out=1):
        # Method to build and return the Keras model.
        # Input:
        #   embedding_matrix: Pre-trained word embeddings matrix.
        #   labels: Number of labels.
        #   out: Output shape of the model.
        # Output:
        #   Keras model.

        # Input layer.
        words = Input(shape=(512,))

        # Embedding layer.
        x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False, mask_zero=True)(words)

        # Dropout layer.
        x = SpatialDropout1D(self.dropout)(x)

        # Concatenation layer of GlobalMaxPooling1D and GlobalAveragePooling1D.
        x = concatenate([
            GlobalMaxPooling1D()(x),
            GlobalAveragePooling1D()(x)
        ])

        # Dense layer with softmax activation.
        logits = Dense(labels*out, activation='softmax')(x)

        # Reshape layer for multi-targets.
        if out != 1:
            logits = Reshape((out, labels))(logits)
    
        # Model creation.
        model = Model(inputs=words, outputs=logits)
      
        return model



class BertMLP2Layers:

  def __init__(self, HIDDEN = 128, dropout = 0.2):
    # Initialize the class with the given hyperparameters.
    self.HIDDEN = HIDDEN
    self.dropout = 0.2

  
  def build_model(self, embedding_matrix, labels, out=1):
    # Define the architecture of the model.
    # Input layer.
    words = Input(shape=(512,))
    # Embedding layer with pre-trained embeddings.
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False, mask_zero=True)(words)
    # Dropout layer.
    x = SpatialDropout1D(self.dropout)(x)
    # Concatenation of GlobalMaxPooling1D and GlobalAveragePooling1D layers.
    x = concatenate([
        GlobalMaxPooling1D()(x),
        GlobalAveragePooling1D()(x),
    ])
    # First dense layer with ReLU activation.
    x = Dense(self.HIDDEN, activation='relu')(x)
    # Output layer with softmax activation.
    logits = Dense(labels*out, activation='softmax')(x)
    # Reshape for mutlti targets.
    if(out != 1):
        logits = Reshape((out, labels))(logits)
    # Define and return the model.
    model = Model(inputs=words, outputs=logits)
    return model


    
class BertGRU:
  
  def __init__(self, GRU_UNITS=128, dropout=0.2):
    self.GRU_UNITS = GRU_UNITS
    self.dropout = dropout
    self.DENSE_HIDDEN_UNITS = 4 * GRU_UNITS  # number of units in the dense hidden layer

  def build_model(self, embedding_matrix, labels, out=1):
    words = Input(shape=(None,))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=False, mask_zero=True)(words)

    x = SpatialDropout1D(self.dropout)(x)
    # Bidirectional GRU layer, returns sequences
    y = Bidirectional(GRU(self.GRU_UNITS, return_sequences=True))(x)
    y = SpatialDropout1D(self.dropout)(y)  # apply spatial dropout
    # Bidirectional GRU layer, returns sequences
    z = Bidirectional(GRU(self.GRU_UNITS, return_sequences=True))(y)

    # Global pooling layers for each GRU layer output
    hidden_x = concatenate([GlobalMaxPooling1D()(x), GlobalAveragePooling1D()(x)], axis=1)
    hidden_y = concatenate([GlobalMaxPooling1D()(y), GlobalAveragePooling1D()(y)], axis=1)
    hidden_z = concatenate([GlobalMaxPooling1D()(z), GlobalAveragePooling1D()(z)], axis=1)

    # Concatenate the global pooling outputs from each GRU layer
    hidden = concatenate([hidden_x, hidden_y, hidden_z], axis=1)
    hidden = Dense(self.DENSE_HIDDEN_UNITS, activation='relu')(hidden)  # dense hidden layer
    logits = Dense(labels * out, activation='softmax')(hidden)  # output layer with softmax activation

    if out != 1:
      logits = Reshape((out, labels))(logits)  # Reshape for mutlti targets.

    # create the model
    model = Model(inputs=words, outputs=logits)
    return model


def get_bert_embed_matrix():
    bert = BertModel.from_pretrained('bert-base-uncased')
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat