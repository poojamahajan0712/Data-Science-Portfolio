import tensorflow as tf
from keras import backend as K
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import LSTM, Input, Dense, Embedding, TimeDistributed, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.models import Model,load_model, model_from_json

import pickle as pkl
import unicodedata
import os 
os.chdir('/content/drive/MyDrive/Machine_Translation')

from models.model1 import *
from models.util import *


max_length_french = 53
latent_dim = 50


# Load Saved Tokenizers
with open('data_pkl/NMT_Etokenizer.pkl', 'rb') as f:
    vocab_size_source, Eword2index, englishTokenizer = pkl.load(f)

with open('data_pkl/NMT_Ftokenizer.pkl', 'rb') as f:
    vocab_size_target, Fword2index, frenchTokenizer = pkl.load(f)

Findex2word = frenchTokenizer.index_word

## data preprocess
def tokenize_and_pad(sentence, tokenizer, max_length):
    """Tokenize and pad a single sentence."""
    # Convert sentence to sequence
    sequence = tokenizer.texts_to_sequences([sentence])
    # Pad the sequence
    padded_sequence = pad_sequences(sequence, maxlen=max_length, padding='post')
    return padded_sequence


model = translation_model(max_length_english,vocab_size_source,vocab_size_target,latent_dim)
model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# loading the model architecture and asigning the weights
json_file = open('/content/drive/MyDrive/Machine_Translation/saved_model/NMT_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model_loaded = model_from_json(loaded_model_json)
# load weights into new model
model_loaded.load_weights('/content/drive/MyDrive/Machine_Translation/saved_model/NMT_model_weight.weights.h5')

encoder_model, decoder_model = decoder_inference(model_loaded,latent_dim)

# inference function
def translate_sentence(sentence, max_length_french,encoder_model,Fword2index,decoder_model,Findex2word):
    """Translate an English sentence to French."""
    # Preprocess and tokenize the input sentence
    processed_sentence = lower_and_split_punct(sentence)
    tokenized_input = tokenize_and_pad(processed_sentence, englishTokenizer, max_length_french)
    print(tokenized_input)
    print(len(tokenized_input))
    
    # Decode the predicted sequence to text
    translated_sentence = decode_sequence(encoder_model,Fword2index,decoder_model,tokenized_input.reshape(1,max_length_french),Findex2word)
    
    return translated_sentence

new_sentence = "Run!"  
translate_sentence(new_sentence, max_length_french,encoder_model,Fword2index,decoder_model,Findex2word)
