from keras import backend as K
from tensorflow.keras.layers import LSTM, Input, Dense, Embedding, TimeDistributed, MultiHeadAttention, Concatenate
from tensorflow.keras.models import Model
import numpy as np

def translation_model(max_length_english,vocab_size_source,vocab_size_target):
    K.clear_session()

    latent_dim = 100  # Dimensionality of the latent space

    # Encoder
    encoder_inputs = Input(shape=(max_length_english,))
    enc_emb = Embedding(vocab_size_source, 50, trainable=True)(encoder_inputs)
    ## embedding layer takes max vocab length of english as input and latent_dim refers to output of embedding

    # LSTM 1
    encoder_lstm1 = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_output1, state_h1, state_c1 = encoder_lstm1(enc_emb)
    ## num of units in LSTM can be different from embedding length also, basically the output dimension will be max_length_english,latent_dim

    # LSTM 2
    encoder_lstm2 = LSTM(latent_dim, return_sequences=True, return_state=True)
    encoder_output2, state_h2, state_c2 = encoder_lstm2(encoder_output1)

    # LSTM 3
    encoder_lstm3 = LSTM(latent_dim, return_state=True, return_sequences=True)
    encoder_outputs, state_h, state_c = encoder_lstm3(encoder_output2)

    # Set up the decoder.
    decoder_inputs = Input(shape=(None,))
    dec_emb_layer = Embedding(vocab_size_target, 50, trainable=True)
    dec_emb = dec_emb_layer(decoder_inputs)

    # LSTM using encoder_states as initial state
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(dec_emb, initial_state=[state_h, state_c])

    # Multi-Head Attention Layer
    multi_head_attn = MultiHeadAttention(num_heads=8, key_dim=latent_dim)
    attn_out, attn_scores = multi_head_attn(query=decoder_outputs, value=encoder_outputs, return_attention_scores=True)

    # Concatenate attention output and decoder LSTM output
    decoder_concat_input = Concatenate(axis=-1)([decoder_outputs, attn_out])

    # Dense layer
    decoder_dense = TimeDistributed(Dense(vocab_size_target, activation='softmax'))
    decoder_outputs = decoder_dense(decoder_concat_input)

    # Define the model
    return Model([encoder_inputs, decoder_inputs], decoder_outputs)


def decoder_inference(model_loaded,latent_dim):

    # latent_dim=50
    # encoder inference
    encoder_inputs = model_loaded.input[0]  #loading encoder_inputs
    encoder_outputs, state_h, state_c = model_loaded.layers[6].output #loading encoder_outputs

    print(encoder_outputs.shape)

    encoder_model = Model(inputs=encoder_inputs,outputs=[encoder_outputs, state_h, state_c])

    # decoder inference
    # Below tensors will hold the states of the previous time step
    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_hidden_state_input = Input(shape=(53,latent_dim))

    # Get the embeddings of the decoder sequence
    decoder_inputs = model_loaded.layers[3].output

    print(decoder_inputs.shape)
    dec_emb_layer = model_loaded.layers[5]

    dec_emb2= dec_emb_layer(decoder_inputs)

    # To predict the next word in the sequence, set the initial states to the states from the previous time step
    decoder_lstm = model_loaded.layers[7]
    decoder_outputs2, state_h2, state_c2 = decoder_lstm(dec_emb2, initial_state=[decoder_state_input_h, decoder_state_input_c])

    #attention inference
    multi_head_attn_layer = model_loaded.layers[8]
    attn_out_inf, attn_scores_inf = multi_head_attn_layer(query=decoder_outputs2, value=decoder_hidden_state_input, return_attention_scores=True)


    concate = model_loaded.layers[9]
    decoder_inf_concat = concate([decoder_outputs2, attn_out_inf])

    # A dense softmax layer to generate prob dist. over the target vocabulary
    decoder_dense = model_loaded.layers[10]
    decoder_outputs2 = decoder_dense(decoder_inf_concat)

    # Final decoder model
    return encoder_model,Model(
    [decoder_inputs] + [decoder_hidden_state_input,decoder_state_input_h, decoder_state_input_c],
    [decoder_outputs2] + [state_h2, state_c2])

def decode_sequence(encoder_model,Fword2index,decoder_model,input_seq,Findex2word):
    # Encode the input as state vectors.
    e_out, e_h, e_c = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))

    # Chose the 'start' word as the first word of the target sequence
    target_seq[0, 0] = Fword2index['start']

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + [e_out, e_h, e_c])

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        if sampled_token_index == 0:
          break
        else:
          sampled_token = Findex2word[sampled_token_index]

          if(sampled_token!='end'):
              decoded_sentence += ' '+sampled_token

              # Exit condition: either hit max length or find stop word.
              if (sampled_token == 'end' or len(decoded_sentence.split()) >= (26-1)):
                  stop_condition = True

          # Update the target sequence (of length 1).
          target_seq = np.zeros((1,1))
          target_seq[0, 0] = sampled_token_index

          # Update internal states
          e_h, e_c = h, c

    return decoded_sentence

    
def seq2summary(input_seq,Fword2index,Findex2word):
    newString=''
    for i in input_seq:
      if((i!=0 and i!=Fword2index['start']) and i!=Fword2index['end']):
        newString=newString+Findex2word[i]+' '
    return newString

def seq2text(input_seq,Eindex2word):
    newString=''
    for i in input_seq:
      if(i!=0):
        newString=newString+Eindex2word[i]+' '
    return newString



    
