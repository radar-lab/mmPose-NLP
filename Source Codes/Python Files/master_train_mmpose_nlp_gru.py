# %tensorflow_version 2.x

import tensorflow as tf
print(tf.test.gpu_device_name())
print(tf.test.is_gpu_available())

#### Import relevant libraries###

import numpy as np
from matplotlib import pyplot as plt
#from scipy.io import loadmat
import pandas as pd
import random
import os
from tensorflow.python.keras import backend as K
from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow.keras.preprocessing.text as kpt
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Layer, Input, Dense, Embedding, LSTM, Flatten, Dropout, SpatialDropout1D, Bidirectional, GRU, LeakyReLU, TimeDistributed, Concatenate, Reshape, Conv1D,MaxPooling1D,Conv2D,MaxPooling2D, AveragePooling1D
#from tensorflow.compat.v1.keras.layers import CuDNNLSTM
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical, plot_model
import re
import math

##### initialize a seed for repeatability

os.environ['PYTHONHASHSEED'] = '0'
random.seed(42)
np.random.seed(42)
tf.random.set_random_seed(42)

def train_model(iii):

    #### Set paths to Training, Dev and Test Data###

    X_tr_pd_path = "../../Dataset and Predictions/Data_%s_Frames/X_tr_pd.csv" %(str(iii+1))
    Y_tr_pd_path = "../../Dataset and Predictions/Data_%s_Frames/Y_tr_pd.csv" %(str(iii+1))

    X_test_pd_path = "../../Dataset and Predictions/Data_%s_Frames/X_test_pd.csv" %(str(iii+1))
    Y_test_pd_path = "../../Dataset and Predictions/Data_%s_Frames/Y_test_pd.csv" %(str(iii+1))

    ####Load the pd Dataframes directly###

    X_tr_df = pd.read_csv(X_tr_pd_path)
    Y_tr_df = pd.read_csv(Y_tr_pd_path)

    X_test_df = pd.read_csv(X_test_pd_path)
    Y_test_df = pd.read_csv(Y_test_pd_path)

    print(X_tr_df.shape)
    print(Y_tr_df.shape)

    print(X_test_df.shape)
    print(Y_test_df.shape)

    ####Preprocess Voxels by creating a voxel vocabulary

    ## Add start and end token to the output voxels
    ###Similar to start and end of sentence
    ###

    Y_tr_df['Voxels'] = Y_tr_df['Voxels'].apply(lambda x : 'sostok '+ x + ' eostok')
    #Y_dev_df['Voxels'] = Y_dev_df['Voxels'].apply(lambda x : 'sostok '+ x + ' eostok')
    Y_test_df['Voxels'] = Y_test_df['Voxels'].apply(lambda x : 'sostok '+ x + ' eostok')

    ##### Concatenate Training, Dev and Test Data to obtain all the possible occuring voxels###

    combined_data_X = pd.concat([X_tr_df,X_test_df])
    combined_data_Y = pd.concat([Y_tr_df,Y_test_df])

    print(combined_data_X.shape)
    print(combined_data_Y.shape)

    ##### Tokenize the vocab on the combined input data###

    x_tokenizer = Tokenizer(num_words=45000,lower=True,split=' ')
    x_tokenizer.fit_on_texts(combined_data_X['Voxels'].values) #We are only interested in the 'Voxels' column's vocabulary
    x_dictionary = x_tokenizer.word_index #A dictionary of Voxel-index pairs

    vocab_size_x = len(x_dictionary)
    print(vocab_size_x)

    X_train_tok = x_tokenizer.texts_to_sequences(X_tr_df['Voxels']) #For the training data
    #X_dev_tok = x_tokenizer.texts_to_sequences(X_dev_df['Voxels']) #For the development data
    X_test_tok = x_tokenizer.texts_to_sequences(X_test_df['Voxels']) #For the test data

    np.shape(X_train_tok)

    ##### Tokenize the vocab on the combined input data###

    y_tokenizer = Tokenizer(num_words=45000,lower=True,split=' ')
    y_tokenizer.fit_on_texts(combined_data_Y['Voxels'].values) #We are only interested in the 'Voxels' column's vocabulary
    y_dictionary = y_tokenizer.word_index #A dictionary of Voxel-index pairs

    vocab_size_y = len(y_dictionary)+1
    print(vocab_size_y)

    Y_train_tok = y_tokenizer.texts_to_sequences(Y_tr_df['Voxels']) #For the training data
    #Y_dev_tok = y_tokenizer.texts_to_sequences(Y_dev_df['Voxels']) #For the development data
    Y_test_tok = y_tokenizer.texts_to_sequences(Y_test_df['Voxels']) #For the test data

    np.shape(Y_train_tok)

    ####Reorganize the input data to a sliding stack of multiple frames

    ##Define a function to take in an array and output a sliding stack of multiple frames
    ###
    X_tr = np.array(X_train_tok)-1
    Y_tr = np.array(Y_train_tok)

    X_test = np.array(X_test_tok)-1
    Y_test = np.array(Y_test_tok)

    print(np.shape(X_tr))
    print(np.shape(Y_tr))
    ##print(np.shape(X_val))
   
    ##print(np.shape(Y_val))
   
    print(np.shape(X_test))
    print(np.shape(Y_test))

    X_tr = np.reshape(X_tr,(np.shape(X_tr)[0],int((np.shape(X_tr)[1])/90),90))

    X_test = np.reshape(X_test,(np.shape(X_test)[0],int((np.shape(X_test)[1])/90),90))

    print(np.shape(X_tr))
    print(np.shape(Y_tr))
    ##print(np.shape(X_val))
   
    ##print(np.shape(Y_val))
   
    print(np.shape(X_test))
    print(np.shape(Y_test))

    X_tot = X_tr
    Y_tot = Y_tr

    from sklearn.utils import shuffle
    X_tot, Y_tot = shuffle(X_tot,Y_tot)
    #### Build a model



    ## Attention Layer
    ###

        
    for jjj in range(0,5):

        n_train_samp = math.ceil(0.8*np.shape(X_tot)[0])
        idx_tot = np.arange(0,np.shape(X_tot)[0])
        np.random.shuffle(idx_tot)
        X_train = X_tot[idx_tot[range(0,n_train_samp)],:,:]
        Y_train = Y_tot[idx_tot[range(0,n_train_samp)],:]
        new_test_idx = np.setdiff1d(idx_tot, idx_tot[range(0,n_train_samp)])
        X_shuffle_test =  X_tot[new_test_idx,:,:]
        Y_shuffle_test = Y_tot[new_test_idx,:]

        X_tr_path = "../../Dataset and Predictions/Data_%s_Frames/Datasets/X_tr_iter_%s.npy" %(str(iii+1),str(jjj+1))
        np.save(X_tr_path,X_train)

        Y_tr_path = "../../Dataset and Predictions/Data_%s_Frames/Datasets/Y_tr_iter_%s.npy" %(str(iii+1),str(jjj+1))
        np.save(Y_tr_path,Y_train)

        X_shuf_test_path = "../../Dataset and Predictions/Data_%s_Frames/Datasets/X_shuf_test_iter_%s.npy" %(str(iii+1),str(jjj+1))
        np.save(X_shuf_test_path,X_shuffle_test)

        Y_shuf_test_path = "../../Dataset and Predictions/Data_%s_Frames/Datasets/Y_shuf_test_iter_%s.npy" %(str(iii+1),str(jjj+1))
        np.save(Y_shuf_test_path,Y_shuffle_test)

        X_cont_test_path = "../../Dataset and Predictions/Data_%s_Frames/Datasets/X_cont_test_iter_%s.npy" %(str(iii+1),str(jjj+1))
        np.save(X_test_path,X_test)

        Y_cont_test_path = "../../Dataset and Predictions/Data_%s_Frames/Datasets/Y_cont_test_iter_%s.npy" %(str(iii+1),str(jjj+1))
        np.save(Y_test_path,Y_test)

        class AttentionLayer(Layer):
            ###
            #This class implements Bahdanau attention (https://arxiv.org/pdf/1409.0473.pdf).
            #There are three sets of weights introduced W_a, U_a, and V_a
            ###

            def __init__(self, **kwargs):
                super(AttentionLayer, self).__init__(**kwargs)

            def build(self, input_shape):
                assert isinstance(input_shape, list)
                # Create a trainable weight variable for this layer.

                self.W_a = self.add_weight(name='W_a',
                                        shape=tf.TensorShape((input_shape[0][2], input_shape[0][2])),
                                        initializer='uniform',
                                        trainable=True)
                self.U_a = self.add_weight(name='U_a',
                                        shape=tf.TensorShape((input_shape[1][2], input_shape[0][2])),
                                        initializer='uniform',
                                        trainable=True)
                self.V_a = self.add_weight(name='V_a',
                                        shape=tf.TensorShape((input_shape[0][2], 1)),
                                        initializer='uniform',
                                        trainable=True)

                super(AttentionLayer, self).build(input_shape)  # Be sure to call this at the end

            def call(self, inputs, verbose=False):
                ###
                #inputs: [encoder_output_sequence, decoder_output_sequence]
                ###
                assert type(inputs) == list
                encoder_out_seq, decoder_out_seq = inputs
                if verbose:
                    print('encoder_out_seq>', encoder_out_seq.shape)
                    print('decoder_out_seq>', decoder_out_seq.shape)

                def energy_step(inputs, states):
                    ### Step function for computing energy for a single decoder state ###

                    assert_msg = "States must be a list. However states {} is of type {}".format(states, type(states))
                    assert isinstance(states, list) or isinstance(states, tuple), assert_msg

                    ### Some parameters required for shaping tensors###
                    en_seq_len, en_hidden = encoder_out_seq.shape[1], encoder_out_seq.shape[2]
                    de_hidden = inputs.shape[-1]

                    ### Computing S.Wa where S=[s0, s1, ..., si]###
                    # <= batch_size*en_seq_len, latent_dim
                    reshaped_enc_outputs = K.reshape(encoder_out_seq, (-1, en_hidden))
                    # <= batch_size*en_seq_len, latent_dim
                    W_a_dot_s = K.reshape(K.dot(reshaped_enc_outputs, self.W_a), (-1, en_seq_len, en_hidden))
                    if verbose:
                        print('wa.s>',W_a_dot_s.shape)

                    ### Computing hj.Ua ###
                    U_a_dot_h = K.expand_dims(K.dot(inputs, self.U_a), 1)  # <= batch_size, 1, latent_dim
                    if verbose:
                        print('Ua.h>',U_a_dot_h.shape)

                    ### tanh(S.Wa + hj.Ua) ###
                    # <= batch_size*en_seq_len, latent_dim
                    reshaped_Ws_plus_Uh = K.tanh(K.reshape(W_a_dot_s + U_a_dot_h, (-1, en_hidden)))
                    if verbose:
                        print('Ws+Uh>', reshaped_Ws_plus_Uh.shape)

                    ### softmax(va.tanh(S.Wa + hj.Ua)) ###
                    # <= batch_size, en_seq_len
                    e_i = K.reshape(K.dot(reshaped_Ws_plus_Uh, self.V_a), (-1, en_seq_len))
                    # <= batch_size, en_seq_len
                    e_i = K.softmax(e_i)

                    if verbose:
                        print('ei>', e_i.shape)

                    return e_i, [e_i]

                def context_step(inputs, states):
                    ### Step function for computing ci using ei ###
                    # <= batch_size, hidden_size
                    c_i = K.sum(encoder_out_seq * K.expand_dims(inputs, -1), axis=1)
                    if verbose:
                        print('ci>', c_i.shape)
                    return c_i, [c_i]

                def create_inital_state(inputs, hidden_size):
                    # We are not using initial states, but need to pass something to K.rnn funciton
                    fake_state = K.zeros_like(inputs)  # <= (batch_size, enc_seq_len, latent_dim
                    fake_state = K.sum(fake_state, axis=[1, 2])  # <= (batch_size)
                    fake_state = K.expand_dims(fake_state)  # <= (batch_size, 1)
                    fake_state = K.tile(fake_state, [1, hidden_size])  # <= (batch_size, latent_dim
                    return fake_state

                fake_state_c = create_inital_state(encoder_out_seq, encoder_out_seq.shape[-1])
                fake_state_e = create_inital_state(encoder_out_seq, encoder_out_seq.shape[1])  # <= (batch_size, enc_seq_len, latent_dim

                ### Computing energy outputs ###
                # e_outputs => (batch_size, de_seq_len, en_seq_len)
                last_out, e_outputs, _ = K.rnn(
                    energy_step, decoder_out_seq, [fake_state_e],
                )

                ### Computing context vectors ###
                last_out, c_outputs, _ = K.rnn(
                    context_step, e_outputs, [fake_state_c],
                )

                return c_outputs, e_outputs

            def compute_output_shape(self, input_shape):
                ### Outputs produced by the layer ###
                return [
                    tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[1][2])),
                    tf.TensorShape((input_shape[1][0], input_shape[1][1], input_shape[0][1]))
                ]

        ##### Seq2Seq Model###

        latent_dim = 512
        embedding_dim=50
        max_text_len = np.shape(X_train)[2]
        n_frames = np.shape(X_train)[1]

        ###### Encoder Stage###

        # Encoder
        encoder_inputs = Input(shape=(n_frames, max_text_len), name="Encoder_Input")

        #embedding layer
        enc_emb =  Embedding(vocab_size_x, embedding_dim,trainable=True,mask_zero=False, name="Encoder_Embedding")(encoder_inputs)

        reshape_emb = (Reshape((n_frames,embedding_dim*max_text_len)))(enc_emb)

        #encoder dropout
        enc_dropout = (TimeDistributed(Dropout(rate=0.5)))(reshape_emb)

        #encoder gru 1
        encoder_gru1 = GRU(2*latent_dim,return_sequences=True,return_state=True,dropout=0.5,recurrent_dropout=0.4, name="Encoder_GRU_1")
        encoder_output1, _ = encoder_gru1(enc_dropout)

        #encoder gru 2

        encoder_gru2= GRU(latent_dim, return_state=True, return_sequences=True,dropout=0.5,recurrent_dropout=0.4, name="Encoder_GRU_2")
        encoder_outputs, f_state_h= encoder_gru2(encoder_output1)

        enc_state_h = f_state_h

        ###### Decoder Stage###

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None,), name="Decoder_Input")

        #embedding layer
        dec_emb_layer = Embedding(vocab_size_y, embedding_dim,trainable=True,mask_zero=False, name="Decoder_Embedding")
        dec_emb = dec_emb_layer(decoder_inputs)

        #decoder dropout
        dec_dropout = (TimeDistributed(Dropout(rate=0.5)))(dec_emb)

        #decoder gru 1
        decoder_gru1 = GRU(latent_dim, return_sequences=True, return_state=True,dropout=0.5,recurrent_dropout=0.4, name="Decoder_GRU1")
        decoder_outputs, _ = decoder_gru1(dec_dropout,initial_state=enc_state_h)


        ###### Add in Attention Layer###

        # Attention layer
        attn_layer = AttentionLayer(name='attention_layer')
        attn_out, attn_states = attn_layer([encoder_outputs, decoder_outputs])

        ###### Concatenate Attn. Output and Decoder Output to the Output Layer###

        # Concat attention input and decoder LSTM output
        decoder_concat_input = Concatenate(axis=-1, name='concat_dec_attn_layer')([decoder_outputs, attn_out])

        attn_dropout = (TimeDistributed(Dropout(rate=0.5)))(decoder_concat_input)

        decoder_dense =  TimeDistributed(Dense(vocab_size_y, activation='softmax'))
        decoder_outputs = decoder_dense(attn_dropout)

        # Define the model 
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        model.summary()

        ###### Add compilation parameters and loss functions###

        model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        callback = EarlyStopping(monitor='loss',min_delta=0.0000,patience=10,verbose=1, mode='auto',restore_best_weights = True)

        ###### Train the model###

        history=model.fit([X_train,Y_train[:,:-1]], Y_train.reshape(Y_train.shape[0],Y_train.shape[1], 1)[:,1:] ,epochs=200,callbacks=[callback],batch_size=64)

        modelpath = "../../Dataset and Predictions/Data_%s_Frames/Models/mmpose-nlp-%s-frames_iter_%s.h5" %(str(iii+1),str(iii+1),str(jjj+1))

        model.save(modelpath)
        print('Model Saved!')

        print('Training Done!')

        tf.keras.backend.clear_session()

for i in range(0,10):
    train_model(i)
