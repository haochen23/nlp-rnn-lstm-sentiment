import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Conv1D
from tensorflow.keras.layers import Embedding, Flatten
from tensorflow.keras.layers import SimpleRNN,LSTM, GRU
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
import numpy as np
import spacy
from utils import lemmatize, load_data, load_glove_model, remove_stopwords
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# function to build RNN, or LSTM or GRU
def build_model(nb_words, rnn_model="SimpleRNN", embedding_matrix=None):
    '''
    build_model function:
    inputs: 
        rnn_model - which type of RNN layer to use, choose in (SimpleRNN, LSTM, GRU)
        embedding_matrix - whether to use pretrained embeddings or not
    '''
    model = Sequential()
    # add an embedding layer
    if embedding_matrix is not None:
        model.add(Embedding(nb_words, 
                        200, 
                        weights=[embedding_matrix], 
                        input_length= max_len,
                        trainable = False))
    else:
        model.add(Embedding(nb_words, 
                        200, 
                        input_length= max_len,
                        trainable = False))
        
    # add an RNN layer according to rnn_model
    if rnn_model == "SimpleRNN":
        model.add(SimpleRNN(200, activation='relu'))
    elif rnn_model == "LSTM":
        model.add(LSTM(200, activation='relu'))
    else:
        model.add(GRU(200, activation='relu'))
    # model.add(Dense(500,activation='relu'))
    # model.add(Dense(500, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', 
                optimizer='adam',
                metrics=['accuracy'])
    return model



if __name__ == '__main__':
    #load glove model
    glove_model = load_glove_model("glove.twitter.27B.200d.txt")
    # load data
    data = load_data("training.1600000.processed.noemoticon.csv")
    # twitter text
    data_X = data[data.columns[5]].to_numpy()
    # twitter label
    data_y = data[data.columns[0]]
    data_y = pd.get_dummies(data_y).to_numpy()
    #number of vacob to keep
    max_vocab = 18000
    #sequence length
    max_len = 15
    tokenizer = Tokenizer(num_words=max_vocab)
    tokenizer.fit_on_texts(data_X)
    sequences = tokenizer.texts_to_sequences(data_X)
    word_index = tokenizer.word_index
    
    data_keras = pad_sequences(sequences, maxlen=max_len, padding="post")
    print("prepared model inputs shape: {}".format(data_keras.shape))
    
    train_X, valid_X, train_y, valid_y = train_test_split(data_keras, data_y, test_size = 0.3, random_state=42)
    
    # calculate number of words
    nb_words = len(tokenizer.word_index) + 1

    # obtain the word embedding matrix
    embedding_matrix = np.zeros((nb_words, 200))
    for word, i in word_index.items():
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

    model = build_model(nb_words=nb_words, rnn_model="SimpleRNN", embedding_matrix=embedding_matrix)
    model.fit(train_X, train_y, epochs=20, batch_size=120,
          validation_data=(valid_X, valid_y), callbacks=EarlyStopping(monitor='val_accuracy', mode='max',patience=3))
    print(model.summary())
    
    predictions = model.predict(valid_X)
predictions = predictions.argmax(axis=1)
print(classification_report(valid_y.argmax(axis=1), predictions))