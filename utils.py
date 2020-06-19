import pandas as pd 
import spacy
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences

nlp = spacy.load("en")

def load_glove_model(glove_file):
    '''
    function to load glove model for pre-trained word embeddings
    '''
    print("[INFO]Loading GloVe Model...")
    model = {}
    with open(glove_file, 'r') as f:
        for line in f:
            split_line = line.split()
            word = split_line[0]
            embeddings = [float(val) for val in split_line[1:]]
            model[word] = embeddings
    print("[INFO] Done...{} words loaded!".format(len(model)))
    return model

def remove_stopwords(sentence):
    '''
    function to remove stopwords
        input: sentence - string of sentence
    '''
    new = []
    # tokenize sentence
    sentence = nlp(sentence)
    for tk in sentence:
        if (tk.is_stop == False) & (tk.pos_ !="PUNCT"):
            new.append(tk.string.strip())
    # convert back to sentence string
    c = " ".join(str(x) for x in new)
    return c


def lemmatize(sentence):
    '''
    function to do lemmatization
        input: sentence - string of sentence
    '''
    sentence = nlp(sentence)
    s = ""
    for w in sentence:
        s +=" "+w.lemma_
    return nlp(s)

def sent_vectorizer(sent, model):
    '''
    sentence vectorizer using the pretrained glove model
    '''
    sent_vector = np.zeros(200)
    num_w = 0
    for w in sent.split():
        try:
            # add up all token vectors to a sent_vector
            sent_vector = np.add(sent_vector, model[str(w)])
            num_w += 1
        except:
            pass
    return sent_vector

def load_data(csv_path):
    '''
    load data from csv to a data frame
    '''
    return pd.read_csv(csv_path, header=None, encoding="latin-1")

# def prepare_text_data(text_array, max_vocab=18000, max_len=15):
#     '''
#     prepare text data
#         inputs:
#             text_array - input text
#             max_vocab - number of vacob to keep
#             max_len - sequence length
            
#         returns:
#             data - prepared data
#             nb_words - total number of words in the data
            
    