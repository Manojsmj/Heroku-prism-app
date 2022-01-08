import numpy as np
import pandas as pd
import re
import string
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from itertools import chain
from nltk.stem import WordNetLemmatizer
wl=WordNetLemmatizer()
from nltk.stem import PorterStemmer
ps=PorterStemmer()
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

def cleanText(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    lst = [wl.lemmatize(w, pos = "a") for w in text]
    return lst
input_sentence='this is mj'
MWUC_sentence='this is manoj'
sen_1 = cleanText(input_sentence)
sen_2 = cleanText(MWUC_sentence)
model = keras.models.load_model(r'C:\Users\HP\Desktop\Prism\new\model.h5')
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sen_1)
tokenizer.fit_on_texts(sen_2)
X_train_1_seq = tokenizer.texts_to_sequences(sen_1)
X_test_1_seq = tokenizer.texts_to_sequences(sen_2)
sequence_1 = list(chain.from_iterable(X_train_1_seq))
sequence_2 = list(chain.from_iterable(X_test_1_seq))
pad_sequence_1 = pad_sequences([sequence_1], 50)
pad_sequence_2 = pad_sequences([sequence_2], 50)
preds = model.predict([pad_sequence_1, pad_sequence_2])
avg = np.average(preds)
label = 1 if avg > 0.5 else 0
print('label :', label)