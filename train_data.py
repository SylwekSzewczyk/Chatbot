# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 21:20:21 2020

@author: Sylwek Szewczyk
"""

import nltk, json, pickle, random
from nltk.stem import WordNetLemmatizer
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD

data_file = open('intents.json').read()
intents = json.loads(data_file)
words = []
classes = []
documents = []
ignore = ['?', '!']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        
        documents.append((w, intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

lemmatizer = WordNetLemmatizer()

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore]
words = sorted(set(words))
classes = sorted(set(classes))

pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(words, open('classes.pkl', 'wb'))