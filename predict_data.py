# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 22:16:38 2020

@author: Sylwek Szewczyk
"""

import keras, nltk, json, random, pickle, numpy as np
from nltk.stem import WordNetLemmatizer
from keras.models import load_model

model = load_model('model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

class Chatbot:
    
    def __init__(self):
        pass
    
    def clean_up_sentence(self, sentence):
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
        return sentence_words
