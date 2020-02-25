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
    
    def bag_of_words(self, sentence, words):
        
        sentence_words = self.clean_up_sentence(sentence)
        bag = [0]*len(words)
        for s in sentence_words:
            for i, w in enumerate(words):
                if w == s:
                    bag[i] = 1
        return np.array(bag)
    
    def predict_class(self, sentence, model):
        
        bag = self.bag_of_words(sentence, words)
        res = model.predict(np.array([bag]))[0]
        THRESHOLD = 0.25
        results = [[i,r] for i, r in enumerate(res) if r>THRESHOLD]
        results.sort(key = lambda x: x[1], reverse = True)
        return_list = []
        for r in results:
            return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
        return return_list