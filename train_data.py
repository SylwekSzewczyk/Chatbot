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


'''
Data preparation segment

'''
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
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0]*len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training)
x_train = list(training[:,0])
y_train = list(training[:,1])

'''

Deep learning model application

'''

model = Sequential()
model.add(Dense(128, input_shape=(len(x_train[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]), activation = 'softmax'))

sgd = SGD(lr=0.01, momentum = 1e-6, decay = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = sgd, metrics = ['accuracy'])

model_fit = model.fit(np.array(x_train), np.array(y_train), epochs = 200, batch_size = 5, verbose = 1)
model.save('model.h5', model_fit)
