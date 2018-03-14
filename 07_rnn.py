#!/usr/bin/python3
import sys
import numpy as np
import glob
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Activation, LeakyReLU
from keras.optimizers import Adam
from keras.utils.np_utils import to_categorical
import random
import os
import pickle
CHARMAP = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-!:()\",.? \n\t"
# CHARMAP = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-!:()\",.? "

SEQLEN = 100
BATCHSIZE = 102
4ALPHASIZE = len(CHARMAP)
FILES = "shakespeare/*.txt"
EPOCHS = 1
TRIAL_FILE = '.trials/07_rnn';
MODEL_FILE = '.models/07_rnn.h5';

## Data related stuff
char_to_int = dict((c, i) for i, c in enumerate(CHARMAP))
int_to_char = dict((i, c) for i, c in enumerate(CHARMAP))
        
def char_to_value(char):
    return char_to_int[char]

def char_to_class_map(char):
    value = char_to_value(char)
    if value is None: return None
    return to_categorical(value,ALPHASIZE)
    
def value_to_char(value):
    return int_to_char[value]

def ohe_to_string(res):
    words = ''
    for r in res:
        words = words + value_to_char(np.argmax(r))
    return words

# iterate every single file
def get_file_data(pattern, index):
    paths = glob.glob(pattern)
    length = len(paths)
    
    if index < length:
        data = ''
        with open(paths[index], "r") as file:
            data = file.read()
            data = filter_string(data)
            data = [char_to_class_map(l) for l in data]
        return data
    else:
        return None

# get batch data in file
def build_batch_data(file_data, seqlen, batch_index, batch_count):
    length = len(file_data)
    start = batch_index * batch_count
    end = start+seqlen
    x = []
    y = []
    while end < length and (len(x) < batch_count or 0 > batch_count):
        x_line = file_data[start:end]
        y_line = file_data[end]
        x.append(x_line)
        y.append(y_line)
        start = start + 1
        end = start + seqlen
    x = np.array(x)
    y = np.array(y)
    return x,y


def create_model():
    if os.path.isfile(MODEL_FILE):
        model = load_model(MODEL_FILE)
    else:
        model = Sequential()
        model.add(LSTM(256,input_shape=(SEQLEN, ALPHASIZE), dropout=0.2, return_sequences=True))
        model.add(LeakyReLU(alpha=0.3))
        model.add(LSTM(256, dropout=0.2, return_sequences=False))
        model.add(LeakyReLU(alpha=0.3))
        model.add(Dense(ALPHASIZE, activation='softmax'))
        #adam optimizer
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def run_trial(length, sample, print_output=False):
    model = load_model(MODEL_FILE)
    words = filter_string(sample).rjust(SEQLEN) # start with a capital letter

    for _ in range(length):
        trimmed = words[-SEQLEN:]
        res = [char_to_class_map(x) for x in trimmed]
        res = np.array([res])
        new_res = model.predict(res)
        words = words + ohe_to_string(new_res)

    words = words.lstrip(' ')

    if print_output:
        print(words)
    else:
        with open('{}.txt'.format(TRIAL_FILE),'w')  as file:
            file.write(words) 

def get_sample(file):
    data = ''
    with open(file, "r") as file:
        while len(data) < SEQLEN:
            data = data + filter_string(file.read(SEQLEN-len(data)))
    return data

def filter_string(s):
    return ''.join([c for c in s if c in CHARMAP])




# run_trial(1000, get_sample('shakespeare/1kinghenryiv.txt'))
def train():

    model = create_model()

    if os.path.isfile(MODEL_FILE+'.pkl'):
        with open(MODEL_FILE+'.pkl', 'rb') as f:
            fileIndex, idx, batchNumber = pickle.load(f)
        recovery = True
    else:
        recovery = False

    model.save(MODEL_FILE)



    if not recovery:
        fileIndex = 0
        batchNumber = 1
    for fileIndex in range(fileIndex, 42):
        file_data = get_file_data(FILES, fileIndex)
        if not recovery:
            idx = 0
        else:
            recovery = False
        while True:
            x,y = build_batch_data(file_data, SEQLEN, idx ,BATCHSIZE)
            print('File #'+str(fileIndex+1)+' Batch #'+str(batchNumber+1))
            if 0 == len(x):
                break

            model.fit(x, y, epochs=EPOCHS, batch_size=len(x), shuffle=False)


            model.save(MODEL_FILE)
            with open(MODEL_FILE+'.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
                pickle.dump([fileIndex, idx, batchNumber], f)

            # if 0 == batchNumber % 100:
            paths = glob.glob(FILES)
            # if batchNumber % 10:
            run_trial(1000, get_sample(paths[fileIndex]))

            idx = idx + 1
            batchNumber = batchNumber + 1
        

        fileIndex=fileIndex+1
    if os.path.isfile(MODEL_FILE+'.pkl'):
        os.remove(MODEL_FILE+'.pkl')


if 'train' == sys.argv[1]:
    train()
elif 'generate' == sys.argv[1]:
    title = sys.argv[3]
    author = sys.argv[4]

    seed = "\t{}\n\n\t{}\n\n".format(title, author)
    run_trial(int(sys.argv[2]), seed, print_output=True)