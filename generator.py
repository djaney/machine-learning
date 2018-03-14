#!/usr/bin/python3
import sys
import argparse
from utils.downloader import get_file
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, LeakyReLU
from keras.utils.np_utils import to_categorical

parser = argparse.ArgumentParser(description='Generate stuff.')
parser.add_argument('command')
parser.add_argument('-d','--data')
parser.add_argument('-m','--model')

TOKENS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-!:()\",.? \n\t"
TOKEN_SIZE = len(TOKENS)
SEQ_LEN = 100
char_to_int = dict((c, i) for i, c in enumerate(TOKENS))
int_to_char = dict((i, c) for i, c in enumerate(TOKENS))

def char_to_value(char):
    return char_to_int[char]

def value_to_char(value):
    return int_to_char[value]



def generator(path,size):
	with open(path,'r') as f:
		while True:
			line = f.readline() + "\n"
			# code here




def create_model():

	model = Sequential()
	model.add(LSTM(256,input_shape=(SEQ_LEN, TOKEN_SIZE), dropout=0.2, return_sequences=True))
	model.add(LeakyReLU(alpha=0.3))
	model.add(LSTM(256, dropout=0.2, return_sequences=False))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(TOKEN_SIZE, activation='softmax'))
	#adam optimizer
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model



def _main(args):
	if 'train' == args.command:
		if args.data is None:
			parser.error('training data required')
		if args.model is None:
			parser.error('model required')
		model_path = get_file(args.data)
		model = create_model()
		model.fit_generator(generator(model_path, 100),steps_per_epoch=100)
	return 0

sys.exit(_main(parser.parse_args()))