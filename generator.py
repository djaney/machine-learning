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
BATCH_SIZE = 100
char_to_int = dict((c, i) for i, c in enumerate(TOKENS))
int_to_char = dict((i, c) for i, c in enumerate(TOKENS))

def char_to_value(char):
    return char_to_int[char]

def value_to_char(value):
    return int_to_char[value]



def generator(path, batch_size):
	with open(path,'r') as f:
		data_buffer = ''
		x_buffer = []
		y_buffer = []
		while True:
			line = f.readline()
			if not line: break
			data_buffer = data_buffer + line
			data_buffer = ''.join([c for c in data_buffer if c in char_to_int])
			#flush, +1 to include forcasted character
			if len(data_buffer) > SEQ_LEN+1:
				data = data_buffer[0:SEQ_LEN+1]
				data_buffer = data_buffer[SEQ_LEN+1:]
				# data = [char_to_value(c) for c  in data]
				# data = [to_categorical(c,TOKEN_SIZE) for c  in data]
				x_buffer.append(data[0:-1])
				y_buffer.append(data[1:])
				if len(x_buffer) >= batch_size:
					x = x_buffer
					y = y_buffer
					x_buffer = []
					y_buffer = []
					yield x,y # no need to reset since they automatically reset




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
		for i in generator(model_path,SEQ_LEN):
			print(i)
		# model = create_model()
		# model.fit_generator(generator(model_path,BATCH_SIZE),steps_per_epoch=1000)
	return 0

sys.exit(_main(parser.parse_args()))