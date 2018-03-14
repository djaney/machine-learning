#!/usr/bin/python3
import sys
import argparse
from utils.downloader import get_file
from keras.models import Sequential
from keras.layers import LSTM, Dense, Activation, LeakyReLU


parser = argparse.ArgumentParser(description='Generate stuff.')
parser.add_argument('command')
parser.add_argument('-d','--data')
parser.add_argument('-m','--model')


def create_model():

	model = Sequential()
	model.add(LSTM(256,input_shape=(SEQLEN, ALPHASIZE), dropout=0.2, return_sequences=True))
	model.add(LeakyReLU(alpha=0.3))
	model.add(LSTM(256, dropout=0.2, return_sequences=False))
	model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(ALPHASIZE, activation='softmax'))
	#adam optimizer
	model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	return model

def _main(args):
	if 'train' == args.command:
		if args.data is None:
			parser.error('training data required')
		if args.model is None:
			parser.error('model required')
		print('asdasd')
	return 0

sys.exit(_main(parser.parse_args()))