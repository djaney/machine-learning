#!/usr/bin/python3
import sys
import argparse
from utils.downloader import get_file
from keras.models import Sequential,load_model
from keras.layers import LSTM, Dense, Activation, LeakyReLU
from keras.utils.np_utils import to_categorical
from keras.callbacks import Callback
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import math
import os
import glob
import time
import configparser

parser = argparse.ArgumentParser(description='Generate stuff.')
parser.add_argument('command')
parser.add_argument('config')
parser.add_argument('-d','--data')
parser.add_argument('-s','--steps_per_epochs', default=-1, type=int)
parser.add_argument('-b','--batch_size', default=1, type=int)
parser.add_argument('-e','--epochs', default=1, type=int)
parser.add_argument('--internal_size', default=256, type=int)
parser.add_argument('--model_depth', default=1, type=int)
parser.add_argument('--sequence_length', default=100, type=int)
parser.add_argument('--seed', default="A")
parser.add_argument('--length', default=1000, type=int)
parser.add_argument('--reset', action='store_true')


TOKENS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890-!:()\",.? \n\t"
TOKEN_SIZE = len(TOKENS)
BATCH_SIZE = 500
char_to_int = dict((c, i) for i, c in enumerate(TOKENS))
int_to_char = dict((i, c) for i, c in enumerate(TOKENS))


class Checkpoint(Callback):
	def __init__(self, path):
		self.path = path
		self.time = time.time()
		self.time_quick = time.time()
	def on_epoch_end(self, acc, loss):
		self.model.reset_states()
		print('States cleared')
	def on_epoch_begin(self, epoch, logs={}):
		self.epoch = epoch
	def on_batch_end(self, batch, logs={}):
		elapsed = math.floor(time.time() - self.time)
		elapsed_quick = math.floor(time.time() - self.time_quick)
		self.model.save('{}'.format(self.path))

		if elapsed_quick > 10:
			self.model.save('{}'.format(self.path))
			self.time_quick = time.time()
		if elapsed > 1600: # save every 30 minutes
			self.time = time.time()
			# transfer training weights to saved model
			self.model.save('{}.{:05d}-{:05d}.h5'.format(self.path,self.epoch,batch))

def char_to_value(char):
    return char_to_int[char]

def value_to_char(value):
    return int_to_char[value]

def get_file_size(path):
	size = 0
	with open(path,'r') as f:
		while True:
			line = f.readline()
			if not line: break
			size = size + len([c for c in line if c in char_to_int])

	return size

def data_generator(path, batch_size, seq_len):
	data_buffer = ''
	x_buffer = []
	y_buffer = []
	while True:
		with open(path,'r') as f:
			while True:
				line = f.readline()
				if not line: break
				data_buffer = data_buffer + line
				data_buffer = ''.join([c for c in data_buffer if c in char_to_int])
				#flush, +1 to include forcasted character
				if len(data_buffer) > seq_len+1:
					data = data_buffer[0:seq_len+1]
					data_buffer = data_buffer[seq_len+1:]
					data = [char_to_value(c) for c  in data]
					data = [to_categorical(c,TOKEN_SIZE) for c  in data]
					x_buffer.append(data[0:-1])
					y_buffer.append(data[-1])
					if len(x_buffer) >= batch_size:
						x = x_buffer
						y = y_buffer
						x_buffer = []
						y_buffer = []
						yield np.array(x),np.array(y)




def create_model(internal_size,model_depth, batch_size,seq_len):

	model = Sequential()

	for i in range(model_depth):
		return_sequences = False if i == model_depth-1 else True
		if i == 0:
			model.add(LSTM(internal_size,
				batch_input_shape=(batch_size,seq_len, TOKEN_SIZE), 
				dropout=0.2, 
				return_sequences=return_sequences, 
				stateful=True))
			model.add(LeakyReLU(alpha=0.3))
		else:
			model.add(LSTM(internal_size, dropout=0.2, return_sequences=return_sequences, stateful=True))
			model.add(LeakyReLU(alpha=0.3))
	model.add(Dense(TOKEN_SIZE, activation='softmax'))
	#adam optimizer
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def play(model_path, seed, length, internal_size, model_depth):
	model_train = load_model(model_path)
	model = create_model(internal_size,model_depth, 1,1)

	words = ''.join([s for s in seed if s in char_to_int]) # start with a capital letter
	for c in seed[:-1]:
		c = char_to_value(c)
		c = to_categorical(c,TOKEN_SIZE)
		c = np.resize(c, (1,1,TOKEN_SIZE))
		model.predict(c,batch_size=1)

	for _ in range(length):
		trimmed = words[-1:]
		trimmed = [char_to_value(x) for x in trimmed]
		trimmed = pad_sequences([trimmed],maxlen=1)
		trimmed = to_categorical(trimmed,TOKEN_SIZE)
		trimmed = np.resize(trimmed, (1,1,TOKEN_SIZE))
		pred = model.predict(trimmed,batch_size=1).flatten()
		out = np.argmax(pred)
		char = value_to_char(out)
		words = words + char
	model.reset_states()
	print(words)

def train(model_path, data_path, steps_per_epoch, epochs, batch_size, sequence_length, internal_size, model_depth, reset):
	if data_path is None:
		parser.error('training data required')
	data_path = get_file(data_path)

	if not reset and os.path.exists(model_path):
		# get weights from saved model and apply to newly created model
		model = load_model(model_path)
	else:
		model = create_model(internal_size,model_depth, batch_size,sequence_length)

	size = get_file_size(data_path)
	max_epoch = math.floor(size / sequence_length / batch_size)
	
	if steps_per_epoch > max_epoch or steps_per_epoch < 0: steps_per_epoch = max_epoch

	# remove checkpoints
	for f in glob.glob(model_path+".*.h5"):
		os.remove(f)

	# instantiate checkpoint callback
	saver = Checkpoint(model_path)
	model.fit_generator(data_generator(data_path,batch_size, sequence_length),
		steps_per_epoch=steps_per_epoch, 
		shuffle=False, 
		epochs=epochs, 
		callbacks=[saver])
	model.save(model_path)
def _main(args):

	config = configparser.ConfigParser()
	config.readfp(open(args.config))
	model_path = config.get('Default','model')
	data_path = config.get('Default', 'data');
	steps_per_epoch = config.getint('Default', 'steps_per_epoch')
	epochs = config.getint('Default', 'epochs')
	batch_size = config.getint('Default', 'batch_size')
	sequence_length = config.getint('Default', 'sequence_length')
	internal_size = config.getint('Default', 'internal_size')
	model_depth = config.getint('Default', 'model_depth')


	if 'train' == args.command:
		train(model_path, data_path, steps_per_epoch, epochs, batch_size, sequence_length, internal_size, model_depth, args.reset)
	elif 'play' == args.command:
		play(model_path, args.seed, args.length, internal_size, model_depth)

	return 0

sys.exit(_main(parser.parse_args()))