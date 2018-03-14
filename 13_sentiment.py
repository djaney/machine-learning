#!/usr/bin/python3
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
from keras.utils import Sequence
import numpy as np
import math
vocab_size = 50
doc_size = 10

class DatSequence(Sequence):

	def __init__(self,doc_size,vocab_size):
		self.x = [
			"very good",
			"good",
			"alright",
			"awesome",
			"wonderful",
			"terrible",
			"dissapointing",
			"horrifying",
			"disgusting",
			"very bad"
		]
		self.y = [1,1,1,1,1,0,0,0,0,0]
		self.batch_size = 2
		self.doc_size = doc_size
		self.vocab_size = vocab_size

	def __len__(self):
		return math.ceil(len(self.x) / self.batch_size)

	def __getitem__(self, idx):
		batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]
		batch_x = [one_hot(x, self.vocab_size) for x in batch_x]
		batch_x = pad_sequences(batch_x,maxlen=self.doc_size)
		return np.array(batch_x), np.array(batch_y)



model = Sequential()
model.add(Embedding(vocab_size, 3, input_length=doc_size))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile('rmsprop', loss='binary_crossentropy', metrics=['accuracy'])
model.fit_generator(DatSequence(doc_size,vocab_size), epochs=50)

print(model.summary())