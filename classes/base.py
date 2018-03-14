import os
import numpy as np
from keras.models import Model, load_model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical


class Seq2Seq:
	
	def __init__(self,tokens,model_filename=None, epochs=100, encode_sequence_size=10, decode_sequence_size=10, internal_size=256):


		self.PAD = 0
		self.END = 1

		self.EPOCHS = epochs

		self.ENC_SEQ_SIZE = encode_sequence_size
		self.DEC_SEQ_SIZE = decode_sequence_size


		self.enc_tokens = []
		self.dec_tokens = []
		self.enc_token_map = []
		self.dec_token_map = []
		self.enc_token_size = 0
		self.dec_token_size = 0
		self.internal_size = internal_size
		
		self.load_tokens(tokens)

		if model_filename is not None and os.path.isfile(model_filename):
			self.model = load_model(model_filename)
		else:
			self.create_model()

	def enc_idx_to_val(self, idx):
		return self.enc_tokens[idx - 1]
	def enc_val_to_idx(self, val):
		return self.enc_token_map[val] + 1
	def dec_idx_to_val(self, idx):
		return self.dec_tokens[idx - 2]
	def dec_val_to_idx(self, val):
		return self.dec_token_map[val] + 2

	def data_batch(self, inp, out):

		# x,y,z encoder input, decoder input, decoder target
		x = []
		y = []
		z = []

		# remove if not in vocabulary

		enc = [self.enc_val_to_idx(i) for i in inp if i in self.enc_token_map.keys()]
		end_target = [self.dec_val_to_idx(i) for i in out if i in self.dec_token_map.keys()]
		end_target.insert(0, self.PAD)
		end_target.append(self.END)
		
		for i in range(len(end_target)-1):
			
			dec = end_target[0:i+1]
			tar = end_target[i+1]
			x.append(enc)
			y.append(dec)
			z.append(tar)
		x = np.array(pad_sequences(x, maxlen=self.ENC_SEQ_SIZE, value=self.PAD))
		y = np.array(pad_sequences(y, maxlen=self.DEC_SEQ_SIZE, value=self.PAD))
		
		x = to_categorical(x, num_classes=self.enc_token_size)
		y = to_categorical(y, num_classes=self.dec_token_size)
		z = to_categorical(z, num_classes=self.dec_token_size)

		return x,y,z

	def load_data(self):
		raise "implement"

	def load_tokens(self, tokens):
		if(tokens != None):
			self.enc_tokens = tokens[0]
			self.dec_tokens = tokens[1]

		#map
		self.enc_token_map = dict((c, i) for i, c in enumerate(self.enc_tokens))
		self.dec_token_map = dict((c, i) for i, c in enumerate(self.dec_tokens))
		self.enc_token_size = len(self.enc_tokens) + 1
		self.dec_token_size = len(self.dec_tokens) + 2 # add 2 for pad and end

	def data_all(self, data):
		x, y, z = self.data_batch(data[0][0],data[0][1])
		for d in data[1:]:
			_x, _y, _z = self.data_batch(d[0],d[1])
			x = np.concatenate((x,_x), axis=0)
			y = np.concatenate((y,_y), axis=0)
			z = np.concatenate((z,_z), axis=0)
		return x, y, z

	def train(self, data):

		x,y,z = self.data_all(data)
		return self.model.fit([x, y], z, epochs=self.EPOCHS,shuffle=False)

	def create_model(self):
		# Define an input sequence and process it.
		encoder_inputs = Input(shape=(None, self.enc_token_size))
		encoder = LSTM(self.internal_size, return_state=True, dropout=0.01)
		encoder_outputs, state_h, state_c = encoder(encoder_inputs)
		# We discard `encoder_outputs` and only keep the states.
		encoder_states = [state_h, state_c]

		# Set up the decoder, using `encoder_states` as initial state.
		decoder_inputs = Input(shape=(None, self.dec_token_size))
		# We set up our decoder to return full output sequences,
		# and to return internal states as well. We don't use the 
		# return states in the training model, but we will use them in inference.
		decoder_lstm = LSTM(self.internal_size, return_sequences=False, return_state=True, dropout=0.01)
		decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
		                                     initial_state=encoder_states)
		decoder_dense = Dense(self.dec_token_size, activation='softmax')
		decoder_outputs = decoder_dense(decoder_outputs)

		# Define the model that will turn
		# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
		self.model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
		self.model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])		

	def predict(self, inp):
		out = []
		unknown = []
		inp = list(set(inp) & set(self.enc_tokens))
		enc = [self.enc_val_to_idx(i) for i in inp if i in self.enc_tokens]
		for i in inp:
			if i in self.enc_tokens:
				enc.append(self.enc_val_to_idx(i))
			else:
				unknown.append(i)

		dec = [self.PAD] * self.DEC_SEQ_SIZE
		enc = np.array([to_categorical(enc, num_classes=self.enc_token_size)])
		dec = np.array([to_categorical(dec, num_classes=self.dec_token_size)])

		# predict first
		x = [enc,dec]
		pred = self.model.predict(x)
		pred_class = np.argmax(pred[0])
		out.append(self.dec_idx_to_val(pred_class))

		for _ in range(100):
			dec = [[self.dec_val_to_idx(i) for i in out]]
			dec = pad_sequences(dec, maxlen=self.DEC_SEQ_SIZE, value=self.PAD)
			dec = np.array(to_categorical(dec, num_classes=self.dec_token_size))

			x = [enc,dec]
			pred = self.model.predict(x)
			pred_class = np.argmax(pred[0])
			if pred_class == self.END:
				break
			out.append(self.dec_idx_to_val(pred_class))

		return out, unknown
	def load_weights(self, weights_file):
		self.model.load_weights(weights_file)