#!/usr/bin/python3
import sys
import pandas as pd
import pickle
import matplotlib.pyplot as plt  
from classes.base import Seq2Seq
from keras.preprocessing.text import text_to_word_sequence
from keras.utils import plot_model

def main():
	if 'train' == sys.argv[1]:
		model, history = train()
		if len(sys.argv) == 3 and sys.argv[2] == 'plot':
			print('Plotting chart')
			plot_model(model, to_file='.graphs/chatbot.png')
			plot(history)
	elif 'play' == sys.argv[1]:
		play()

def play():
	with open('data/cornell/data.pkl', 'rb') as file:
		conversations, tokens = pickle.load(file)

	agent = Seq2Seq((tokens,tokens),internal_size=256,epochs=1)
	agent.load_weights('.models/chatbot.weights.h5')
	# out, unk = agent.predict([])
	# print('<Bot>: '+' '.join(out))
	while True:
		inp = input('<You>: ')
		out, unk = agent.predict(text_to_word_sequence(inp))
		if 0 == len(unk):
			print('<Bot>: '+' '.join(out))
		else:
			print('<Bot>: Sorry, I don\'t know ' + inp)

def train():
	data = []
	
	with open('data/cornell/data.pkl', 'rb') as file:
		conversations, tokens = pickle.load(file)
	print('{} tokens'.format(len(tokens)))
	agent = Seq2Seq((tokens,tokens),internal_size=256,epochs=1)

	client_text = []
	for c in conversations:
		for l in c:
			agent_text = l
			data.append((client_text,agent_text))
			client_text = agent_text

		history = agent.train(data)
		model = agent.model
		data = []
		model.save_weights('.models/chatbot.weights.h5')
		



	out = ''
	inp = None

	while True:

		inp = input('<You>: ')

		if inp == '':
			break
			

		out, unk = agent.predict(text_to_word_sequence(inp))
		if 0 == len(unk):
			print('<Bot>: '+' '.join(out))
		else:
			print('<Bot>: Sorry, I don\'t know ' + inp)

	return (model, history)


def plot(history):
	plt.figure(1)  

	# summarize history for accuracy  

	plt.subplot(211)  
	plt.plot(history.history['acc'])  
	plt.plot(history.history['val_acc'])  
	plt.title('model accuracy')  
	plt.ylabel('accuracy')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'test'], loc='upper left')  

	# summarize history for loss  

	plt.subplot(212)  
	plt.plot(history.history['loss'])  
	plt.plot(history.history['val_loss'])  
	plt.title('model loss')  
	plt.ylabel('loss')  
	plt.xlabel('epoch')  
	plt.legend(['train', 'test'], loc='upper left')  
	plt.show() 

main()