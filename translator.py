import sys
import glob
from classes.base import Seq2Seq
from keras.preprocessing.text import text_to_word_sequence


class BisayaSeqToSeq(Seq2Seq):
	def load_data(self):
		data = []
		with open('data/translate/beki-bisaya.txt') as file:
			for line in list(file):
				arr = line.split('|')
				data.append((text_to_word_sequence(arr[0].strip()), text_to_word_sequence(arr[1].strip())))

		return data

def main():


	if 'train' == sys.argv[1]:
		agent = BisayaSeqToSeq(epochs=100)
		model = agent.train()
		model.save('.models/translator.h5')
	elif 'play' == sys.argv[1]:
		agent = BisayaSeqToSeq(model_filename='.models/translator.h5')


	while True:
		out, unk = agent.evaluate(text_to_word_sequence(input('Input: ')))
		if 0 == len(unk):
			print('Output: '+' '.join(out))
		else:
			print('unable to translate')

main()