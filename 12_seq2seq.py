import sys
from classes.base import Seq2Seq

class BisayaSeqToSeq(Seq2Seq):
	def load_data(self):
		data = []
		data.append((list('what is your name'), list('unsa imong pangalan')))
		data.append((list('what is your gender'), list('lalaki ka o babae')))
		return data

def main():


	if 'train' == sys.argv[1]:
		agent = BisayaSeqToSeq(epochs=200)
		model = agent.train()
		model.save('.models/12_seq2seq.h5')
	elif 'play' == sys.argv[1]:
		agent = BisayaSeqToSeq(model_filename='.models/12_seq2seq.h5')


	while True:
		out, unk = agent.evaluate(list(input('Input: ')))
		print('Output: ',''.join(out))

main()