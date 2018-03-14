import pandas as pd
import sys
import re
import pickle
from keras.preprocessing.text import Tokenizer, one_hot

# df = pd.read_csv('/home/dmabelin/freecodecamp_casual_chatroom.csv', usecols=['fromUser.id','text'])
# print('CSV Loaded {} rows'.format(df.shape[0]))

# SEQLEN = 10

# texts = []
# last_user = ''
# message = ''
# encoder = []
# decoder = []
# target = []
# tok = Tokenizer()

# for idx, row in df.iterrows():

# 	#exclude last
# 	if idx >= df.shape[0]-1:
# 		break

# 	if pd.isna(row['text']):
# 		continue
# 	message = message + row['text']
# 	if last_user != row['fromUser.id']:
# 		tok.fit_on_texts([message])
# 		texts.append(message)
# 		message = ''
# 		if len(texts) > 1:

# 			encoder.append(texts[0])
# 			decoder.append(texts[1])
# 			target.append(texts[1])
# 			texts.pop(0)

# 	else:
# 		last_user = row['fromUser.id']

# 	if idx > 10:
# 		break
# encoder = tok.texts_to_sequences(encoder)
# decoder = tok.texts_to_sequences(decoder)
# target = tok.texts_to_sequences(target)
# for idx in range(len(encoder)):
# 	print(encoder[idx], decoder[idx], target[idx])


# create word_index
tok = Tokenizer(oov_token='***')
tok.fit_on_texts(['there hello'])
seq = tok.texts_to_sequences(['hello there dude'])
# texts to numeric
print(seq)
print(tok.word_index)
