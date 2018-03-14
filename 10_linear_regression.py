import numpy as np
from keras.models import Sequential
from keras.layers import Input, Dense, Activation

def create_model():
	model = Sequential()
	model.add(Dense(1, input_dim=1, activation='linear'))
	model.compile(loss='mean_absolute_error', optimizer='adam', metrics=['accuracy'])
	return model

model = create_model()


x = [i for i in range(1000)]
y = [i for i in range(1,1001)]
test = [i for i in range(200,300)]

model.fit(x=x, y=y, epochs=100)

