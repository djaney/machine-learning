import random
import gym
import numpy as np
import tensorflow as tf
import numpy
from keras.models import Model
from keras.layers import Input, Dense
class Agent(object):
    """The world's simplest agent!"""
    def __init__(self, strain_count = 10, mutation_chance=0.001, crossover_points = 1):
        self.strains = []
        self.nextGen = []
        self.strain_count = strain_count
        self.crossover_points = crossover_points
        self.model = self.createModel()
        self.mutation_chance = mutation_chance
        self.best = 0
        self.shape = self.model.get_weights()[0].shape
        for _ in range(self.strain_count):
            self.strains.append(self.createModel().get_weights())

    def generationSize(self):
        return len(self.strains)

    def act(self, observation, strainIndex):
        self.model.set_weights(self.strains[strainIndex])
        y = self.model.predict(np.array([observation]), batch_size=1)
        res = numpy.argmax(y[0])
        return res,self.strains[strainIndex]

    def next(self, reward, strain):
        self.nextGen.append((reward,strain[0].flatten()))

    def createModel(self):
        a = Input(shape=(4,))
        b = Dense(2)(a)
        return Model(inputs=a, outputs=b)

    def getBestReward(self):
        return self.best

    def evolve(self):

        # find fittest
        self.nextGen = sorted(self.nextGen, key=lambda item: item[0], reverse=True)
        # breed
        father = self.nextGen[0][1]
        mother = self.nextGen[1][1]

        self.best = self.nextGen[0][0]

        self.strains = []
        self.nextGen = []

        # add the best strain back into the pool
        self.strains.append([numpy.reshape(father, self.shape)])

        crossover_point_locations = []
        while self.crossover_points > len(crossover_point_locations):
            loc = random.randrange(len(father))
            if loc not in crossover_point_locations:
                crossover_point_locations.append(loc)


        switch = False

        for _ in range(self.strain_count):
            newStrain = []
            for i in range(len(father)):

                if i in crossover_point_locations:
                    switch = not switch


                if switch:
                    newStrain.append(father[i])
                else:
                    newStrain.append(mother[i])

                # a chance to mutate
                if random.random() < self.mutation_chance:
                    mIdx = random.randrange(0, len(newStrain))
                    newStrain[mIdx] = random.uniform(-1, 1)

            newStrain = numpy.reshape(newStrain, self.shape)
            self.strains.append([newStrain])




    
env = gym.make('CartPole-v1')
agent = Agent(strain_count=10, crossover_points=3)

done = False
generationSize = 10
iterations = 100

# learn
total_episodes = 0
while True:
    for strainIndex in range(agent.generationSize()):
        ob = env.reset()
        reward_sum = 0
        while True:
            action = agent.act(ob, strainIndex)
            ob, reward, done, info = env.step(action[0])
            reward_sum=reward_sum+reward
            if done:
                agent.next( reward_sum, action[1])
                total_episodes=total_episodes+1
                break
    agent.evolve()

    print(total_episodes, agent.getBestReward())

    if 475 <= agent.getBestReward():
        break

# View
ob = env.reset()
while True:
    env.render()
    action = agent.act(ob, 0)
    ob, reward, done, info = env.step(action[0])