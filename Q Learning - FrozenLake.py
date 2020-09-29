# This is one of the first reinforcement/Q-learning codes I made using gym.
# Please note that this code may be broken now since gym stopped woking on my laptop.

import gym
import numpy as np
import random
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v0')
env.reset()

q_table = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.2
gamma = 0.6
epsilon = 0.7
total_episodes = 10000



for episode in range(total_episodes):
    state = env.reset()
    done = False
    j = 0
    while j < 99:
        if random.uniform(0,1) < epsilon:
            action = np.random.randint(env.action_space.n)
        else:
            action = np.argmax(q_table[state])
        j+=1
        state2, reward, done , info = env.step(action)
        old_value = q_table[state,action]
        next_max = max(q_table[state2])
        q_table[state,action] = (1-alpha)* old_value + alpha*(reward+ gamma*next_max)
        state = state2
        if done == True:
            print('Sucess on episode', episode)
            break


print('Training completed')
print(q_table)




