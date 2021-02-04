import numpy as np
import gym
import gym_cart

env = gym.make('cart-v0')


while True:
    env.render()
    # action = [np.random.randint(-1, 1), np.random.randint(-1, 1)]
    action = [99, 99] # forca, forceb currently from 0 to 100 that correspond from -1 to 1
    s_, r, done, info = env.step(action)

