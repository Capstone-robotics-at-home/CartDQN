import numpy as np
import gym
import gym_cart

env = gym.make('cart-v0')


while True:
    env.render()
    # action = [np.random.randint(-1, 1), np.random.randint(-1, 1)]
    action = [1, 0.9]
    s_, r, done, info = env.step(action)

