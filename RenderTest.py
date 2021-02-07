import numpy as np
import gym
import gym_cart

env = gym.make('cart-v0')
count = 0

while True:
    env.render()
    # action = [np.random.randint(-1, 1), np.random.randint(-1, 1)]
    if count < 475:
        action = [200, 190]  # forca, forceb currently from 0 to 200 that correspond from -1 to 1
    else:
        action = [190, 200]  # forca, forceb currently from 0 to 200 that correspond from -1 to 1
        if count == 950:
            count = 0
    s_, r, done, info = env.step(action)
    count = count + 1

