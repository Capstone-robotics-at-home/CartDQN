import numpy as np
import matplotlib.pyplot as plt
import gym
import gym_cart

env = gym.make('cart-v0')
# count = 0
x = [0]
y = [0]
theta = [0]
t = np.linspace(0, 499, 500)

for i in range(int(t[-1])):
    # action = int(input("Enter a command from 0 to 8: "))
    action = 8
    # env.render()
    # action = [np.random.randint(-1, 1), np.random.randint(-1, 1)]
    # if count < 475:
    #     action = 0  # forcea, forceb currently from 0 to 200 that correspond from -1 to 1
    # else:
    #     action = 8  # forcea, forceb currently from 0 to 200 that correspond from -1 to 1
    #     if count == 950:
    #         count = 0
    s, r, done, info = env.step(action)
    x.append(s[1])
    y.append(s[2])
    # count = count + 1

plt.plot(t, x, label='x')
plt.plot(t, y, linestyle='--', label='y')
# plt.plot(x, y)
plt.plot(x[0], y[0], '*')
plt.title('Action ' + str(action) + " Forward")
plt.xlabel('Time [s]')
plt.ylabel('Location [cm]')
plt.grid()
plt.legend()
plt.show()

