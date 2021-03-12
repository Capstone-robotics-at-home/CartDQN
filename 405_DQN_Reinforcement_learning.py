"""
View more, visit my tutorial page: https://mofanpy.com/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://mofanpy.com/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.4
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gym
import gym_cart
import matplotlib
from matplotlib import pyplot as plt

# Values from Camera Vision
goal_x = np.array([200, 120, 350])
goal_y = np.array([450, 200, 300])
obst_x = np.array([50])
obst_y = np.array([50])

# Hyper Parameters
BATCH_SIZE = 32             # batch size: default 32, 1 to 100+
LR = 0.001                   # learning rate: default 0.01, 0 longer training to 1
EPSILON = 0.9               # greedy policy: default 0.9, change to 1 after training or add decay rate
GAMMA = 0.999                 # reward discount: 0 shortsighted to 1 farsighted
TARGET_REPLACE_ITER = 10   # target update frequency
MEMORY_CAPACITY = 10
N_EPS = 10000
env = gym.make('cart-v0', goal_x=goal_x, goal_y=goal_y, obst_x=obst_x, obst_y=obst_y)
env = env.unwrapped
N_ACTIONS = env.action_space.n
N_STATES = env.observation_space.shape[0]
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape  # confirm the shape

reward = np.zeros(N_EPS)

# dqn = DQN()
# dqn.eval_net=torch.load('Cart.pkl')


class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = Net(), Net()
        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = torch.FloatTensor(b_memory[:, :N_STATES])
        b_a = torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])
        b_s_ = torch.FloatTensor(b_memory[:, -N_STATES:])

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


dqn = DQN()

print('\nCollecting experience...')
for i_episode in range(N_EPS):
    s = env.reset(goal_x, goal_y)
    ep_r = 0
    while True:
        # env.render()
        a = dqn.choose_action(s)
        # print(a)
        # take action
        s_, r, done, info = env.step(a)
        # print(r)

        # modify the reward

        theta, xp, yp = s_
        # r1 = (env.max_position - abs(xp)) / env.max_position
        # r2 = (env.max_position - abs(yp)) / env.max_position
        # r = r1 + r2
        dqn.store_transition(s, a, r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            # if done:
            #     print('Ep: ', i_episode,
            #           '| Ep_r: ', round(ep_r, 2))

        if done:
            reward[i_episode] = r
            break
        s = s_
        # print(s)

eps = np.linspace(1, N_EPS, N_EPS)
plt.plot(eps, reward)
# print ave reward also
plt.ylabel('Reward')
plt.xlabel('Eps')
plt.show()
# torch.save(dqn.eval_net, 'Cart.pkl')
