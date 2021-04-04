"""
Adapted from classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import gym
from gym import spaces, logger
from gym.utils import seeding
import numpy as np
from gym_cart.envs.CartRender import Cart, Obstacle, Goal


class CartEnv(gym.Env):
    """
    Description:
        Simple 2D cart motion along frictionless track. The cart starts at (0, 0) and wants to stay in bounds.
    Observation:
        Type: Box(3)
        Num     Observation                Min                    Max
        0       Cart Angle [theta]          ?                      ?
        1       Cart X Position [cm]      -500                    500
        2       Cart X Position [cm]      -500                    500
    Actions:
        Type: Discrete(9)
        Num   Action(L , R)
        0     -1 , -1: Backwards
        1     -1 , 0: Left Motor Backwards
        2     -1 , 1: Left Motor Backwards, Right Motor Forward
        3      0 , -1: Right Motor Backwards
        4      0 , 0: Stay still
        5      0 , 1: Right Motor Forward
        6      1 , -1: Left Motor Forward, Right Motor Backwards
        7      1 , 0: Left Motor Forward
        8      1 , 1: Forward
        Note: The amount the velocity that is increased is dependent on a scaling factor "power"
    Reward:
        Reward is 1 for every step taken, including the termination step
    Starting State: (?)
        All observations are assigned a uniform random value in [-0.05..0.05]
    Episode Termination:
        Any cart Position is more than +/- 10 (sides of the cart reach the edge of
        the display).
        Episode length is greater than 200. <-- did not set up yet
        Solved Requirements: <-- did not set up yet
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials. <-- did not set up yet
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, goal_x, goal_y, obst_x, obst_y):

        self.num_carts = 1  # define no of carts
        self.cart_length = 10.5  # distance between a and b
        self.num_obstacles = 1  # define no of obstacles

        # State Thresholds
        self.min_position = 0  # [cm]
        self.max_position = 50  # [cm]
        self.max_speed = 300  # need to understand scale better
        self.max_angle = 180  # not sure if this is correct

        self.low_state = np.array(
            [-self.max_angle, self.min_position, self.min_position]
        )
        self.high_state = np.array(
            [-self.max_angle, self.max_position, self.max_position]
        )
        # Goals
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.terminal_x = goal_x[-1]
        self.terminal_y = goal_y[-1]
        self.num_goals = 0
        self.goal_radius = 2

        # Obstacles
        self.obst_x = obst_x
        self.obst_y = obst_y
        self.obst_radius = 2

        # Action: 3 possible movements per cart
        self.action_space = spaces.Discrete(3*self.num_carts)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        # Reward info
        self.done = False
        self.crash = False

        # Render info
        self.screen_width = 500
        self.screen_height = 500

        self.world_width = self.max_position
        self.scale = self.screen_width / self.world_width

        self.seed()
        self.viewer = None

        # Initialize cart data
        self.state_start = np.array([0, 7.5, 10])
        self.state = self.state_start

        self.carts = []
        for i in range(self.num_carts):
            self.carts.append(Cart(self.scale, self.state_start, self.cart_length))

        self.steps = 0

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # for i, cart in enumerate(self.carts):
        #     cart.step(action[9 * i:9 * (i + 1)])

        # for i in range(self.num_carts):
        self.state = self.carts[0].step(action)
        theta, xp, yp = self.state

        # Scenario 1: Stay inside a given box
        # done = bool(
        #     xp < self.min_position
        #     or xp > self.max_position
        #     or yp < self.min_position
        #     or yp > self.max_position
        # )

        # Scenario 2a: Crashed into bounds or obstacle (neg reward)
        for i in range(len(self.obst_x)):
            self.crash = bool(
                xp - self.cart_length/2 < self.min_position
                or xp + self.cart_length/2 > self.max_position
                or yp - self.cart_length/2 < self.min_position
                or yp + self.cart_length/2 > self.max_position
                or abs(xp - self.obst_x[i]) <= self.obst_radius + self.cart_length/2 and abs(yp - self.obst_y[i]) <= self.obst_radius + self.cart_length/2
            )
            if self.crash:
                break

        # Scenario 2b: Reached single goal (pos reward)
        for i in range(len(self.goal_x)):
            goal = bool(
                abs(xp - self.goal_x[i]) <= self.goal_radius and abs(yp - self.goal_y[i]) <= self.goal_radius
            )
            if goal:
                self.num_goals += 1  # increase num_goals found
                self.goal_x = np.delete(self.goal_x, i)  # remove found goal from original list
                self.goal_y = np.delete(self.goal_y, i)
                break

        # Scenario 2c: Reached terminal goal (largest pos reward)
        self.done = bool(
            self.goal_x.size == 0
        )

        # Scenario 2d: Nothing (neg reward to improve time)

        # Initialize reward
        reward = 0.0

        if self.done:
            reward = 10.0*self.num_goals
        if not self.done:
            if self.crash:
                reward = -10.0
                self.done = self.crash
                # reward = - 1 + 1 / np.sqrt((goal_x[2] - xp)**2 + (goal_y[2] - yp)**2)
            if goal:
                reward = 10.0
                goal = False
            # if action is not 8:
            #     reward = -0.1
            # else:
            #     reward = -0.01

        if self.steps >= 10000:
            self.done = True

        # print(crash)

        self.steps += 1
        # print(self.num_goals)
        return np.array(self.state), reward, self.done, {}

    def reset(self, goal_x, goal_y):
        self.state = self.carts[0].reset()
        self.done = False
        self.crash = False
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.terminal_x = goal_x[-1]
        self.terminal_y = goal_y[-1]
        self.num_goals = 0
        self.steps = 0
        return np.array(self.state)

    def render(self, mode='human'):
        if self.state is not None:
            theta, xp, yp = self.state

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_width, self.screen_height)

            cart = self.carts[0].render()  # call method for Cart
            self.carttrans = rendering.Transform(translation=(50, 450))
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            obst = []
            for i in range(len(self.obst_x)):
                obst.append(Obstacle(None, None, self.obst_radius*self.scale))  # instantiate class instance
                obst[i] = obst[i].render()  # call method for Goal
                self.obsttrans = rendering.Transform(translation=(self.obst_x[i]*self.scale, self.obst_y[i]*self.scale))
                obst[i].set_color(255, 0, 0)
                obst[i].add_attr(self.obsttrans)
                self.viewer.add_geom(obst[i])

            goal = []
            for i in range(len(self.goal_x)):
                goal.append(Goal(None, None, self.goal_radius*self.scale))  # instantiate class instance
                goal[i] = goal[i].render()  # call method for Goal
                self.goaltrans = rendering.Transform(translation=(self.goal_x[i]*self.scale, self.goal_y[i]*self.scale))
                goal[i].set_color(0, 255, 0)
                goal[i].add_attr(self.goaltrans)
                self.viewer.add_geom(goal[i])

        if self.state is None:
            return None

        cartx = self.scale*xp
        carty = self.scale*yp
        self.carttrans.set_translation(cartx, carty)
        self.carttrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

