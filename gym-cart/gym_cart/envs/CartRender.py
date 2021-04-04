from abc import ABC

import numpy as np
# import polytope as pt
import gym
from gym.envs.classic_control import rendering


class Cart:
    def __init__(self, scale, state_start, cart_length):
        self.scale = scale
        self.length = cart_length  # distance between point a and b [dm]
        self.power = 10  # WOOD scaling factor
        self.tau = 0.01  # seconds between state updates INCREASE = FASTER MOVEMENTS
        self.thetaCurr = 0  # initial and current angle

        self.state_start = state_start
        self.state = self.state_start

        radius = np.array([1.5 * self.length / 2, self.length / 2, self.length / 2])*self.scale
        res = 3  # 3-sided
        self.points = [np.array([np.cos(2 * np.pi * i / res) * radius[i], np.sin(2 * np.pi * i / res) * radius[i]])
                  for i in range(res)]

    def step(self, action):
        theta, xp, yp = self.state  # define 3 states

        # forcea = (action // 3) - 1
        # forceb = (action % 3) - 1
        if action == 0:  # turn left
            forcea = 1
            forceb = 0
        if action == 1:  # turn right
            forcea = 0
            forceb = 1
        if action == 2:  # go forward
            forcea = 1
            forceb = 1

        # scale accordingly
        va = forcea * self.power  # cm/s
        vb = forceb * self.power

        theta = self.thetaCurr + self.tau * (vb - va) / self.length

        xp = xp + 0.5*(va+vb)*np.cos(self.thetaCurr)*self.tau
        yp = yp + 0.5*(va+vb)*np.sin(self.thetaCurr)*self.tau

        # update state
        self.thetaCurr = theta
        self.state = (theta, xp, yp)

        return np.array(self.state)

    def reset(self):
        self.state = self.state_start
        self.thetaCurr = 0
        return self.state

    def render(self):
        cart = rendering.FilledPolygon(self.points)
        return cart


class Obstacle:
    def __init__(self, obst_x, obst_y, obst_rad):
        self.obstacles = list()  # list of polytopes
        self.obst_x = obst_x
        self.obst_y = obst_y
        self.obst_rad = obst_rad

    def crash(self, cartx, carty):
        pass

    def render(self):
        # obst = np.array([])
        # for i in range(len(self.obst_x)):
        #     obst[i] = rendering.FilledPolygon(self.points)
        #     obsttrans = rendering.Transform(translation=(self.obst_x[i], self.obst_y[i]))
        #     obst[i].set_color(255, 0, 0)
        #     obst[i].add_attr(obsttrans)
        # return obst
        obst = rendering.make_circle(radius=self.obst_rad, res=30, filled=True)
        return obst

class Goal:
    def __init__(self, goal_x, goal_y, goal_rad):
        self.goal_rad = goal_rad

    def finished(self):
        pass

    def render(self):
        goal = rendering.make_circle(radius=self.goal_rad, res=30, filled=True)
        return goal
