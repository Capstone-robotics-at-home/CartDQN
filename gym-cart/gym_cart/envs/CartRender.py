from abc import ABC

import numpy as np
import gym
from gym.envs.classic_control import rendering


class Cart:
    def __init__(self):
        radius = [25.0, 10.0, 10.0]
        res = 3  # 3-sided
        points = [(np.cos(2 * np.pi * i / res) * radius[i], np.sin(2 * np.pi * i / res) * radius[i])
                  for i in range(res)]
        self.cart = rendering.FilledPolygon(points)
        self.cumstate.theta = []
        self.cumstate.xp = []
        self.cumstate.yp = []

    def step(self):

        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        theta, xp, yp = self.state  # define 7 states

        forcea = (action // 3) - 1
        forceb = (action % 3) - 1

        # scale accordingly
        va = forcea * self.power  # cm/s
        vb = forceb * self.power

        self.thetaCurr = self.thetaCurr + self.tau * (vb - va) / self.length

        xp = xp + 0.5*(va+vb)*math.cos(self.thetaCurr)*self.tau
        yp = yp + 0.5*(va+vb)*math.sin(self.thetaCurr)*self.tau

        # update state
        self.thetaCurr = theta
        self.state = (theta, xp, yp)

        self.cumstate.theta.append(theta)
        self.cumstate.xp.append(xp)
        self.cumstate.yp.append(yp)

    def reset(self):
        pass

    def new(self):
        return self.cart


class Obstacle:
    def __init__(self):
        radius = 50.0
        res = 4  # 4-sided
        points = [(np.cos(2 * np.pi * i / res) * radius, np.sin(2 * np.pi * i / res) * radius)
                  for i in range(res)]
        self.obstacle = rendering.FilledPolygon(points)

    def new(self):
        return self.obstacle


class Goal:
    def __init__(self):
        radius = 50.0
        res = 5  # star
        points = [(np.cos(2 * np.pi * i / res) * radius, np.sin(2 * np.pi * i / res) * radius)
                  for i in range(res)]
        points = [points[0], points[2], points[4], points[1], points[3]]
        self.goal = rendering.make_polygon(points, False)

    def new(self):
        return self.goal
