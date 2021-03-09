from abc import ABC

import numpy as np
# import polytope as pt
import gym
from gym.envs.classic_control import rendering


class Cart:
    def __init__(self, scale):
        self.length = 10.5  # distance between point a and b [cm]
        self.power = 60  # WOOD scaling factor
        self.tau = 0.01  # seconds between state updates INCREASE = FASTER MOVEMENTS
        self.thetaCurr = 0  # initial and current angle

        self.state = np.array([0, 50, 450])

        # self.cumstate.theta = []
        # self.cumstate.xp = []
        # self.cumstate.yp = []

        radius = np.array([3 * self.length / 2, self.length / 2, self.length / 2]) / scale
        res = 3  # 3-sided
        self.points = [np.array([np.cos(2 * np.pi * i / res) * radius[i], np.sin(2 * np.pi * i / res) * radius[i]])
                  for i in range(res)]
        # self.cart = pt.qhull(np.vstack([self.points[0], self.points[1], self.points[2]]))

    def step(self, action):
        theta, xp, yp = self.state  # define 3 states

        forcea = (action // 3) - 1
        forceb = (action % 3) - 1

        # scale accordingly
        va = forcea * self.power  # cm/s
        vb = forceb * self.power

        theta = self.thetaCurr + self.tau * (vb - va) / self.length

        xp = xp + 0.5*(va+vb)*np.cos(self.thetaCurr)*self.tau
        yp = yp + 0.5*(va+vb)*np.sin(self.thetaCurr)*self.tau

        # update state
        self.thetaCurr = theta
        self.state = (theta, xp, yp)

        # self.cumstate.theta.append(theta)
        # self.cumstate.xp.append(xp)
        # self.cumstate.yp.append(yp)

        return np.array(self.state)

    def reset(self):
        # self.state = np.array([0, 0, 0])
        pass

    def render(self):
        cart = rendering.FilledPolygon(self.points)
        return cart


class Obstacle:
    def __init__(self, obst_x, obst_y):
        self.obstacles = list()  # list of polytopes
        self.obst_x = obst_x
        self.obst_y = obst_y
        radius = 30.0
        res = 4  # 4-sided
        self.points = [(np.cos(2 * np.pi * i / res) * radius, np.sin(2 * np.pi * i / res) * radius)
                  for i in range(res)]

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
        obst = rendering.FilledPolygon(self.points)
        return obst

class Goal:
    def __init__(self, goal_x, goal_y):
        radius = 30.0
        res = 4  # 4-sided
        self.points = [(np.cos(2 * np.pi * i / res) * radius, np.sin(2 * np.pi * i / res) * radius)
                  for i in range(res)]

    def finished(self):
        pass

    def render(self):
        goal = rendering.FilledPolygon(self.points)
        return goal
