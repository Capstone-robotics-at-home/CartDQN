from abc import ABC

import numpy as np
import gym
from gym.envs.classic_control import rendering


class Cart:
    def __init__(self):
        radius = [75.0, 50.0, 50.0]
        res = 3  # 3-sided
        points = [(np.cos(2 * np.pi * i / res) * radius[i], np.sin(2 * np.pi * i / res) * radius[i])
                  for i in range(res)]
        self.cart = rendering.FilledPolygon(points)

    def new(self):
        return self.cart


class Obstacle:
    def __init__(self):
        radius = 50.0
        res = 4  # 3-sided
        points = [(np.cos(2 * np.pi * i / res) * radius, np.sin(2 * np.pi * i / res) * radius)
                  for i in range(res)]
        self.obstacle = rendering.FilledPolygon(points)

    def new(self):
        return self.obstacle


class Goal:
    def __init__(self):
        radius = 50.0
        res = 5  # 3-sided
        points = [(np.cos(2 * np.pi * i / res) * radius, np.sin(2 * np.pi * i / res) * radius)
                  for i in range(res)]
        points = [points[0], points[2], points[4], points[1], points[3]]
        self.goal = rendering.make_polygon(points, False)

    def new(self):
        return self.goal
