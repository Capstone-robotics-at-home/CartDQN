import gym
from gym.envs.classic_control import rendering


class Cart:
    cartwidth = 20.0
    cartheight = 40.0

    def __init__(self):
        l, r, t, b, c = -cartwidth / 2, cartwidth / 2, cartheight / 2, -cartheight / 2, 0
        return rendering.FilledPolygon([(c, t), (l, b), (r, b)])
    # def cart(self, x, y):

class Obstacle:
    obstwidth = 30.0
    obstheight = 30.0

    def __init__(self, scale):
        l, r, t, b = -obswidth * scale / 2, obswidth * scale / 2, obsheight * scale / 2, -obsheight * scale/ 2, 0
        return rendering.FilledPolygon([(l, t), (l, b), (r, b), (r, t)])
    # def obstacle(self, x, y):
