import numpy as np
import matplotlib

from numpy import linalg
from matplotlib import animation

"""
A_star information
    pixel world
Mujoco information
    pixel world
Sim information
    real world: need scaling factor based on camera setup to determine jetbot actions
Jetbot information
    real world
"""

path = [(219, 453), (218, 454), (217, 455), (216, 456), (215, 457),
        (214, 458), (213, 459), (212, 460), (211, 461), (210, 462)]
dict = {'Jetbot': [(210, 462), 107, 314, 577, 347], 'Obstacle': [(758, 292), 693, 823, 388, 197],
        'Target': [(1070, 199), 1036, 1105, 256, 143], 'Grabber': [(174, 591), 141, 207, 660, 523]}


class Translate():
    def __init__(self, path, dictionary, des_theta):
        self.x = []
        self.y = []

        for i in reversed(range(len(path))):
            self.x.append(path[i][0])
            self.y.append(path[i][1])

        self.grabber = dictionary['Grabber'][0]
        self.jetbot = dictionary['Jetbot'][0]

        self.horizontal = [1, 0]
        cart_direction = [self.grabber[0] - self.jetbot[0], self.grabber[1] - self.jetbot[1]]
        self.unit_vector_1 = self.horizontal / np.linalg.norm(self.horizontal)
        unit_vector_2 = cart_direction / np.linalg.norm(cart_direction)
        dot_product = np.dot(self.unit_vector_1, unit_vector_2)
        self.theta = np.arccos(dot_product)

        self.des_theta = des_theta

    def angle(self):
        next_point = [self.x[1] - self.jetbot[0], self.y[1] - self.jetbot[1]]
        unit_vector_3 = next_point / np.linalg.norm(next_point)
        # print(unit_vector_3)
        dot_product = np.dot(self.unit_vector_1, unit_vector_3)
        self.des_theta = np.arccos(dot_product)
        return self.des_theta

    def step(self):
        # print(self.des_theta - self.theta)
        if abs(self.des_theta - self.theta) <= 0.1:
            action = 0  # move forward
            self.jetbot = (self.x[1], self.y[1])  # assumes perfect model
            self.x = self.x[1:]
            self.y = self.y[1:]
        elif (self.des_theta - self.theta) < np.pi:
            action = 1  # turn left
        elif (self.des_theta - self.theta) < -np.pi:
            action = 2  # turn right
        self.theta = self.des_theta

        return action

    def render(self):
        pass


#  main loop
#  Astar to get new path, dict
#  Initialize cart
cart = Translate(path, dict, 0)
master_x = [cart.jetbot[0]]
master_y = [cart.jetbot[1]]

while True:
    if cart.jetbot == path[0]:
        break
    theta = cart.angle()
    actionz = cart.step()
    master_x.append(cart.jetbot[0])
    master_y.append(cart.jetbot[1])
    print(actionz)
    # print(cart.jetbot)
    #  Send action to cart

#  Plot/animation
