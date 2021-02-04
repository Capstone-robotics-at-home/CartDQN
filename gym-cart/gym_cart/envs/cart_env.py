"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

"""
NEXT STEPS 1/29
 - NEW SCENARIOS: ADD OBSTACLE, GOAL
"""

import math
import gym
from gym import spaces, logger
from gym.utils import seeding
from scipy.integrate import solve_ivp
import numpy as np
from .CartRender import Cart, Obstacle, Goal


class CartEnv(gym.Env):
    """
    Description:
        Simple 2D cart motion along frictionless track. The cart starts at (0, 0) and wants to stay in bounds.
    Observation:
        Type: Box(7)
        Num     Observation               Min                     Max
        0       Cart Position a             -10                     10
        1       Cart Velocity a            -1                      1
        2       Cart Velocity b            -10                      10
        3       Cart Velocity a            -1                      1
    Actions:
        Type: Continuous
        Num   Action
        0     Left motor (a) input force
        1     Right motor (b) input force
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

    def __init__(self):
        self.length = 0.15  # distance between point a and b NEED TO MEASURE IN METERS
        self.power = 1  # scaling factor for actual distance NEED TO DETERMINE
        self.tau = 0.01  # seconds between state updates INCREASE = FASTER MOVEMENTS
        self.thetaCurr = 0  # initial and current angle
        # self.goal_x = 1
        # self.goal_y = 1
        self.kinematics_integrator = 'euler'

        # Action Thresholds
        self.min_action = 0
        self.max_action = 100

        # State Thresholds
        self.max_position = 5
        self.max_speed = 300  # need to understand scale better
        self.max_angle = 180  # not sure if this is correct

        self.low_state = np.array(
            [-self.max_speed, -self.max_speed, -self.max_angle, -self.max_position,
             -self.max_position, -self.max_position, -self.max_position]
        )
        self.high_state = np.array(
            [-self.max_speed, -self.max_speed, -self.max_angle, -self.max_position,
             -self.max_position, -self.max_position, -self.max_position]
        )

        # Action: continuous left, right motor control
        self.action_space = spaces.Discrete(201)

        self.observation_space = spaces.Box(
            low=self.low_state,
            high=self.high_state,
            dtype=np.float32
        )

        self.seed()
        self.viewer = None
        self.state = np.zeros((7, 1)) # initial velocity always forward

        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        # err_msg = "%r (%s) invalid" % (action, type(action))
        # assert self.action_space.contains(action), err_msg

        va, vb, theta, xa, xb, ya, yb = self.state  # define 7 states

        # action on each motor
        # forcea = min(max(action[0], self.min_action), self.max_action)
        # forceb = min(max(action[1], self.min_action), self.max_action)

        forces = np.linspace(-1, 1, 201) # according to jetbot physical capabilities

        forcea = forces[action[0]]
        forceb = forces[action[1]]

        # scale accordingly
        va = forcea * self.power # cm/s
        vb = forceb * self.power

        # set up solver
        def sol_angle(t, y, vb, va):
            return (vb - va) / self.length

        t_span = [0, self.tau]
        sol = solve_ivp(sol_angle, t_span, [self.thetaCurr], args=(vb, va))

        # determine angles for movement
        costheta = math.cos(self.thetaCurr)
        sintheta = math.sin(self.thetaCurr)

        if self.kinematics_integrator == 'euler':
            xa = xa + va * costheta * self.tau
            ya = ya + va * sintheta * self.tau
            xb = xb + vb * costheta * self.tau
            yb = yb + vb * sintheta * self.tau
            theta = sol.y[0][-1]

        # For Scenario 2, better way is find ave position from a and b
        # if xa < -self.max_position: xa = -self.max_position
        # if xa > self.max_position: xa = self.max_position
        # if xb < -self.max_position: xa = -self.max_position
        # if xb > self.max_position: xa = self.max_position
        # if ya < -self.max_position: xa = -self.max_position
        # if ya > self.max_position: xa = self.max_position
        # if yb < -self.max_position: xa = -self.max_position
        # if yb > self.max_position: xa = self.max_position

        # update state
        self.thetaCurr = theta
        self.state = (va, vb, theta, xa, xb, ya, yb)

        # Scenario 1: Stay inside a given box
        done = bool(
            xa < -self.max_position
            or xa > self.max_position
            or xb < -self.max_position
            or xb > self.max_position
            or ya < -self.max_position
            or ya > self.max_position
            or yb < -self.max_position
            or yb > self.max_position
        )
        # Scenario 2: Reach a target
        # done = bool(
        #     xa >= self.goal_x and ya >= self.goal_y
        # )

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:  # not sure what any of these conditions are
            # Cart just exited!
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            if self.steps_beyond_done == 0:
                logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned done = True. You "
                    "should always call 'reset()' once you receive 'done = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_done += 1
            reward = 0.0

        return np.array(self.state), reward, done, {}

    def reset(self):
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(7,))  # need to look into
        self.steps_beyond_done = None
        return np.array(self.state)

    def render(self, mode='human'):
        screen_width = 500
        screen_height = 500

        world_width = self.max_position * 2
        scale = screen_width / world_width
        cartwidth = 20.0
        cartheight = 40.0

        if self.state is not None:
            va, vb, theta, xa, xb, ya, yb = self.state

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)

            # cart = rendering.FilledPolygon([(0, 0), (cartwidth, 0), (cartwidth/2, cartheight)])
            # cart.add_attr(rendering.Transform(translation=(0, 0)))

            cart = Cart()  # instantiate class instance
            cart = cart.new()  # call method for Cart
            self.carttrans = rendering.Transform()
            cart.add_attr(self.carttrans)
            self.viewer.add_geom(cart)

            # obst1 = Obstacle()  # instantiate class instance
            # obst1 = obst1.new()  # call method for Obstacle
            # self.obsttrans = rendering.Transform(translation=(screen_width/2, screen_height/2)) # set object position
            # obst1.add_attr(self.obsttrans)
            # self.viewer.add_geom(obst1)
            #
            # goal = Goal()  # instantiate class instance
            # goal = goal.new()  # call method for Goal
            # self.goaltrans = rendering.Transform(translation=(300, 300))
            # goal.set_color(.8, .8, 0)
            # goal.add_attr(self.goaltrans)
            # self.viewer.add_geom(goal)

        if self.state is None:
            return None

        cartx = scale*(xa + xb)/2 + screen_width/2.0  #
        carty = scale*(ya + yb)/2 + screen_height/2.0  #
        self.carttrans.set_translation(cartx, carty)
        self.carttrans.set_rotation(theta)

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
