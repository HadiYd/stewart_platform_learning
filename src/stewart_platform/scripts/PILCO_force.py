#!/usr/bin/env python

# import tensorflow as tf
# import tensorflow.keras as keras
# from tensorflow.keras.layers import Input, Dense, Lambda, concatenate

import gym
# import argparse
import numpy as np
# import random
from collections import deque
# import time 

import reaching_pose_env_force
import rospy



from PILCO.models.pilco import PILCO
from PILCO.controllers import RbfController, LinearController
# from PILCO.rewards import ExponentialReward
import tensorflow as tf
# from gpflow import set_trainable
# from tensorflow import logging
np.random.seed(0)

from PILCO.utils import rollout, policy

tf.keras.backend.set_floatx('float64')
    
def main(args=None):

    rospy.init_node('stewart_gym_PILCO')
    env_name = 'StewartPose-v0'

    # Initial environment
    env = gym.make(env_name)

    # Initial random rollouts to generate a dataset
    X,Y, _, _ = rollout(env=env, pilco=None, verbose=False, random=True, timesteps=50, render=False)
    for i in range(1,5):
        X_, Y_, _, _ = rollout(env=env, pilco=None, verbose=False, random=True,  timesteps=50, render=False)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))


    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
    # controller = LinearController(state_dim=state_dim, control_dim=control_dim)

    pilco = PILCO((X, Y), controller=controller, horizon=20)
    # Example of user provided reward function, setting a custom target state
    # R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
    # pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

    for rollouts in range(3):
        pilco.optimize_models()
        pilco.optimize_policy()
        # import pdb; pdb.set_trace()
        X_new, Y_new, _, _ = rollout(env=env, pilco=pilco, timesteps=50, render=False, random=False)
        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))


if __name__ == "__main__":
    main()





