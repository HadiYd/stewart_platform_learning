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
import wandb
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


import warnings 
warnings.filterwarnings('ignore')




from PILCO.models.pilco import PILCO
from PILCO.controllers import RbfController, LinearController
# from PILCO.rewards import ExponentialReward

# from gpflow import set_trainable
# from tensorflow import logging
np.random.seed(0)

from PILCO.utils import rollout, policy

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, default=6)
parser.add_argument('--load_checkpoint', type=bool, default=False)


args = parser.parse_args()


tf.keras.backend.set_floatx('float64')
    
def train(args=None):

    rospy.init_node('stewart_gym_PILCO')
    env_name = 'StewartPose-v0'

    # Initial environment
    env = gym.make(env_name)


    # TODO we defined max episod step inside the taskEnv which is 200! 
    # TODO change timestep based to episodic based to match the 3 DRLs setup
    max_episod_timestep = 200
    max_episod = 200

    # Initial random rollouts to generate a dataset
    X,Y, _, _ = rollout(env=env, pilco=None,  random=True, timesteps= max_episod_timestep)
    for i in range(1,3):
        X_, Y_, _, _ = rollout(env=env, pilco=None, random=True,  timesteps= max_episod_timestep)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))


    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
    # controller = LinearController(state_dim=state_dim, control_dim=control_dim)

    print("\n\n######################## RBF Controller SELECTED ###############")
    pilco = PILCO((X, Y), controller=controller, horizon=20)
    # Example of user provided reward function, setting a custom target state
    # R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
    # pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

    for rollouts in range(max_episod - 3):
        # import pdb; pdb.set_trace()
        # IT Is very computational expensive because of these to optimization step.
        pilco.optimize_models(save_model=True)
        pilco.optimize_policy(save_policy=True)        
       
        print(f"######## rollout num: {rollouts}  - Run with optimized model and policy")
        pilco.load_policy()
        X_new, Y_new, _, _ = rollout(env=env, pilco=pilco, random=False, timesteps= max_episod_timestep )
        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))


def play_trained(args=None):

    rospy.init_node('stewart_gym_PILCO')
    env_name = 'StewartPose-v0'
    max_episod_timestep = 200
    state_dim = 6 
    control_dim = 6

    # Initial environment
    env = gym.make(env_name)

    # Select controller 
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
    X = []; Y = [];
    pilco = PILCO((X, Y), controller=controller, horizon=20)
    pilco.load_model()
    pilco.load_policy()

    X_new, Y_new, _, _ = rollout(env=env, pilco=pilco, random=False, timesteps=max_episod_timestep )
    print("DONE Playing")


if __name__ == "__main__":
    run = 30
    project_name = "FORCE"

    if args.load_checkpoint:
        # wandb.init(name=f'PILCO_run_{args.run}',project=f"{project_name}_Run_Trained")
        play_trained()
    else:
        print("training")
        # wandb.init(name=f'PILCO_run_{args.run}', project=f"{project_name}_Train_and_Save")
        train()





