#!/usr/bin/env python

import gym
import numpy as np
from collections import deque

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
from PILCO.utils import rollout, policy , rollout_trained

np.random.seed(0)

import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--run', type=int, default=20)
parser.add_argument('--load_checkpoint', type=bool, default=True)
args = parser.parse_args()


tf.keras.backend.set_floatx('float64')
    
def train(args=None):
    env_name = 'StewartPose-v0'

    # Initial environment
    env = gym.make(env_name)
    max_action = env.action_space.high[0] #
    horizon = 20

    max_episod = 200

    # Initial random rollouts to generate a dataset
    X,Y, _, _ = rollout(env=env, pilco=None,  random=True)
    for i in range(1,3):
        X_, Y_, _, _ = rollout(env=env, pilco=None, random=True)
        X = np.vstack((X, X_))
        Y = np.vstack((Y, Y_))


    state_dim = Y.shape[1]
    control_dim = X.shape[1] - state_dim
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5,max_action=max_action)
    # controller = LinearController(state_dim=state_dim, control_dim=control_dim)

    print("\n\n######################## RBF Controller SELECTED ###############")
    pilco = PILCO((X, Y), controller=controller, horizon=horizon)
    # Example of user provided reward function, setting a custom target state
    # R = ExponentialReward(state_dim=state_dim, t=np.array([0.1,0,0,0]))
    # pilco = PILCO(X, Y, controller=controller, horizon=40, reward=R)

    for rollouts in range(max_episod - 3):
        # import pdb; pdb.set_trace()
        # IT Is very computational expensive because of these to optimization step.
        try:
            pilco.optimize_models(save_model=True ,  maxiter=200)
        except:
            print("passed model optimization step.")
        try:
            pilco.optimize_policy(save_policy=True ,  maxiter=5)   
        except:
            print("passed policy optimization")     
       
        print(f"######## rollout num: {rollouts}  - Run with optimized model and policy")
        X_new, Y_new, _, _ = rollout(env=env, pilco=pilco, random=False)

        print("\n\n######## finished rollout ########")

        # Update dataset
        X = np.vstack((X, X_new)); Y = np.vstack((Y, Y_new))
        pilco.mgpr.set_data((X, Y))
        print("\n\n######## finished updating dataset ########")


def play_trained(args=None):
    env_name = 'StewartPose-v0' 
    max_episod_timestep = 200
    state_dim = 6 
    control_dim = 6

    # Initial environment
    env = gym.make(env_name)

    # Select controller 
    controller = RbfController(state_dim=state_dim, control_dim=control_dim, num_basis_functions=5)
    # controller = LinearController(state_dim=state_dim, control_dim=control_dim)

    X = []; Y = [];
    x = env.reset()
    x = np.array(x)
    u = np.zeros(6) # initial force 
    y_init = np.zeros(6)
    X.append(np.hstack((x, u)))
    Y.append(y_init) 
    X, Y = np.stack(X), np.stack(Y)

    # X,Y, _, _ = rollout(env=env, pilco=None,  random=True)
    pilco = PILCO((X, Y), controller=controller, horizon=20)
    try:
        pilco.load_model()
    except:
        print("couldn't load model ...")
    pilco.load_policy()

    print(" PLAYING")
    X_new, Y_new, _, _ = rollout_trained(env=env, pilco=pilco, random=False)
    print("DONE Playing")


if __name__ == "__main__":
    project_name = "FORCE_tunned_PID"
    rospy.init_node('stewart_gym_PILCO')

    if args.load_checkpoint:
        print("play trained.")
        wandb.init(name=f'PILCO_run_{args.run}', project=f"{project_name}_Run_Trained_new_Task")
        play_trained()
    else:
        print("training")
        wandb.init(name=f'PILCO_run_{args.run}', project=f"{project_name}_Train_and_Save")
        train()





