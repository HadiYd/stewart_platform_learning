#!/usr/bin/env python

import gym
import numpy as np
# import random
from collections import deque
# import time 

import reaching_pose_env_force
import rospy

from MBPO.tf_models.replay_memory import ReplayMemory
# from sac.sac import SAC
from MBPO.predict_env import PredictEnv
from MBPO.sample_env import EnvSampler
from MBPO.tf_models.constructor import construct_model  #, format_samples_for_training
from MBPO.MBPO_main import train , play_trained
from MBPO.tf_models.sac import SAC
from MBPO.args import readParser


# tf.keras.backend.set_floatx('float64')

def main(args=None):

    rospy.init_node('stewart_gym_MBPO')
    env_name = 'StewartPose-v0'
    if args is None:
        args = readParser()

    # Initial environment
    env = gym.make(env_name)

    # Intial agent
    agent = SAC(env.observation_space.shape[0], env.action_space, args)

    # Initial ensemble model
    state_size = np.prod(env.observation_space.shape)
    action_size = np.prod(env.action_space.shape)

    env_model = construct_model(obs_dim=state_size, act_dim=action_size, hidden_dim=args.pred_hidden_size, num_networks=args.num_networks,
                                    num_elites=args.num_elites)

    # Predict environments
    predict_env = PredictEnv(env_model, args.env_name, args.model_type) # specifying model name and model type : Tensorflow or pytoch

    # Initial pool for env
    env_pool = ReplayMemory(args.replay_size)
    # Initial pool for model
    rollouts_per_epoch = args.rollout_batch_size * args.epoch_length / args.model_train_freq
    model_steps_per_epoch = int(1 * rollouts_per_epoch)
    new_pool_size = args.model_retain_epochs * model_steps_per_epoch
    model_pool = ReplayMemory(new_pool_size)

    # Sampler of environment
    env_sampler = EnvSampler(env, max_path_length=args.max_path_length)

    if args.load_checkpoint:
        print("play trained")
        play_trained( args, env,agent,max_episodes=1)
    else:
        print("training")
        train(args, env_sampler, predict_env, agent, env_pool, model_pool)


if __name__ == "__main__":
    main()

