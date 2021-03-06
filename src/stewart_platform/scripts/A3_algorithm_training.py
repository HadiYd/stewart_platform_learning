#!/usr/bin/env python
"""
main repo: https://github.com/marload/DeepRL-TensorFlow2
"""
import wandb
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import gym
import argparse
import numpy as np

# to run script fast: 
from threading import Thread
from multiprocessing import cpu_count

import reaching_pose_env
import rospy

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--run', type=int, default=1)


args = parser.parse_args()

tf.keras.backend.set_floatx('float64')
wandb.init(name=f'A3C_algorithm_{args.run}', project="Distance_plot_1000")

CUR_EPISODE = 0

class Actor:
    def __init__(self, state_dim, action_dim, action_bound_high,action_bound_low, std_bound):
        print("START Actor init")
        self.state_dim = state_dim
        self.action_dim = action_dim
        # self.action_bound = action_bound
        self.action_bound_high = action_bound_high
        self.action_bound_low = action_bound_low
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.entropy_beta = 0.01
        print("END Actor init....")

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(400, activation='relu')(state_input)
        dense_2 = Dense(300, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='sigmoid')(dense_2)  # changed to sigmoid instead of tanh
        mu_output = Lambda(lambda x: x * (self.action_bound_high-self.action_bound_low) + self.action_bound_low)(out_mu)  # Denormalize output layer to adapt PID values
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        mu, std = mu[0], std[0]
        return np.random.normal(mu, std, size=self.action_dim)

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def compute_loss(self, mu, std, actions, advantages):
        log_policy_pdf = self.log_pdf(mu, std, actions)
        loss_policy = log_policy_pdf * advantages
        return tf.reduce_sum(-loss_policy)

    def train(self, states, actions, advantages):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            loss = self.compute_loss(mu, std, actions, advantages)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, state_dim):
        self.state_dim = state_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(400, activation='relu'),
            Dense(300, activation='relu'),
            Dense(200, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(states, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env_name):
        print("Start Agent init....")
        env = gym.make(env_name)
        self.env_name = env_name
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]
        # self.action_bound = env.action_space.high
        self.action_bound_high = env.action_space.high 
        self.action_bound_low  = env.action_space.low  
        self.std_bound =  [1e-2, 10.0]   

        self.global_actor = Actor(self.state_dim, self.action_dim, self.action_bound_high,self.action_bound_low  ,self.std_bound)
        self.global_critic = Critic(self.state_dim)

        
        self.num_workers = 1
        print("NUMBER OF WORKERS=CPUs available for this training="+str(self.num_workers))
        # self.num_workers = cpu_count()
        print("END Agent init....")

    def train(self, max_episodes=1000):  
        workers = []

        for i in range(self.num_workers):
            env = gym.make(self.env_name)
            workers.append(WorkerAgent(
                env, self.global_actor, self.global_critic, max_episodes))

        for worker in workers:
            worker.start()

        for worker in workers:
            worker.join()


class WorkerAgent(Thread):
    def __init__(self, env, global_actor, global_critic, max_episodes):
        print("Start Worker Agent init....")
        Thread.__init__(self)
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound = self.env.action_space.high # self.env.action_space.high[0]
        self.action_bound_high = self.env.action_space.high #self.env.action_space.high[0] this was useful to scale each!
        self.action_bound_low  = self.env.action_space.low  # in order to specify PID low limits

        self.std_bound = [1e-2, 10.0]

        self.max_episodes = max_episodes
        self.global_actor = global_actor
        self.global_critic = global_critic
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound_high, self.action_bound_low, self.std_bound)
        self.critic = Critic(self.state_dim)

        self.actor.model.set_weights(self.global_actor.model.get_weights())
        self.critic.model.set_weights(self.global_critic.model.get_weights())
        print("END Worker Agent init....")

    def n_step_td_target(self, rewards, next_v_value, done):
        td_targets = np.zeros_like(rewards)
        cumulative = 0
        if not done:
            cumulative = next_v_value

        for k in reversed(range(0, len(rewards))):
            cumulative = args.gamma * cumulative + rewards[k]
            td_targets[k] = cumulative
        return td_targets

    def advatnage(self, td_targets, baselines):
        return td_targets - baselines

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self):
        print("####################### START Train")
        global CUR_EPISODE

        while self.max_episodes >= CUR_EPISODE:
            state_batch = []
            action_batch = []
            reward_batch = []
            episode_reward, done = 0, False

            print("####################### START AC3")


            state = self.env.reset()

            while not done:

                action = self.actor.get_action(state)

                action = np.clip(action,  self.action_bound_low , self.action_bound_high)

                wandb.log({'Action_P': list(action)[0] })
                wandb.log({'Action_I': list(action)[1] })
                wandb.log({'Action_D': list(action)[2] })
                wandb.log({'heave_z': list(state)[2] })
                wandb.log({'yaw': list(state)[5] })


                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append(reward)

                if len(state_batch) >= args.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)

                    next_v_value = self.critic.model.predict(next_state)
                    td_targets = self.n_step_td_target(
                        (rewards+8)/8, next_v_value, done)
                    advantages = td_targets - self.critic.model.predict(states)

                    actor_loss = self.global_actor.train(
                        states, actions, advantages)
                    critic_loss = self.global_critic.train(
                        states, td_targets)

                    self.actor.model.set_weights(
                        self.global_actor.model.get_weights())
                    self.critic.model.set_weights(
                        self.global_critic.model.get_weights())

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    td_target_batch = []
                    advatnage_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(CUR_EPISODE, episode_reward))
            wandb.log({'Reward': episode_reward})
            
            CUR_EPISODE += 1

    def run(self):
        self.train()


def main():

    rospy.init_node('DRL Agent')
    env_name = 'StewartPose-v0'
    agent = Agent(env_name)
    agent.train(max_episodes=500)


if __name__ == "__main__":
    main()