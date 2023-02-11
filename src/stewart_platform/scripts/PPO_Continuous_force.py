#!/usr/bin/env python
"""
main repo: https://github.com/marload/DeepRL-TensorFlow2
"""
import wandb
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda
import time
import csv

import gym
import argparse
import numpy as np

import reaching_pose_env_force
import rospy
# import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--lmbda', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--run', type=int, default=1)
parser.add_argument('--load_checkpoint', type=bool, default=False)


args = parser.parse_args()

tf.keras.backend.set_floatx('float64')


class Actor:
    def __init__(self, state_dim, action_dim,action_bound_high , action_bound_low, std_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_high = action_bound_high
        self.action_bound_low = action_bound_low
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        mu, std = self.model.predict(state)
        action = np.random.normal(mu[0], std[0], size=self.action_dim)
        action = np.clip(action,  self.action_bound_low , self.action_bound_high)
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
            var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def create_model(self):
        state_input = Input((self.state_dim,))
        dense_1 = Dense(400, activation='relu')(state_input)
        dense_2 = Dense(300, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='sigmoid')(dense_2)   # changed to sigmoid instead of tanh
        mu_output = Lambda(lambda x: x * (self.action_bound_high-self.action_bound_low) + self.action_bound_low)(out_mu) # Denormalize output layer to adapt PID values
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)   # adding an entropy bonus to ensure sufficient exploration
        return tf.keras.models.Model(state_input, [mu_output, std_output])

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0-args.clip_ratio, 1.0+args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, states, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(states, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save_weights(self,path):
        self.model.save_weights(path)

    def load_weights(self,path):
        self.model.load_weights(path)

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
            Dense(100, activation='relu'),
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

    def save_weights(self,path):
        self.model.save_weights(path)

    def load_weights(self,path):
        self.model.load_weights(path)


class Agent:
    def __init__(self, env,chkpt_dir='models/ppo/',chkpt_dir_w = 'models_weights/ppo/'):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound_high = self.env.action_space.high #self.env.action_space.high[0] this was useful to scale each!
        self.action_bound_low  = self.env.action_space.low  # in order to specify PID low limits
        self.std_bound = [1e-2, 1.0]
        self.chkpt_dir = chkpt_dir
        self.chkpt_dir_w = chkpt_dir_w

        self.actor_opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(args.critic_lr)
        self.actor = Actor(self.state_dim, self.action_dim,
                           self.action_bound_high , self.action_bound_low, self.std_bound)
        self.critic = Critic(self.state_dim)

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards)
        gae = np.zeros_like(rewards)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + args.gamma * forward_val - v_values[k]
            gae_cumulative = args.gamma * args.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, max_episodes=1000):
        best_score = self.env.reward_range[0]
        reward_history = []
        for ep in range(max_episodes):
            episode_reward, done = 0, False

            state_batch = []
            action_batch = []
            reward_batch = []
            old_policy_batch = []
    
            state = self.env.reset()
            while not done:
                log_old_policy, action = self.actor.get_action(state)

                wandb.log({'Action_f1': list(action)[0] })
                wandb.log({'Action_f2': list(action)[1] })
                wandb.log({'Action_f3': list(action)[2] })
                wandb.log({'Action_f4': list(action)[3] })
                wandb.log({'Action_f5': list(action)[4] })
                wandb.log({'Action_f6': list(action)[5] })
                wandb.log({'heave_z': list(state)[2] })
                wandb.log({'yaw': list(state)[5] })

                next_state, reward, done, _ = self.env.step(action)

                state = np.reshape(state, [1, self.state_dim])
                action = np.reshape(action, [1, self.action_dim])
                next_state = np.reshape(next_state, [1, self.state_dim])
                reward = np.reshape(reward, [1, 1])
                log_old_policy = np.reshape(log_old_policy, [1, 1])

                state_batch.append(state)
                action_batch.append(action)
                reward_batch.append((reward+8)/8)
                old_policy_batch.append(log_old_policy)

                if len(state_batch) >= args.update_interval or done:
                    states = self.list_to_batch(state_batch)
                    actions = self.list_to_batch(action_batch)
                    rewards = self.list_to_batch(reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.critic.model.predict(states)
                    next_v_value = self.critic.model.predict(next_state)

                    gaes, td_targets = self.gae_target(
                        rewards, v_values, next_v_value, done)

                    for epoch in range(args.epochs):
                        actor_loss = self.actor.train(
                            old_policys, states, actions, gaes)
                        critic_loss = self.critic.train(states, td_targets)

                    state_batch = []
                    action_batch = []
                    reward_batch = []
                    old_policy_batch = []

                episode_reward += reward[0][0]
                state = next_state[0]

            print('EP{} EpisodeReward={}'.format(ep, episode_reward))
            wandb.log({'Reward': episode_reward})

            # Save model 
            reward_history.append(episode_reward)
            avg_score = np.mean(reward_history[-100:])
            if avg_score > best_score:
                best_score = avg_score
                self.save_models_weights()

    def over_shoot(self, yout):
        return (yout.max()/yout[-1]-1)*100

    def rise_time(self,t,yout):
        return t[next(i for i in range(0,len(yout)-1) if yout[i]>yout[-1]*.90)]-t[0]

    def settling_time(self,t,yout):
        return t[next(len(yout)-i for i in range(2,len(yout)-1) if abs(yout[-i]/yout[-1]-1)>0.02 )]-t[0]

    def play_trained(self, max_episodes=10):
        wandb.define_metric("Simulation Time (second)")
        sim_step = 0.05
        log_dict = {}
        step_loop = 1
        heave_spec = {}
        yaw_spec = {}
        start_time = time.time()

        for ep in range(max_episodes):
            episode_reward, done = 0, False
            state = self.env.reset()  
            t_end = time.time() + 20         
            while not done or (time.time() < t_end):
                _, action = self.actor.get_action(state)
                log_dict = {
                            'Action_f1': list(action)[0], 'Action_f2': list(action)[1]  , 'Action_f3': list(action)[2],
                            'Action_f3': list(action)[3], 'Action_f5': list(action)[5]  , 'Action_f6': list(action)[5],
                            'Surge(x)': list(state)[0] ,'Sway(y)': list(state)[1],'Heave(z)': list(state)[2],
                            'Roll': list(state)[3], 'Pitch': list(state)[4] ,'Yaw': list(state)[5],
                            "Simulation Time (Seconds)": step_loop*sim_step,
                            }    
                wandb.log(log_dict)

                next_state, reward, done, _ = self.env.step(action)             
                episode_reward += reward
                state = next_state

                heave_spec[step_loop*sim_step] = list(state)[2]
                yaw_spec[step_loop*sim_step]   = list(state)[5]
                step_loop +=1
            print('Trained EP{} EpisodeReward={}'.format(ep, episode_reward))
            wandb.log({'Reward': episode_reward})

            # initialize list of lists
            time_spec = [['Heave','Over_Shoot', self.over_shoot(np.array(list(heave_spec.values())))],
                    ['Heave','Rise_Time',  self.rise_time(np.array(list(heave_spec.keys())), np.array(list(heave_spec.values())))],
                    ['Heave','Settling_Time', self.settling_time(np.array(list(heave_spec.keys())), np.array(list(heave_spec.values())))],
                    ['Yaw','Over_Shoot', self.over_shoot(np.array(list(yaw_spec.values())))],
                    ['Yaw','Rise_Time',  self.rise_time(np.array(list(yaw_spec.keys())), np.array(list(yaw_spec.values())))],
                    ['Yaw','Settling_Time', self.settling_time(np.array(list(yaw_spec.keys())), np.array(list(yaw_spec.values())))]]
            
            # Save the required time domain specifications in a pandas Data frame
            spec_table = wandb.Table( data=time_spec, columns=['State', 'Specification', 'Value'])
            wandb.log({f"Specification_PPO_run_{args.run}": spec_table})
            # spec_df = pd.DataFrame(time_spec, columns=['State', 'Specification', 'Value'])
            # wandb.log({f"'Specification_PPO_run_{args.run}'": spec_df})
        print("Total Time to exacute: ", time.time() - start_time)
        time.sleep(3)

    def save_models_weights(self):
        print('... saving models weights ...')
        self.actor.save_weights(self.chkpt_dir_w+'actor_weights.h5')
        self.critic.save_weights(self.chkpt_dir_w+'critic_weights.h5')
    
    def load_models_weights(self):
        print('... loading models weights ...')
        self.actor.load_weights(self.chkpt_dir_w+'actor_weights.h5')
        self.critic.load_weights(self.chkpt_dir_w+'critic_weights.h5')



def main():
    rospy.init_node('stewart_gym_PPO')
    env_name = 'StewartPose-v0'
    env = gym.make(env_name)
    agent = Agent(env)


    # Train or play the trained one!
    project_name = "Feedforward_Control_Force_0_10"
    if args.load_checkpoint:
        wandb.init(name=f'PPO_run_{args.run}', project=f"Run_Trained_{project_name}")
        agent.load_models_weights()
        agent.play_trained(max_episodes=1)
    else:
        print("training")
        wandb.init(name=f'PPO_run_{args.run}', project=f"Train_and_Save_{project_name}")
        agent.train(max_episodes=1000)


if __name__ == "__main__":
    main()



