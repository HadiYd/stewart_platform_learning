#!/usr/bin/env python
"""
main repo: https://github.com/marload/DeepRL-TensorFlow2
"""
import wandb
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate

import gym
import argparse
import numpy as np
import random
from collections import deque

import reaching_pose_env
import rospy


parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--actor_lr', type=float, default=0.0001)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--tau', type=float, default=0.001)
parser.add_argument('--train_start', type=int, default=2000)
parser.add_argument('--run', type=int, default=1)


args = parser.parse_args()


tf.keras.backend.set_floatx('float64')
wandb.init(name=f'DDPG_run_{args.run}', project="Train_and_Run_trained")

class ReplayBuffer:
    def __init__(self, capacity=20000):   
        self.buffer = deque(maxlen=capacity)
    
    def put(self, state, action, reward, next_state, done):
        self.buffer.append([state, action, reward, next_state, done])
    
    def sample(self):
        sample = random.sample(self.buffer, args.batch_size)
        states, actions, rewards, next_states, done = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(args.batch_size, -1)
        next_states = np.array(next_states).reshape(args.batch_size, -1)
        return states, actions, rewards, next_states, done
    
    def size(self):
        return len(self.buffer)

class Actor:
    def __init__(self, state_dim, action_dim, action_bound_high , action_bound_low):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound_high = action_bound_high
        self.action_bound_low = action_bound_low
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.state_dim,)),
            Dense(400, activation='relu'),    
            Dense(300, activation='relu'),
            Dense(self.action_dim, activation='sigmoid'),   # changed to sigmoid instead of tanh
            Lambda(lambda x: x * (self.action_bound_high-self.action_bound_low) + self.action_bound_low)  # Denormalize output layer to adapt PID values
        ])

    def train(self, states, q_grads):
        with tf.GradientTape() as tape:
            grads = tape.gradient(self.model(states), self.model.trainable_variables, -q_grads)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
    
    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        return self.model.predict(state)[0]
    def save(self,path):
        self.model.save(path)



class Critic:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        state_input = Input((self.state_dim,))
        s1 = Dense(400, activation='relu')(state_input)
        s2 = Dense(300, activation='relu')(s1)
        action_input = Input((self.action_dim,))
        a1 = Dense(32, activation='relu')(action_input)
        c1 = concatenate([s2, a1], axis=-1)
        c2 = Dense(16, activation='relu')(c1)
        output = Dense(1, activation='linear')(c2)
        return tf.keras.Model([state_input, action_input], output)
    
    def predict(self, inputs):
        return self.model.predict(inputs)
    
    def q_grads(self, states, actions):
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model([states, actions])
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, actions)

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model([states, actions], training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss
    def save(self,path):
        self.model.save(path)


class Agent:
    def __init__(self, env,chkpt_dir='models/ddpg/'):
        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.action_bound_high = self.env.action_space.high #self.env.action_space.high[0] this was useful to scale each!
        self.action_bound_low  = self.env.action_space.low  # in order to specify PID low limits
        self.chkpt_dir = chkpt_dir

        self.buffer = ReplayBuffer()

        self.actor =  Actor( self.state_dim, self.action_dim, self.action_bound_high,self.action_bound_low )
        self.critic = Critic(self.state_dim, self.action_dim)
        
        self.target_actor  = Actor( self.state_dim, self.action_dim, self.action_bound_high,self.action_bound_low )
        self.target_critic = Critic(self.state_dim, self.action_dim)

        actor_weights  = self.actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        self.target_actor.model.set_weights(actor_weights)
        self.target_critic.model.set_weights(critic_weights)
        
    
    def target_update(self):
        actor_weights = self.actor.model.get_weights()
        t_actor_weights = self.target_actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        t_critic_weights = self.target_critic.model.get_weights()

        for i in range(len(actor_weights)):
            t_actor_weights[i] = args.tau * actor_weights[i] + (1 - args.tau) * t_actor_weights[i]

        for i in range(len(critic_weights)):
            t_critic_weights[i] = args.tau * critic_weights[i] + (1 - args.tau) * t_critic_weights[i]
        
        self.target_actor.model.set_weights(t_actor_weights)
        self.target_critic.model.set_weights(t_critic_weights)


    def td_target(self, rewards, q_values, dones):
        targets = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            if dones[i]:
                targets[i] = rewards[i]
            else:
                targets[i] = args.gamma * q_values[i]
        return targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch
    
    def ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return x + rho * (mu-x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
    
    def replay(self):
        for _ in range(10):
            states, actions, rewards, next_states, dones = self.buffer.sample()
            target_q_values = self.target_critic.predict([next_states, self.target_actor.predict(next_states)])
            td_targets = self.td_target(rewards, target_q_values, dones)
            
            self.critic.train(states, actions, td_targets)
            
            s_actions = self.actor.predict(states)
            s_grads = self.critic.q_grads(states, s_actions)
            grads = np.array(s_grads).reshape((-1, self.action_dim))
            self.actor.train(states, grads)
            self.target_update()

    def train(self, max_episodes=1000):
        reward_history = []
        for ep in range(max_episodes):
            episode_reward, done = 0, False

            state = self.env.reset()
            best_score = self.env.reward_range[0]
            bg_noise = np.zeros(self.action_dim)
            
            while not done:
                action = self.actor.get_action(state)
                noise = self.ou_noise(bg_noise, dim=self.action_dim)
                action = np.clip(action + noise, self.action_bound_low , self.action_bound_high)

                wandb.log({'Action_P': list(action)[0] })
                wandb.log({'Action_I': list(action)[1] })
                wandb.log({'Action_D': list(action)[2] })
                wandb.log({'heave_z': list(state)[2] })
                wandb.log({'yaw': list(state)[5] })

                next_state, reward, done, _ = self.env.step(action)
                self.buffer.put(state, action, (reward+8)/8, next_state, done)
                bg_noise = noise
                episode_reward += reward
                state = next_state
            if self.buffer.size() >= args.batch_size and self.buffer.size() >= args.train_start:
                self.replay()                
            print('EP{} EpisodeReward={}'.format(ep, episode_reward))
            wandb.log({'Reward': episode_reward})

            # Save model 
            reward_history.append(episode_reward)
            avg_score = np.mean(reward_history[-100:])
            if avg_score > best_score:
                best_score = avg_score
                self.save_models()

    def play_trained(self, max_episodes=10):
        for ep in range(max_episodes):
            episode_reward, done = 0, False
            state = self.env.reset()            
            while not done:
                action = self.actor.get_action(state)
                action = np.clip(action , self.action_bound_low , self.action_bound_high)
                wandb.log({'Action_P_Trained': list(action)[0] })
                wandb.log({'Action_I_Trained': list(action)[1] })
                wandb.log({'Action_D_Trained': list(action)[2] })
                wandb.log({'heave_z_Trained': list(state)[2] })
                wandb.log({'yaw_Trained': list(state)[5] })
                next_state, reward, done, _ = self.env.step(action)             
                episode_reward += reward
                state = next_state
            print('Trained EP{} EpisodeReward={}'.format(ep, episode_reward))
            wandb.log({'Reward_trained': episode_reward})

    def save_models(self):
        print('... saving models ...')
        self.actor.save(self.chkpt_dir+'actor')
        self.target_actor.save(self.chkpt_dir+'target_actor')
        self.critic.save(self.chkpt_dir+'critic')
        self.target_critic.save(self.chkpt_dir+'target_critic')

    def load_models(self):
        print('... loading models ...')
        self.actor = keras.models.load_model(self.chkpt_dir+'actor')
        self.target_actor = \
            keras.models.load_model(self.chkpt_dir+'target_actor')
        self.critic = keras.models.load_model(self.chkpt_dir+'critic')
        self.target_critic = \
            keras.models.load_model(self.chkpt_dir+'target_critic')


def main():
    rospy.init_node('stewart_gym_DDPG')
    env_name = 'StewartPose-v0'
    env = gym.make(env_name)
    agent = Agent(env)

    # Train or play the trained one!
    load_checkpoint = False 
    if load_checkpoint:
        agent.load_models()
        agent.play_trained(max_episodes=10)
    else:
        print("training")
        agent.train(max_episodes=500)
    

if __name__ == "__main__":
    main()
