import numpy as np
from gpflow import config
from gym import make
float_type = config.default_float()
# import wandb


def rollout(env, pilco, timesteps, verbose=False, random=False, SUBS=1, render=False):
        X = []; Y = [];
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0
        for timestep in range(timesteps):
            if render: env.render()
            u = policy(env, pilco, x, random)
            # wandb.log({'Action_f1': list(u)[0] })
            # wandb.log({'Action_f2': list(u)[1] })
            # wandb.log({'Action_f3': list(u)[2] })
            # wandb.log({'Action_f4': list(u)[3] })
            # wandb.log({'Action_f5': list(u)[4] })
            # wandb.log({'Action_f6': list(u)[5] })
            for i in range(SUBS):
                x_new, r, done, _ = env.step(u)
                # wandb.log({'heave_z': list(x_new)[2] })
                # wandb.log({'yaw': list(x_new)[5] })
                ep_return_full += r
                # wandb.log({'Reward': ep_return_full})
                if done: break
                if render: env.render()
            if verbose:
                print("Action: ", u)
                print("State : ", x_new)
                print("Return so far: ", ep_return_full)
            X.append(np.hstack((x, u)))
            x_new = np.array(x_new)
            x = np.array(x)
            Y.append(x_new - x)
            ep_return_sampled += r
            x = x_new
            if done: break
        return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full


def policy(env, pilco, x, random):
    if random:
        return env.action_space.sample()
    else:
        return pilco.compute_action(x).numpy().reshape(6,) #pilco.compute_action(x[None, :])[0, :]

class Normalised_Env():
    def __init__(self, env_id, m, std):
        self.env = make(env_id).env
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.m = m
        self.std = std

    def state_trans(self, x):
        return np.divide(x-self.m, self.std)

    def step(self, action):
        ob, r, done, _ = self.env.step(action)
        return self.state_trans(ob), r, done, {}

    def reset(self):
        ob =  self.env.reset()
        return self.state_trans(ob)

    def render(self):
        self.env.render()