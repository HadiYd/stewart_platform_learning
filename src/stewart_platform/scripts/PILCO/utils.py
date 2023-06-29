import numpy as np
from gpflow import config
from gym import make
float_type = config.default_float()
import wandb
import time


def rollout(env, pilco, random=False):
        X = []; Y = [];
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0
        done = False
        while not done:
            u = policy(env, pilco, x, random)
            print('\nselect action according pilco policy')
            x_new, r, done, _ = env.step(u)
            ep_return_full += r
            print('\nnext state')
            wandb.log({'Action_f1': list(u)[0] })
            wandb.log({'Action_f2': list(u)[1] })
            wandb.log({'Action_f3': list(u)[2] })
            wandb.log({'Action_f4': list(u)[3] })
            wandb.log({'Action_f5': list(u)[4] })
            wandb.log({'Action_f6': list(u)[5] })
            wandb.log({'heave_z': list(x_new)[2] })
            wandb.log({'yaw': list(x_new)[5] })
            wandb.log({'Reward': ep_return_full})
            print("\nlogged wandb")

            X.append(np.hstack((x, u)))
            x_new = np.array(x_new)
            x = np.array(x)
            Y.append(x_new - x)
            ep_return_sampled += r
            x = x_new
        print("\n###  RECIEVED EPISOD DONE from ENV ####\n")
    
        return np.stack(X), np.stack(Y), ep_return_sampled, ep_return_full


def over_shoot( yout):
    return (yout.max()/yout[-1]-1)*100

def rise_time(t,yout):
    return t[next(i for i in range(0,len(yout)-1) if yout[i]>yout[-1]*.90)]-t[0]

def settling_time(t,yout):
    return t[next(len(yout)-i for i in range(2,len(yout)-1) if abs(yout[-i]/yout[-1]-1)>0.02 )]-t[0]

def rollout_trained(env, pilco, random=False):
        X = []; Y = [];
        sim_step = 0.1 # 0.05
        step_loop = 1
        log_dict = {}
        heave_spec = {}
        yaw_spec = {}
        x = env.reset()
        ep_return_full = 0
        ep_return_sampled = 0
        done = False
        t_end = time.time() + 40    
        while not done or (time.time() < t_end):
            u = policy(env, pilco, x, random)
            log_dict = {
                        'Action_f1':  list(u)[0], 'Action_f2': list(u)[0] , 'Action_f3': list(u)[2],
                        'Action_f3': list(u)[3], 'Action_f5': list(u)[5]  , 'Action_f6': list(u)[5],
                        'Surge(x)': list(x)[0] ,'Sway(y)': list(x)[1],'Heave(z)': list(x)[2],
                        'Roll': list(x)[3], 'Pitch': list(x)[4] ,'Yaw': list(x)[5],
                        "Simulation Time (Seconds)": step_loop*sim_step,
                        }    
            
            x_new, r, done, _ = env.step(u)
            wandb.log(log_dict)
            print('\nselect action according pilco policy')
            print('\nnext state')
            print("\nlogged wandb")
            ep_return_full += r

            X.append(np.hstack((x, u)))
            x_new = np.array(x_new)
            x = np.array(x)
            Y.append(x_new - x)
            ep_return_sampled += r
            x = x_new
            heave_spec[step_loop*sim_step] = list(x)[2]
            yaw_spec[step_loop*sim_step]   = list(x)[5]
            step_loop +=1
        print("\n###  RECIEVED EPISOD DONE from ENV ####\n")
        # initialize list of lists
        time_spec = [['Heave','Over_Shoot',   over_shoot(np.array(list(heave_spec.values())))],
                ['Heave','Rise_Time',    rise_time(np.array(list(heave_spec.keys())), np.array(list(heave_spec.values())))],
                ['Heave','Settling_Time',   settling_time(np.array(list(heave_spec.keys())), np.array(list(heave_spec.values())))],
                ['Yaw','Over_Shoot',   over_shoot(np.array(list(yaw_spec.values())))],
                ['Yaw','Rise_Time',    rise_time(np.array(list(yaw_spec.keys())), np.array(list(yaw_spec.values())))],
                ['Yaw','Settling_Time',   settling_time(np.array(list(yaw_spec.keys())), np.array(list(yaw_spec.values())))]]
        
        # Save the required time domain specifications in a pandas Data frame
        spec_table = wandb.Table( data=time_spec, columns=['State', 'Specification', 'Value'])
        wandb.log({f"Specification_PPO_run_1": spec_table})
    
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