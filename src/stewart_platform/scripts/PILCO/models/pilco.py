import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

import gpflow
import pandas as pd
import time

from .mgpr import MGPR
from .smgpr import SMGPR
from .. import controllers
from .. import rewards
import os
import pickle

float_type = gpflow.config.default_float()
from gpflow import set_trainable

class PILCO(gpflow.models.BayesianModel):
    def __init__(self, data, num_induced_points=None, horizon=30, controller=None,
                reward=None, m_init=None, S_init=None, name=None):
        super(PILCO, self).__init__(name)
        if num_induced_points is None:
            self.mgpr = MGPR(data)
        else:
            self.mgpr = SMGPR(data, num_induced_points)
        self.state_dim = data[1].shape[1]
        self.control_dim = data[0].shape[1] - data[1].shape[1]
        self.horizon = horizon

        if controller is None:
            self.controller = controllers.LinearController(self.state_dim, self.control_dim)
        else:
            self.controller = controller

        if reward is None:
            self.reward = rewards.ExponentialReward(self.state_dim)
        else:
            self.reward = reward

        if m_init is None or S_init is None:
            # If the user has not provided an initial state for the rollouts,
            # then define it as the first state in the dataset.
            self.m_init = data[0][0:1, 0:self.state_dim]
            self.S_init = np.diag(np.ones(self.state_dim) * 0.1)
        else:
            self.m_init = m_init
            self.S_init = S_init
        self.optimizer = None

    def training_loss(self):
        # This is for tuning controller's parameters
        reward = self.predict(self.m_init, self.S_init, self.horizon)[2]
        return -reward

    def optimize_models(self,save_model=False,load_model = False,  maxiter=200, restarts=1):
        '''
        Optimize GP models
        '''
        print("\n\n######################## OPTIMIZING MODEL ............... ########################\n")
        self.mgpr.optimize(restarts=restarts)
        # Print the resulting model parameters
        # ToDo: only do this if verbosity is large enough
        # lengthscales = {}; variances = {}; noises = {};

        # model_num = 1
        # gp_models = {}
        # for model in self.mgpr.models:
        #     best_params = {
        #         "lengthscales" : model.kernel.lengthscales,
        #         "k_variance" : model.kernel.variance,
        #         "l_variance" : model.likelihood.variance}
        #     gp_models[f"model_{model_num}"] = best_params
        #     lengthscales['GP' + str(model_num)] = model.kernel.lengthscales.numpy()
        #     variances['GP' + str(model_num)] = np.array([model.kernel.variance.numpy()])
        #     noises['GP' + str(model_num)] = np.array([model.likelihood.variance.numpy()])
        #     model_num += 1

        # print('-----Learned models------')
        # pd.set_option('precision', 3)
        # print('---Lengthscales---')
        # print(pd.DataFrame(data=lengthscales))
        # print('---Variances---')
        # print(pd.DataFrame(data=variances))
        # print('---Noises---')
        # print(pd.DataFrame(data=noises))

        if save_model:
            if not os.path.exists('models/pilco_mgpr/'):
                os.makedirs('models/pilco_mgpr/')
            self.mgpr.save_model('models/pilco_mgpr/')
            print("Saved optimized model parameters.")


    def load_model(self):
        self.mgpr.load_trained('models/pilco_mgpr/')
        print("Loaded optimized model parameters.")

            

    def optimize_policy(self, save_policy=False, maxiter=50, restarts=1):
        '''
        Optimize controller's parameter's
        '''
        print("\n\n######################## OPTIMIZING POLICY ............. ########################")
        start = time.time()
        mgpr_trainable_params = self.mgpr.trainable_parameters
        for param in mgpr_trainable_params:
            set_trainable(param, False)

        if not self.optimizer:
            self.optimizer = gpflow.optimizers.Scipy()
            # self.optimizer = tf.optimizers.Adam()
            self.optimizer.minimize(self.training_loss, self.trainable_variables, options=dict(maxiter=maxiter))
            # self.optimizer.minimize(self.training_loss, self.trainable_variables)
        else:
            self.optimizer.minimize(self.training_loss, self.trainable_variables, options=dict(maxiter=maxiter))
            # self.optimizer.minimize(self.training_loss, self.trainable_variables)
        end = time.time()
        print("Controller's optimization: done in %.1f seconds with reward=%.3f." % (end - start, self.compute_reward()))
        restarts -= 1

        best_parameter_values = [param.numpy() for param in self.trainable_parameters]
        best_reward = self.compute_reward()
        for restart in range(restarts):
            self.controller.randomize()
            start = time.time()
            self.optimizer.minimize(self.training_loss, self.trainable_variables, options=dict(maxiter=maxiter))
            end = time.time()
            reward = self.compute_reward()
            print("Controller's optimization: done in %.1f seconds with reward=%.3f." % (end - start, self.compute_reward()))
            if reward > best_reward:
                best_parameter_values = [param.numpy() for param in self.trainable_parameters]
                best_reward = reward

        for i,param in enumerate(self.trainable_parameters):
            param.assign(best_parameter_values[i])
        end = time.time()

        for param in mgpr_trainable_params:
            set_trainable(param, True)
        
        if save_policy:
            if not os.path.exists('models/pilco_policy/'):
                os.makedirs('models/pilco_policy/')
            filename = 'models/pilco_policy/' + 'best_param_values.pickle'
            with open(filename, 'wb') as handle:
                pickle.dump(best_parameter_values, handle, protocol=pickle.HIGHEST_PROTOCOL)




    def load_policy(self, saved_policy_path ='models/pilco_policy/'):
        mgpr_trainable_params = self.mgpr.trainable_parameters
        for param in mgpr_trainable_params:
            set_trainable(param, False)

        file_path = saved_policy_path +  'best_param_values.pickle'
        with open(file_path, 'rb') as file:
            best_param_values = pickle.load(file)
        for i,param in enumerate(self.trainable_parameters):
            param.assign(best_param_values[i])

        for param in mgpr_trainable_params:
            set_trainable(param, True)

        print("Loaded policy paramters.")

  




    def compute_action(self, x_m):
        return self.controller.compute_action(x_m, tf.zeros([self.state_dim, self.state_dim], float_type))[0]

    def predict(self, m_x, s_x, n):
        loop_vars = [
            tf.constant(0, tf.int32),
            m_x,
            s_x,
            tf.constant([[0]], float_type)
        ]

        _, m_x, s_x, reward = tf.while_loop(
            # Termination condition
            lambda j, m_x, s_x, reward: j < n,
            # Body function
            lambda j, m_x, s_x, reward: (
                j + 1,
                *self.propagate(m_x, s_x),
                tf.add(reward, self.reward.compute_reward(m_x, s_x)[0])
            ), loop_vars
        )
        return m_x, s_x, reward

    def propagate(self, m_x, s_x):
        m_u, s_u, c_xu = self.controller.compute_action(m_x, s_x)

        m = tf.concat([m_x, m_u], axis=1)
        s1 = tf.concat([s_x, s_x@c_xu], axis=1)
        s2 = tf.concat([tf.transpose(s_x@c_xu), s_u], axis=1)
        s = tf.concat([s1, s2], axis=0)

        M_dx, S_dx, C_dx = self.mgpr.predict_on_noisy_inputs(m, s)
        M_x = M_dx + m_x
        #TODO: cleanup the following line
        S_x = S_dx + s_x + s1@C_dx + tf.matmul(C_dx, s1, transpose_a=True, transpose_b=True)

        # While-loop requires the shapes of the outputs to be fixed
        M_x.set_shape([1, self.state_dim]); S_x.set_shape([self.state_dim, self.state_dim])
        return M_x, S_x

    def compute_reward(self):
        return -self.training_loss()

    @property
    def maximum_log_likelihood_objective(self):
        return -self.training_loss()