from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import time
import pdb
import itertools
from collections import OrderedDict

import tensorflow.compat.v1 as tf
import numpy as np
from tqdm import trange
from scipy.io import savemat, loadmat

from tf_models.utils import get_required_argument, TensorStandardScaler
from tf_models.fc import FC

from tf_models.tf_logging import Progress, Silent

np.set_printoptions(precision=5)


class BNN:
    """Neural network models which model aleatoric uncertainty (and possibly epistemic uncertainty
    with ensembling).
    """
    def __init__(self, params):
        """Initializes a class instance.

        Arguments:
            params (DotMap): A dotmap of model parameters.
                .name (str): Model name, used for logging/use in variable scopes.
                    Warning: Models with the same name will overwrite each other.
                .num_networks (int): (optional) The number of networks in the ensemble. Defaults to 1.
                    Ignored if model is being loaded.
                .model_dir (str/None): (optional) Path to directory from which model will be loaded, and
                    saved by default. Defaults to None.
                .load_model (bool): (optional) If True, model will be loaded from the model directory,
                    assuming that the files are generated by a model of the same name. Defaults to False.
                .sess (tf.Session/None): The session that this model will use.
                    If None, creates a session with its own associated graph. Defaults to None.
        """
        self.name = get_required_argument(params, 'name', 'Must provide name.')
        self.model_dir = params.get('model_dir', None)

        print('[ BNN ] Initializing model: {} | {} networks | {} elites'.format(params['name'], params['num_networks'], params['num_elites']))
        if params.get('sess', None) is None:
            # config = tf.ConfigProto()
            # config.gpu_options.allow_growth = True
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.5)
            self._sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        else:
            self._sess = params.get('sess')

        # Instance variables
        self.finalized = False
        self.layers, self.max_logvar, self.min_logvar = [], None, None
        self.decays, self.optvars, self.nonoptvars = [], [], []
        self.end_act, self.end_act_name = None, None
        self.scaler = None

        # Training objects
        self.optimizer = None
        self.sy_train_in, self.sy_train_targ = None, None
        self.train_op, self.mse_loss = None, None

        # Prediction objects
        self.sy_pred_in2d, self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = None, None, None
        self.sy_pred_mean2d, self.sy_pred_var2d = None, None
        self.sy_pred_in3d, self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac = None, None, None

        if params.get('load_model', False):
            if self.model_dir is None:
                raise ValueError("Cannot load model without providing model directory.")
            self._load_structure()
            self.num_nets, self.model_loaded = self.layers[0].get_ensemble_size(), True
            print("Model loaded from %s." % self.model_dir)
            self.num_elites = params['num_elites']
        else:
            self.num_nets = params.get('num_networks', 1)
            self.num_elites = params['num_elites'] #params.get('num_elites', 1)
            self.model_loaded = False

        if self.num_nets == 1:
            print("Created a neural network with variance predictions.")
        else:
            print("Created an ensemble of {} neural networks with variance predictions | Elites: {}".format(self.num_nets, self.num_elites))

    @property
    def is_probabilistic(self):
        return True

    @property
    def is_tf_model(self):
        return True

    @property
    def sess(self):
        return self._sess

    ###################################
    # Network Structure Setup Methods #
    ###################################

    def add(self, layer):
        """Adds a new layer to the network.

        Arguments:
            layer: (layer) The new layer to be added to the network.
                   If this is the first layer, the input dimension of the layer must be set.

        Returns: None.
        """
        if self.finalized:
            raise RuntimeError("Cannot modify network structure after finalizing.")
        if len(self.layers) == 0 and layer.get_input_dim() is None:
            raise ValueError("Must set input dimension for the first layer.")
        if self.model_loaded:
            raise RuntimeError("Cannot add layers to a loaded model.")

        layer.set_ensemble_size(self.num_nets)
        if len(self.layers) > 0:
            layer.set_input_dim(self.layers[-1].get_output_dim())
        self.layers.append(layer.copy())

    def pop(self):
        """Removes and returns the most recently added layer to the network.

        Returns: (layer) The removed layer.
        """
        if len(self.layers) == 0:
            raise RuntimeError("Network is empty.")
        if self.finalized:
            raise RuntimeError("Cannot modify network structure after finalizing.")
        if self.model_loaded:
            raise RuntimeError("Cannot remove layers from a loaded model.")

        return self.layers.pop()

    def finalize(self, optimizer, optimizer_args=None, *args, **kwargs):
        """Finalizes the network.

        Arguments:
            optimizer: (tf.train.Optimizer) An optimizer class from those available at tf.train.Optimizer.
            optimizer_args: (dict) A dictionary of arguments for the __init__ method of the chosen optimizer.

        Returns: None
        """
        if len(self.layers) == 0:
            raise RuntimeError("Cannot finalize an empty network.")
        if self.finalized:
            raise RuntimeError("Can only finalize a network once.")

        optimizer_args = {} if optimizer_args is None else optimizer_args
        self.optimizer = optimizer(**optimizer_args)

        # Add variance output.
        self.layers[-1].set_output_dim(2 * self.layers[-1].get_output_dim())

        # Remove last activation to isolate variance from activation function.
        self.end_act = self.layers[-1].get_activation()
        self.end_act_name = self.layers[-1].get_activation(as_func=False)
        self.layers[-1].unset_activation()

        # Construct all variables.
        with self.sess.as_default():
            with tf.variable_scope(self.name):
                self.scaler = TensorStandardScaler(self.layers[0].get_input_dim())
                self.max_logvar = tf.Variable(np.ones([1, self.layers[-1].get_output_dim() // 2])/2., dtype=tf.float32,
                                              name="max_log_var")
                self.min_logvar = tf.Variable(-np.ones([1, self.layers[-1].get_output_dim() // 2])*10., dtype=tf.float32,
                                              name="min_log_var")
                for i, layer in enumerate(self.layers):
                    with tf.variable_scope("Layer%i" % i):
                        layer.construct_vars()
                        self.decays.extend(layer.get_decays())
                        self.optvars.extend(layer.get_vars())
        self.optvars.extend([self.max_logvar, self.min_logvar])
        self.nonoptvars.extend(self.scaler.get_vars())

        # Set up training
        with tf.variable_scope(self.name):
            self.optimizer = optimizer(**optimizer_args)
            self.sy_train_in = tf.placeholder(dtype=tf.float32,
                                              shape=[self.num_nets, None, self.layers[0].get_input_dim()],
                                              name="training_inputs")
            self.sy_train_targ = tf.placeholder(dtype=tf.float32,
                                                shape=[self.num_nets, None, self.layers[-1].get_output_dim() // 2],
                                                name="training_targets")
            train_loss = tf.reduce_sum(self._compile_losses(self.sy_train_in, self.sy_train_targ, inc_var_loss=True))
            train_loss += tf.add_n(self.decays)
            train_loss += 0.01 * tf.reduce_sum(self.max_logvar) - 0.01 * tf.reduce_sum(self.min_logvar)
            self.mse_loss = self._compile_losses(self.sy_train_in, self.sy_train_targ, inc_var_loss=False)

            self.train_op = self.optimizer.minimize(train_loss, var_list=self.optvars)

        # Initialize all variables
        self.sess.run(tf.variables_initializer(self.optvars + self.nonoptvars + self.optimizer.variables()))

        # Set up prediction
        with tf.variable_scope(self.name):
            self.sy_pred_in2d = tf.placeholder(dtype=tf.float32,
                                               shape=[None, self.layers[0].get_input_dim()],
                                               name="2D_training_inputs")
            self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac = \
                self.create_prediction_tensors(self.sy_pred_in2d, factored=True)
            self.sy_pred_mean2d = tf.reduce_mean(self.sy_pred_mean2d_fac, axis=0)
            self.sy_pred_var2d = tf.reduce_mean(self.sy_pred_var2d_fac, axis=0) + \
                tf.reduce_mean(tf.square(self.sy_pred_mean2d_fac - self.sy_pred_mean2d), axis=0)

            self.sy_pred_in3d = tf.placeholder(dtype=tf.float32,
                                               shape=[self.num_nets, None, self.layers[0].get_input_dim()],
                                               name="3D_training_inputs")
            self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac = \
                self.create_prediction_tensors(self.sy_pred_in3d, factored=True)

        # Load model if needed
        if self.model_loaded:
            with self.sess.as_default():
                params_dict = loadmat(os.path.join(self.model_dir, "%s.mat" % self.name))
                all_vars = self.nonoptvars + self.optvars
                for i, var in enumerate(all_vars):
                    var.load(params_dict[str(i)])
        self.finalized = True

    ##################
    # Custom Methods #
    ##################

    def _save_state(self, idx):
        self._state[idx] = [layer.get_model_vars(idx, self.sess) for layer in self.layers]

    def _set_state(self):
        keys = ['weights', 'biases']
        ops = []
        num_layers = len(self.layers)
        for layer in range(num_layers):
            # net_state = self._state[i]
            params = {key: np.stack([self._state[net][layer][key] for net in range(self.num_nets)]) for key in keys}
            ops.extend(self.layers[layer].set_model_vars(params))
        self.sess.run(ops)

    def _save_best(self, epoch, holdout_losses):
        updated = False
        for i in range(len(holdout_losses)):
            current = holdout_losses[i]
            _, best = self._snapshots[i]
            improvement = (best - current) / best
            if improvement > 0.01:
                self._snapshots[i] = (epoch, current)
                self._save_state(i)
                updated = True
                improvement = (best - current) / best
                # print('epoch {} | updated {} | improvement: {:.4f} | best: {:.4f} | current: {:.4f}'.format(epoch, i, improvement, best, current))

        if updated:
            self._epochs_since_update = 0
        else:
            self._epochs_since_update += 1

        if self._epochs_since_update > self._max_epochs_since_update:
            # print('[ BNN ] Breaking at epoch {}: {} epochs since update ({} max)'.format(epoch, self._epochs_since_update, self._max_epochs_since_update))
            return True
        else:
            return False

    def _start_train(self):
        self._state = {}
        self._snapshots = {i: (None, 1e10) for i in range(self.num_nets)}
        self._epochs_since_update = 0

    def _end_train(self, holdout_losses):
        sorted_inds = np.argsort(holdout_losses)
        self._model_inds = sorted_inds[:self.num_elites].tolist()
        print('Using {} / {} models: {}'.format(self.num_elites, self.num_nets, self._model_inds))

    def random_inds(self, batch_size):
        inds = np.random.choice(self._model_inds, size=batch_size)
        return inds

    def reset(self):
        print('[ BNN ] Resetting model')
        [layer.reset(self.sess) for layer in self.layers]

    def validate(self, inputs, targets):
        inputs = np.tile(inputs[None], [self.num_nets, 1, 1])
        targets = np.tile(targets[None], [self.num_nets, 1, 1])
        losses = self.sess.run(
            self.mse_loss,
            feed_dict={
                self.sy_train_in: inputs,
                self.sy_train_targ: targets
                }
        )
        mean_elite_loss = np.sort(losses)[:self.num_elites].mean()
        return mean_elite_loss

    #################
    # Model Methods #
    #################

    def train(self, inputs, targets,
              batch_size=32, max_epochs=None, max_epochs_since_update=5,
              hide_progress=False, holdout_ratio=0.0, max_logging=5000, max_grad_updates=None, timer=None, max_t=None):
        """Trains/Continues network training

        Arguments:
            inputs (np.ndarray): Network inputs in the training dataset in rows.
            targets (np.ndarray): Network target outputs in the training dataset in rows corresponding
                to the rows in inputs.
            batch_size (int): The minibatch size to be used for training.
            epochs (int): Number of epochs (full network passes that will be done.
            hide_progress (bool): If True, hides the progress bar shown at the beginning of training.

        Returns: None
        """
        # with open('./test.npy', 'wb') as f:
        #     np.save(f, inputs)
        #     np.save(f, targets)
        # exit()
        self._max_epochs_since_update = max_epochs_since_update
        self._start_train()
        break_train = False

        def shuffle_rows(arr):
            idxs = np.argsort(np.random.uniform(size=arr.shape), axis=-1)
            return arr[np.arange(arr.shape[0])[:, None], idxs]

        # Split into training and holdout sets
        num_holdout = min(int(inputs.shape[0] * holdout_ratio), max_logging)
        permutation = np.random.permutation(inputs.shape[0])
        inputs, holdout_inputs = inputs[permutation[num_holdout:]], inputs[permutation[:num_holdout]]
        targets, holdout_targets = targets[permutation[num_holdout:]], targets[permutation[:num_holdout]]
        holdout_inputs = np.tile(holdout_inputs[None], [self.num_nets, 1, 1])
        holdout_targets = np.tile(holdout_targets[None], [self.num_nets, 1, 1])

        print('[ BNN ] Training {} | Holdout: {}'.format(inputs.shape, holdout_inputs.shape))
        with self.sess.as_default():
            self.scaler.fit(inputs)

        idxs = np.random.randint(inputs.shape[0], size=[self.num_nets, inputs.shape[0]])
        if hide_progress:
            progress = Silent()
        else:
            progress = Progress(max_epochs)

        if max_epochs:
            epoch_iter = range(max_epochs)
        else:
            epoch_iter = itertools.count()
        # else:
        #     epoch_range = trange(epochs, unit="epoch(s)", desc="Network training")
        # trainable_vars = [v for v in tf.trainable_variables()]
        t0 = time.time()
        grad_updates = 0
        for epoch in epoch_iter:
            print('epoch:', epoch)
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                self.sess.run(
                    self.train_op,
                    feed_dict={self.sy_train_in: inputs[batch_idxs], self.sy_train_targ: targets[batch_idxs]}
                )
                grad_updates += 1

            idxs = shuffle_rows(idxs)
            if not hide_progress:
                if holdout_ratio < 1e-12:
                    losses = self.sess.run(
                            self.mse_loss,
                            feed_dict={
                                self.sy_train_in: inputs[idxs[:, :max_logging]],
                                self.sy_train_targ: targets[idxs[:, :max_logging]]
                            }
                        )
                    named_losses = [['M{}'.format(i), losses[i]] for i in range(len(losses))]
                    progress.set_description(named_losses)
                else:
                    losses = self.sess.run(
                            self.mse_loss,
                            feed_dict={
                                self.sy_train_in: inputs[idxs[:, :max_logging]],
                                self.sy_train_targ: targets[idxs[:, :max_logging]]
                            }
                        )
                    holdout_losses = self.sess.run(
                            self.mse_loss,
                            feed_dict={
                                self.sy_train_in: holdout_inputs,
                                self.sy_train_targ: holdout_targets
                            }
                        )
                    named_losses = [['M{}'.format(i), losses[i]] for i in range(len(losses))]
                    named_holdout_losses = [['V{}'.format(i), holdout_losses[i]] for i in range(len(holdout_losses))]
                    named_losses = named_losses + named_holdout_losses + [['T', time.time() - t0]]
                    progress.set_description(named_losses)
                    break_train = self._save_best(epoch, holdout_losses)

            progress.update()
            t = time.time() - t0
            if break_train or (max_grad_updates and grad_updates > max_grad_updates):
                break
            if max_t and t > max_t:
                descr = 'Breaking because of timeout: {}! (max: {})'.format(t, max_t)
                progress.append_description(descr)
                # print('Breaking because of timeout: {}! | (max: {})\n'.format(t, max_t))
                # time.sleep(5)
                break

        progress.stamp()
        if timer: timer.stamp('bnn_train')

        self._set_state()
        if timer: timer.stamp('bnn_set_state')

        holdout_losses = self.sess.run(
            self.mse_loss,
            feed_dict={
                self.sy_train_in: holdout_inputs,
                self.sy_train_targ: holdout_targets
            }
        )

        if timer: timer.stamp('bnn_holdout')

        self._end_train(holdout_losses)
        if timer: timer.stamp('bnn_end')

        val_loss = (np.sort(holdout_losses)[:self.num_elites]).mean()
        model_metrics = {'val_loss': val_loss}
        print('[ BNN ] Holdout', np.sort(holdout_losses), model_metrics)
        return OrderedDict(model_metrics)
        # return np.sort(holdout_losses)[]

        # pdb.set_trace()

    def predict(self, inputs, factored=False, *args, **kwargs):
        """Returns the distribution predicted by the model for each input vector in inputs.
        Behavior is affected by the dimensionality of inputs and factored as follows:

        inputs is 2D, factored=True: Each row is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of shape
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector j.

        inputs is 2D, factored=False: Each row is treated as an input vector.
            Returns a mean of shape [batch_size, output_dim] and variance of shape
            [batch_size, output_dim], where aggregation is performed as described in the paper.

        inputs is 3D, factored=True/False: Each row in the last dimension is treated as an input vector.
            Returns a mean of shape [ensemble_size, batch_size, output_dim] and variance of sha
            [ensemble_size, batch_size, output_dim], where N(mean[i, j, :], diag([i, j, :])) is the
            predicted output distribution by the ith model in the ensemble on input vector [i, j].

        Arguments:
            inputs (np.ndarray): An array of input vectors in rows. See above for behavior.
            factored (bool): See above for behavior.
        """
        if len(inputs.shape) == 2:
            if factored:
                return self.sess.run(
                    [self.sy_pred_mean2d_fac, self.sy_pred_var2d_fac],
                    feed_dict={self.sy_pred_in2d: inputs}
                )
            else:
                return self.sess.run(
                    [self.sy_pred_mean2d, self.sy_pred_var2d],
                    feed_dict={self.sy_pred_in2d: inputs}
                )
        else:
            return self.sess.run(
                [self.sy_pred_mean3d_fac, self.sy_pred_var3d_fac],
                feed_dict={self.sy_pred_in3d: inputs}
            )

    def create_prediction_tensors(self, inputs, factored=False, *args, **kwargs):
        """See predict() above for documentation.
        """
        factored_mean, factored_variance = self._compile_outputs(inputs)
        if inputs.shape.ndims == 2 and not factored:
            mean = tf.reduce_mean(factored_mean, axis=0)
            variance = tf.reduce_mean(tf.square(factored_mean - mean), axis=0) + \
                       tf.reduce_mean(factored_variance, axis=0)
            return mean, variance
        return factored_mean, factored_variance

    def save(self, savedir, timestep):
        """Saves all information required to recreate this model in two files in savedir
        (or self.model_dir if savedir is None), one containing the model structuure and the other
        containing all variables in the network.

        savedir (str): (Optional) Path to which files will be saved. If not provided, self.model_dir
            (the directory provided at initialization) will be used.
        """
        if not self.finalized:
            raise RuntimeError()
        model_dir = self.model_dir if savedir is None else savedir

        # Write structure to file
        with open(os.path.join(model_dir, '{}_{}.nns'.format(self.name, timestep)), "w+") as f:
            for layer in self.layers[:-1]:
                f.write("%s\n" % repr(layer))
            last_layer_copy = self.layers[-1].copy()
            last_layer_copy.set_activation(self.end_act_name)
            last_layer_copy.set_output_dim(last_layer_copy.get_output_dim() // 2)
            f.write("%s\n" % repr(last_layer_copy))

        # Save network parameters (including scalers) in a .mat file
        var_vals = {}
        for i, var_val in enumerate(self.sess.run(self.nonoptvars + self.optvars)):
            var_vals[str(i)] = var_val
        savemat(os.path.join(model_dir, '{}_{}.mat'.format(self.name, timestep)), var_vals)

    def _load_structure(self):
        """Uses the saved structure in self.model_dir with the name of this network to initialize
        the structure of this network.
        """
        structure = []
        with open(os.path.join(self.model_dir, "%s.nns" % self.name), "r") as f:
            for line in f:
                kwargs = {
                    key: val for (key, val) in
                    [argval.split("=") for argval in line[3:-2].split(", ")]
                }
                kwargs["input_dim"] = int(kwargs["input_dim"])
                kwargs["output_dim"] = int(kwargs["output_dim"])
                kwargs["weight_decay"] = None if kwargs["weight_decay"] == "None" else float(kwargs["weight_decay"])
                kwargs["activation"] = None if kwargs["activation"] == "None" else kwargs["activation"][1:-1]
                kwargs["ensemble_size"] = int(kwargs["ensemble_size"])
                structure.append(FC(**kwargs))
        self.layers = structure

    #######################
    # Compilation methods #
    #######################

    def _compile_outputs(self, inputs, ret_log_var=False):
        """Compiles the output of the network at the given inputs.

        If inputs is 2D, returns a 3D tensor where output[i] is the output of the ith network in the ensemble.
        If inputs is 3D, returns a 3D tensor where output[i] is the output of the ith network on the ith input matrix.

        Arguments:
            inputs: (tf.Tensor) A tensor representing the inputs to the network
            ret_log_var: (bool) If True, returns the log variance instead of the variance.

        Returns: (tf.Tensors) The mean and variance/log variance predictions at inputs for each network
            in the ensemble.
        """
        dim_output = self.layers[-1].get_output_dim()
        cur_out = self.scaler.transform(inputs)
        for layer in self.layers:
            cur_out = layer.compute_output_tensor(cur_out)

        mean = cur_out[:, :, :dim_output//2]
        if self.end_act is not None:
            mean = self.end_act(mean)

        logvar = self.max_logvar - tf.nn.softplus(self.max_logvar - cur_out[:, :, dim_output//2:])
        logvar = self.min_logvar + tf.nn.softplus(logvar - self.min_logvar)

        if ret_log_var:
            return mean, logvar
        else:
            return mean, tf.exp(logvar)

    def _compile_losses(self, inputs, targets, inc_var_loss=True):
        """Helper method for compiling the loss function.

        The loss function is obtained from the log likelihood, assuming that the output
        distribution is Gaussian, with both mean and (diagonal) covariance matrix being determined
        by network outputs.

        Arguments:
            inputs: (tf.Tensor) A tensor representing the input batch
            targets: (tf.Tensor) The desired targets for each input vector in inputs.
            inc_var_loss: (bool) If True, includes log variance loss.

        Returns: (tf.Tensor) A tensor representing the loss on the input arguments.
        """
        mean, log_var = self._compile_outputs(inputs, ret_log_var=True)
        inv_var = tf.exp(-log_var)

        if inc_var_loss:
            mse_losses = tf.reduce_mean(tf.reduce_mean(tf.square(mean - targets) * inv_var, axis=-1), axis=-1)
            var_losses = tf.reduce_mean(tf.reduce_mean(log_var, axis=-1), axis=-1)
            total_losses = mse_losses + var_losses
        else:
            total_losses = tf.reduce_mean(tf.reduce_mean(tf.square(mean - targets), axis=-1), axis=-1)

        return total_losses