# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Define a family of neural network architectures for bandits.

The network accepts different type of optimizers that could lead to different
approximations of the posterior distribution or simply to point estimates.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from absl import flags
from bandits.core.bayesian_nn import BayesianNN

FLAGS = flags.FLAGS
slim = tf.contrib.slim
Bernoulli = tf.contrib.distributions.Bernoulli
Normal = tf.contrib.distributions.Normal


class Variational_v4(BayesianNN):
    """Implements a neural network for bandit problems."""

    def __init__(self, optimizer, hparams, name):
        """Saves hyper-params and builds the Tensorflow graph."""

        self.opt_name = optimizer
        self.name = name
        self.hparams = hparams
        self.verbose = getattr(self.hparams, "verbose", True)
        self.num_actions = self.hparams.num_actions
        self.num_contexts = self.hparams.num_contexts
        self.context_dim = self.hparams.context_dim
        self.latent_dim = self.hparams.latent_dim

        self.psigma = self.hparams.psigma
	self.glnoise = self.hparams.glnoise

        self.eps = 1e-32
        self.times_trained = 0
        self.step_trained = 0
        self.t = 0
        self.build_model()

    def encoder(self, x, z_dim, reuse=False, a_fn=tf.nn.relu):
        scope_name = "encoder"
        with tf.variable_scope(scope_name) as scope:
            if reuse:
                scope.reuse_variables()

            h = slim.stack(x, slim.fully_connected, [100, 50],
                            weights_regularizer=slim.l2_regularizer(1e-8), activation_fn=a_fn)

            mu = slim.fully_connected(h, z_dim, activation_fn=None,
                                      scope='encoder_mu', weights_regularizer=slim.l2_regularizer(1e-8))

            sigma = slim.fully_connected(h, z_dim, activation_fn=None,
                                      scope='encoder_sigma', weights_regularizer=slim.l2_regularizer(1e-8))

            return mu, tf.exp(sigma/2)

    def sample_n(self, psi, sigma):
        eps = tf.random_normal(shape=tf.shape(psi))
        z = psi + eps * sigma
        return z

    def decoder(self, z, output_dim, a_fn=tf.nn.relu):

        scope_name = "decoder_1"
        with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):

	    if self.glnoise:
		epsilon = tf.cast(Bernoulli(0.5).sample([tf.shape(self.x)[0], 15]), tf.float32)
	        zx = tf.concat([z, self.x, epsilon], axis=1)
	    else:
                zx = tf.concat([z, self.x], axis=1)

            # h = slim.fully_connected(zx, 50, activation_fn=a_fn, weights_regularizer=slim.l2_regularizer(1e-8))

            h = slim.stack(zx, slim.fully_connected, [100, 50],
                           weights_regularizer=slim.l2_regularizer(1e-8), activation_fn=a_fn)

            mu = slim.fully_connected(h, output_dim, activation_fn=None, scope='decoder_mu'
                                      , weights_regularizer=slim.l2_regularizer(1e-8))

            return mu

    def build_model(self):
        """Defines the actual NN model with fully connected layers.

        The loss is computed for partial feedback settings (bandits), so only
        the observed outcome is backpropagated (see weighted loss).
        Selects the optimizer and, finally, it also initializes the graph.
        """

        # create and store the graph corresponding to the BNN instance
        self.graph = tf.Graph()

        with self.graph.as_default():
            # create and store a new session for the graph
            self.sess = tf.Session()

            with tf.name_scope(self.name):
                self.global_step = tf.train.get_or_create_global_step()

                # context
                self.x = tf.placeholder(
                    shape=[None, self.hparams.context_dim],
                    dtype=tf.float32,
                    name="{}_x".format(self.name))

                # reward vector
                self.y = tf.placeholder(
                    shape=[None, self.hparams.num_actions],
                    dtype=tf.float32,
                    name="{}_y".format(self.name))

                # weights (1 for selected action, 0 otherwise)
                self.weights = tf.placeholder(
                    shape=[None, self.hparams.num_actions],
                    dtype=tf.float32,
                    name="{}_w".format(self.name))

                # KL term weight
                self.kl_weight = tf.placeholder(
                    dtype=tf.float32,
                    name="{}_kl_weight".format(self.name))

                # recom term weight
                self.recon_weight = tf.placeholder(
                    dtype=tf.float32,
                    name="{}_recon_weight".format(self.name))

                self.q_mu, self.q_sigma = self.encoder(self.x, self.latent_dim)

                self.z = self.sample_n(self.q_mu, self.q_sigma)

                self.log_q_prob = tf.reduce_sum(- 0.5 * tf.square((self.z - self.q_mu) / (self.q_sigma + self.eps))
                                                - tf.log(self.q_sigma), axis=1)

                self.prior_sigma = tf.Variable(self.psigma)
                self.log_prior_prob = tf.reduce_sum(-0.5 * tf.square(self.z / (self.prior_sigma + self.eps))
                                                    - tf.log(self.prior_sigma), axis=1)

                # self.prior_mu, self.prior_sigma = self.prior_pass(self.latent_dim)
                # self.log_prior_prob = tf.reduce_sum(-0.5 * tf.square((self.z - self.prior_mu) / (self.prior_sigma + self.eps))
                #                                     - tf.log(self.prior_sigma), axis=1)


                self.y_pred_mu = self.decoder(self.z, self.num_actions)

                y_pred_sigma = tf.Variable(
                     np.random.normal(loc=0.5, scale=0.05, size=(1, self.num_actions)).astype(np.float32))
                self.y_pred_sigma = tf.tile(y_pred_sigma, [tf.shape(self.x)[0], 1])

                self.log_likelihood = tf.reduce_sum((- 0.5 * tf.square(
                    (self.y - self.y_pred_mu) / (self.y_pred_sigma + self.eps)) - tf.log(self.y_pred_sigma)) * self.weights,
                                                    axis=1) * self.num_actions

                self.recon_loss = self.log_likelihood
                # self.recon_loss = tf.reduce_mean(tf.reduce_sum(self.log_likelihood, axis=2), axis=1) * self.num_actions
                self.kl_loss = self.log_prior_prob - self.log_q_prob

                self.kl_loss_star = self.kl_loss * self.kl_weight

                self.recon_loss_star = self.recon_loss * self.recon_weight

                self.neg_elbo = tf.reduce_mean(-(self.recon_loss_star + self.kl_loss_star))

                self.lr = tf.placeholder(
                    dtype=tf.float32,
                    name="{}_lr".format(self.name))

                self.train_op = tf.train.AdamOptimizer(self.lr).minimize(
                    self.neg_elbo, global_step=self.global_step)

                self.init = tf.global_variables_initializer()

                self.initialize_graph()

    def initialize_graph(self):
        """Initializes all variables."""

        with self.graph.as_default():
            if self.verbose:
                print("Initializing model {}.".format(self.name))
            self.sess.run(self.init)

    def select_optimizer(self):
        """Selects optimizer. To be extended (SGLD, KFAC, etc)."""
        # return tf.train.GradientDescentOptimizer(self.lr)
        return tf.train.AdamOptimizer(self.lr)
        # return tf.train.RMSPropOptimizer(self.lr)

    def train(self, data, batch_size, num_steps, global_t, initial_lr):
        """Trains the network for num_steps, using the provided data.

        Args:
          data: ContextualDataset object that provides the data.
          num_steps: Number of minibatches to train the network for.
        """

        # if self.verbose:
        #  print("Training {} for {} steps...".format(self.name, self.times_trained))

        with self.graph.as_default():
            self.t = 0
            for step in range(num_steps):

                x, y, w = data.get_batch_with_weights(batch_size)
                n = x.shape[0]

                # if self.klin:
                #  kl_weight = min(0.2 + self.times_trained / 10, self.hparams.kl)
                # else:
                #  kl_weight = 1.0
                # kl_weight = min(self.step_trained / 2000, self.hparams.kl)
                kl_weight = min(self.times_trained / 15, self.hparams.kl)
                recon_weight = self.hparams.recon  # np.random.normal(loc=1.2, scale=0.1) #self.num_contexts / self.hparams.batch_size

                if self.hparams.lr_decay:
                    # lr = initial_lr * 0.5 ** (step / num_steps * 2)
                    lr = initial_lr * 0.5 ** (self.times_trained / 500)
                else:
                    lr = initial_lr

                # kl_weight = min(self.times_trained / 300, 1.0)
                # recon_weight = self.num_contexts / self.hparams.batch_size #np.random.normal(loc=1.5, scale=0.25)
                # lr = self.hparams.initial_lr * 0.75 ** (self.times_trained / 100)

                # kl_loss, recon_loss, recon_loss_star = self.sess.run(
                #     [self.kl_loss, self.recon_loss, self.recon_loss_star],
                #     feed_dict={self.x: x, self.y: y, self.weights: w,
                #                self.kl_weight: kl_weight, self.recon_weight: recon_weight})

                _, cost, kl_loss, recon_loss, q_mu, q_sigma, y_sigma = self.sess.run(
                    [self.train_op, self.neg_elbo,
                     self.kl_loss, self.recon_loss, self.q_mu, self.q_sigma, self.y_pred_sigma],
                    feed_dict={self.x: x, self.y: y, self.weights: w,
                               self.kl_weight: kl_weight,
                               self.recon_weight: recon_weight,
                               self.lr: lr})

                # print('q_mu: ', q_mu[0, 0])
                # print('y_sigma: ', y_sigma)
                # print('q_sigma: ', q_sigma[0])
                # print('prior_sigma: ', prior_sigma)
                # print(self.step_trained)
                if self.hparams.show_loss:
                    print('loss: ', cost, "= kl_loss ", kl_loss.mean(), " + recon_loss ", recon_loss.mean())

                self.t += 1
                self.step_trained += 1

            self.times_trained += 1

