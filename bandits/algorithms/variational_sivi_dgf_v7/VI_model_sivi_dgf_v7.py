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


class VariationalSivi_dgf_v7(BayesianNN):
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

  
    #self.twolayerq = self.hparams.twolayerq
    self.glnoise = self.hparams.glnoise
    #self.klin = self.hparams.klin
    self.psigma = self.hparams.psigma
    self.two_decoder = self.hparams.two_decoder
 
    noi_dim = self.context_dim #min(self.context_dim, 150)
    
    self.noise_dim = [noi_dim, noi_dim, 50]
    #self.noise_dim = [100, 100, 50]
    self.implicit_hidden_layer_dim = [100, 100, 50]
    self.J = 1
    self.K = 50
    
    self.loss_set = []
    self.eps = 1e-32
    self.times_trained = 0
    self.step_trained = 0
    self.t = 0
    self.build_model()

  def sample_psi(self, x, noise_dim, hidden_dim, n, z_dim, reuse=False, a_fn=tf.nn.relu):
      scope_name = "sample_psi"
      with tf.variable_scope(scope_name) as scope:
          if reuse:
            scope.reuse_variables()
          x_0 = tf.expand_dims(x, axis=1)
          x_1 = tf.tile(x_0, [1, n, 1])  # N*K*784

          #B3 = Bernoulli(0.5)
          B3 = Normal(loc=0.0, scale=2.0)
          e3 = tf.cast(B3.sample([tf.shape(x)[0], n, noise_dim[0]]), tf.float32)
          input_ = tf.concat([e3, x_1], axis=2)
          h3 = slim.stack(input_, slim.fully_connected, [100]
                          , weights_regularizer=slim.l2_regularizer(1e-8), activation_fn=a_fn)

	  B2 = Normal(loc=0.0, scale=2.0)
	  #B2 = Bernoulli(0.5)
          e2 = tf.cast(B2.sample([tf.shape(x)[0], n, noise_dim[1]]), tf.float32)
          input_1 = tf.concat([h3, e2, x_1], axis=2)
          h2 = slim.stack(input_1, slim.fully_connected, [100]
                          , weights_regularizer=slim.l2_regularizer(1e-8), activation_fn=a_fn)


          mu = tf.reshape(slim.fully_connected(h3, z_dim, activation_fn=None, scope='implicit_hyper_mu'
                                               , weights_regularizer=slim.l2_regularizer(1e-8)), [-1, n, z_dim])

	  return mu

  def sample_n(self, psi, sigma):
      eps = tf.random_normal(shape=tf.shape(psi))
      z = psi + eps * sigma
      return z

  def sample_logv(self, x, z_dim, reuse=False, a_fn=tf.nn.relu):
      with tf.variable_scope("hyper_sigma") as scope:
          if reuse:
              scope.reuse_variables()
          net = slim.fully_connected(x, 50, activation_fn=a_fn, weights_regularizer=slim.l2_regularizer(1e-8))
          #net = slim.fully_connected(net, 50, activation_fn=tf.nn.relu, weights_regularizer=slim.l2_regularizer(1e-8))
          z_logv = tf.reshape(slim.fully_connected(net, z_dim, activation_fn=None, scope='z_log_variance'
                                                   , weights_regularizer=slim.l2_regularizer(1e-8)), [-1, z_dim])
          #alpha = slim.fully_connected(net, 1, activation_fn=tf.nn.softplus, scope='z_alpha'
          #                             , weights_regularizer=slim.l2_regularizer(1e-8))

      return z_logv


  def decoder(self, z, output_dim, a_fn = tf.nn.relu):

      scope_name = "decoder_1"
      with tf.variable_scope(scope_name, reuse=tf.AUTO_REUSE):
          half_latent_dim = 15
          x_0 = tf.expand_dims(self.x, axis=1)
          x_1 = tf.tile(x_0, [1, self.J, 1])
	  if self.glnoise:
            epsilon = tf.cast(Bernoulli(0.5).sample([tf.shape(self.x)[0], self.J, half_latent_dim]), tf.float32)
	    zx = tf.concat([z, x_1, epsilon], axis=2)
	  else:
	    zx = tf.concat([z, x_1], axis=2)

	  #z = tf.concat([z, x_1], axis=2)
	  #z = tf.concat([z, x_1, epsilon], axis=2)
	  net1 = slim.fully_connected(zx, 50, activation_fn=a_fn , weights_regularizer=slim.l2_regularizer(1e-8))
    	  #net1 = tf.nn.dropout(net1, rate=0.2)	  
	  #net1 = slim.stack(x_1, slim.fully_connected, [100, 50]
          #                  , weights_regularizer=slim.l2_regularizer(1e-8), activation_fn=a_fn)

	  #xz = tf.concat([net1, z], axis=2)

	  #net1 = slim.stack(xz, slim.fully_connected, [50]
          #                  , weights_regularizer=slim.l2_regularizer(1e-8), activation_fn=a_fn)
          
          # net1 = slim.fully_connected(net1, 50, activation_fn=a_fn, weights_regularizer=slim.l2_regularizer(1e-8))
          #net1 = tf.nn.dropout(net1, rate=0.2)
          mu = slim.fully_connected(net1, output_dim, activation_fn=None, scope='decoder_mu'
                                    , weights_regularizer=slim.l2_regularizer(1e-8))

          #net2 = slim.fully_connected(z, 50, activation_fn=a_fn, weights_regularizer=slim.l2_regularizer(1e-8))

	  #sigma = slim.fully_connected(net1, output_dim, activation_fn=None, scope='decoder_sigma'
          #                             , weights_regularizer=slim.l2_regularizer(1e-8))

          #return mu, tf.exp(sigma/2)

          #sigma = slim.fully_connected(net1, output_dim, activation_fn=None, scope='decoder_sigma'
          #                             , weights_regularizer=slim.l2_regularizer(1e-8))
          #                             , weights_initializer=tf.random_uniform_initializer(-0.05, 0.05))

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

        self.z_logv = self.sample_logv(self.x, self.latent_dim)
        self.q_sigma = tf.exp(self.z_logv/2)
	#self.q_sigma = tf.nn.softplus(self.z_logv/2)

        #log_q_sigma = tf.Variable(np.random.normal(loc=0.1, scale=0.05, size=(1, self.latent_dim)).astype(np.float32))
	#q_sigma = tf.exp(tf.Variable(np.random.normal(loc=0.2, scale=0.01)))
	#self.q_sigma = tf.tile(tf.reshape(q_sigma, [1, -1]), [tf.shape(self.x)[0], 1])

        #self.q_sigma = tf.exp(self.z_logv)
        #log_q_sigma = tf.Variable(np.ones((1, self.latent_dim)).astype(np.float32))
        #log_q_sigma = tf.Variable(tf.log(np.random.gamma(1.5, 1.0, (1, self.latent_dim)).astype(np.float32)))
        #self.q_sigma = tf.tile(tf.exp(log_q_sigma), [tf.shape(self.x)[0], 1])
        #self.q_sigma = tf.nn.softplus(self.z_logv)
        #self.q_sigma = tf.exp(self.z_logv/2)

        self.q_mu_K = self.sample_psi(self.x, self.noise_dim, self.implicit_hidden_layer_dim,
                                                      self.K, self.latent_dim, reuse=False)

        self.q_sigma_K = tf.tile(tf.expand_dims(self.q_sigma, axis=1), [1, self.K, 1])

        self.q_mu_J = self.sample_psi(self.x, self.noise_dim, self.implicit_hidden_layer_dim,
                                                      self.J, self.latent_dim, reuse=True)

        self.q_sigma_J = tf.tile(tf.expand_dims(self.q_sigma, axis=1), [1, self.J, 1])

        self.z_J = self.sample_n(self.q_mu_J, self.q_sigma_J)
        self.z_JK = tf.tile(tf.expand_dims(self.z_J, axis=2), [1, 1, self.K+1, 1])

        self.q_mu_KJ = tf.tile(tf.expand_dims(self.q_mu_K, axis=1), [1, self.J, 1, 1])
        self.q_mu_KJ_star = tf.concat([self.q_mu_KJ, tf.expand_dims(self.q_mu_J, axis=2)], axis=2)

        self.q_sigma_KJ = tf.tile(tf.expand_dims(self.q_sigma_K, axis=1), [1, self.J, 1, 1])
        self.q_sigma_KJ_star = tf.concat([self.q_sigma_KJ, tf.expand_dims(self.q_sigma_J, axis=2)], axis=2)


        #self.q_ker = tf.exp(- tf.reduce_sum(tf.square(self.z_JK - self.q_mu_KJ_star), axis=3))        
	#self.q_ker = tf.exp(- tf.reduce_sum(tf.square(self.z_JK - self.q_mu_KJ_star)) + logv_pram * tf.reduce_sum(tf.log(self.q_sigma_KJ_star + self.eps), axis=3))   

	self.q_ker = tf.exp(
            -0.5 * tf.reduce_sum(tf.square(self.z_JK - self.q_mu_KJ_star) / tf.square(self.q_sigma_KJ_star),
                                 axis=3))  
	

	# option-1 
        #self.log_q_prob = tf.log(tf.reduce_mean(self.q_ker, axis=2) + self.eps) + logv_pram * tf.reduce_sum(tf.log(self.q_sigma_J + self.eps), axis=2)

	#self.log_q_prob = tf.log(tf.reduce_mean(self.q_ker, axis=2) + self.eps)
        
	self.log_q_prob = tf.log(tf.reduce_mean(self.q_ker, axis=2) + self.eps) - tf.reduce_sum(tf.log(self.q_sigma_J+self.eps), axis=2)

	#self.log_q_prob = tf.log(tf.reduce_mean(self.q_prob, 2) + self.eps)
	# option-2
        #self.log_q_prob = tf.log(tf.reduce_mean(self.q_ker, axis=2) + self.eps) - 1e-2 * tf.log(self.q_alpha)
        
	#self.prior_sigma = tf.constant(self.psigma)
	self.prior_sigma = tf.Variable(self.psigma)
        self.log_prior_prob = tf.reduce_sum(-0.5 * tf.square(self.z_J / (self.prior_sigma+self.eps)) - tf.log(self.prior_sigma), axis=2)
 	#self.log_prior_prob = - tf.reduce_sum(tf.square(self.z_J / self.prior_sigma), axis=2)	

        self.y_pred_mu = self.decoder(self.z_J, self.num_actions)

        weights_J = tf.tile(tf.expand_dims(self.weights, axis=1), [1, self.J, 1])
        y_J = tf.tile(tf.expand_dims(self.y, axis=1), [1, self.J, 1])


	self.y_pred_sigma = tf.Variable(np.random.normal(loc=0.5, scale=0.05, size=(1, self.num_actions)).astype(np.float32))
        y_pred_sigma_J = tf.tile(tf.expand_dims(tf.tile(self.y_pred_sigma, [tf.shape(self.x)[0], 1]), axis=1), [1, self.J, 1])
	self.log_likelihood = tf.reduce_sum((- 0.5 * tf.square((y_J - self.y_pred_mu) / (y_pred_sigma_J+self.eps)) - tf.log(y_pred_sigma_J)) * weights_J, axis=2) * self.num_actions

        #self.log_likelihood = - tf.reduce_sum(tf.square(y_J - self.y_pred_mu) * weights_J, axis=2) * self.num_actions

      
        self.recon_loss = tf.reduce_mean(self.log_likelihood, axis=1)
        # self.recon_loss = tf.reduce_mean(tf.reduce_sum(self.log_likelihood, axis=2), axis=1) * self.num_actions
        self.kl_loss = tf.reduce_mean((self.log_prior_prob - self.log_q_prob), axis=1)

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

    if self.verbose:
      print("Training {} for {} steps...".format(self.name, self.times_trained))

    with self.graph.as_default():
      self.t = 0
      for step in range(num_steps):

        x, y, w = data.get_batch_with_weights(batch_size)
        n = x.shape[0]

        #if self.klin:
        #  kl_weight = min(0.2 + self.times_trained / 10, self.hparams.kl)
        #else:
        #  kl_weight = 1.0
        #kl_weight = min(self.step_trained / 2000, self.hparams.kl)
        kl_weight = min(self.times_trained / 15, self.hparams.kl)
        recon_weight = self.hparams.recon #np.random.normal(loc=1.2, scale=0.1) #self.num_contexts / self.hparams.batch_size

        if self.hparams.lr_decay:
            lr = initial_lr * 0.5 ** (step / num_steps)
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
             self.kl_loss, self.recon_loss, self.q_mu_J, self.q_sigma, self.y_pred_sigma],
            feed_dict={self.x: x, self.y: y, self.weights: w,
                       self.kl_weight: kl_weight,
                       self.recon_weight: recon_weight,
                       self.lr: lr})

	#print('q_mu: ', q_mu[0, 0])
        #print('y_sigma: ', y_sigma)
        #print('q_sigma: ', q_sigma[0])
	#print('prior_sigma: ', prior_sigma)
	#print(self.step_trained)
        if self.hparams.show_loss:
          print('loss: ', cost, "= kl_loss ", kl_loss.mean(), " + recon_loss ", recon_loss.mean())

        self.t += 1
        self.step_trained += 1

      self.times_trained += 1



