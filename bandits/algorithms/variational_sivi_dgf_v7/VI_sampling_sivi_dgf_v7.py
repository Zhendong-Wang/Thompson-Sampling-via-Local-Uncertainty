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

"""Thompson Sampling with linear posterior over a learnt deep representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.algorithms.variational_sivi_dgf_v7.VI_dataset import VIContextualDataset
from bandits.algorithms.variational_sivi_dgf_v7.VI_model_sivi_dgf_v7 import VariationalSivi_dgf_v7


class VariationalSamplingSivi_dgf_v7(BanditAlgorithm):
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, name, hparams, optimizer='RMS', mode='variational'):

    self.name = name
    self.hparams = hparams
    self.optimizer_n = optimizer

    self.training_freq = hparams.training_freq
    self.training_epochs = hparams.training_epochs
    self.num_actions = hparams.num_actions
    self.t = 0
    self.data_h = VIContextualDataset(hparams.context_dim, hparams.num_actions,
                                    hparams.buffer_s)

    self.bnn = VariationalSivi_dgf_v7(optimizer, hparams, '{}-bnn'.format(name))
    self.loss = self.bnn.loss_set

  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      return self.t % self.hparams.num_actions

    with self.bnn.graph.as_default():
      self.c = context.reshape((1, self.hparams.context_dim))
      y_pred_mu = self.bnn.sess.run(self.bnn.y_pred_mu, feed_dict={self.bnn.x: self.c})
      r = y_pred_mu.mean(axis=0).mean(axis=0)
      # r = np.random.normal(y_pred_mu, y_pred_var)
      return np.argmax(r)

  def update(self, context, action, reward):
    """Updates the posterior using linear bayesian regression formula."""

    self.t += 1

    self.data_h.add(context, action, reward)
    
    if self.t % self.training_freq == 0:
      self.bnn.train(self.data_h, self.hparams.batch_size, self.hparams.training_epochs, self.t, self.hparams.initial_lr)

    #if self.t < 300:
      #if self.t % self.training_freq == 0:
        #self.bnn.train(self.data_h, 256, 75, self.t, self.hparams.initial_lr)
    #else:
      #if self.t % self.training_freq == 0:
        #self.bnn.train(self.data_h, self.hparams.batch_size, 120, self.t, self.hparams.initial_lr)

    #if self.t < 300:
      #if self.t % self.training_freq == 0:
        #self.bnn.train(self.data_h, self.hparams.batch_size, 75, self.t, 0.001)
    #elif self.t < 800:
      #if self.t % self.training_freq == 0:
        #self.bnn.train(self.data_h, self.hparams.batch_size, 150, self.t, 0.001)
    #else:
      #if self.t % self.training_freq == 0:
        #self.bnn.train(self.data_h, 512, 200, self.t, self.hparams.initial_lr)


