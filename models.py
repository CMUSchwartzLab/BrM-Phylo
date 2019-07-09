""" Matrix deconvolution models.

"""

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

__author__ = "Yifeng Tao"


def _wrap_data(mat):
  """ Wrap default numpy or list data into PyTorch variables.

  """
  return Variable(torch.FloatTensor(mat))


class ModelBase(nn.Module):
  """ Base models for all models.
  """

  def __init__(self, args):
    """ Initialize the hyperparameters of model.

    Parameters
    ----------
    args: arguments for initializing the model.
    """

    super(ModelBase, self).__init__()

    self.epsilon = 1e-10 #1e-4

    self.dim_m = args["dim_m"]
    self.dim_n = args["dim_n"]
    self.dim_k = args["dim_k"]
    self.learning_rate = args["learning_rate"]
    self.weight_decay = args["weight_decay"]


  def build(self):
    """ Define modules of the model.
    """

    raise NotImplementedError


class ICA(ModelBase):
  """ ICA model for deconvolution.
  """

  def __init__(self, args, **kwargs):
    """ Initialize the model.

    Parameters
    ----------
    args: arguments for initializing the model.
    """

    super(ICA, self).__init__(args, **kwargs)


  def build(self):
    """ Define modules of the model.

    """

    self.mat_c = torch.nn.Parameter(
        data=torch.Tensor(self.dim_m, self.dim_k), requires_grad=True)
    self.mat_c.data.uniform_(-1, 1)

    self.mat_f = torch.nn.Parameter(
        data=torch.Tensor(self.dim_k, self.dim_n), requires_grad=True)
    self.mat_f.data.uniform_(0, 1)

    self.optimizer = optim.Adam(
        self.parameters(),
        lr=self.learning_rate,
        weight_decay=self.weight_decay)

  def forward(self):
    """ Forward parameters to the output of estimated/predicted B.

    """
    mat_f_abs = torch.abs(self.mat_f)
    mat_f = F.normalize(mat_f_abs, p=1, dim=0)
    mat_p = torch.mm(self.mat_c, mat_f)

    return mat_p, mat_f


  def train(self, mat_b, M_train, M_test, max_iter=None, inc=1, verbose=False):
    """ Train the matrix factorization using gradient descent and monitor.

    Parameters
    ----------
    mat_b: numpy matrix
      bulk data, each column a sample, each row a gene module.
    M_train: numpy 0/1 mask matrix
      same size of mat_b, positions of 1 mean seen data, otherwise unseen.
    max_iter: int
      max iterations of training.
    inc: int
      intervals to evaluate the training.
    verbose: boolen
      whether print too much itermediat results.

    Returns
    -------
      Deconvolved matrices C, F. Traing L2 loss and test L2 loss.
    """

    mat_b = _wrap_data(mat_b)
    M_train = _wrap_data(M_train)
    M_test = _wrap_data(M_test)

    previous_error = 1e10

    for iter_train in range(0, max_iter+1):
      mat_p, mat_f = self.forward()
      self.optimizer.zero_grad()

      loss = torch.norm((mat_p-mat_b)*M_train, 2)**2 / M_train.sum()
      loss.backward()
      self.optimizer.step()

      if iter_train % inc == 0:
        l2 = 1.0*loss.data.numpy() / M_train.sum().numpy()
        if verbose:
          print("iter=%d, l2=%.2e"% (iter_train,l2))

        if (previous_error - l2) / previous_error < self.epsilon:
          break
        previous_error = l2

    if iter_train >= max_iter-2*inc:
      print("warning: max_iter too small...")
    loss = torch.norm((mat_p-mat_b)*M_test, 2)**2
    l2_test = loss.data.numpy() / M_test.sum().numpy()

    return np.array(self.mat_c.data.numpy(), dtype=float), np.array(mat_f.data.numpy(), dtype=float), l2, l2_test


