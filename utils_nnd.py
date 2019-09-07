""" Util functions for run_nnd.py.

"""

import random
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

from models import NND

__author__ = "Yifeng Tao"


def cv_nnd(B, n_comp, n_splits):
  """ Cross-validation of matrix factorization.

  Parameters
  ----------
  B: matrix
    bulk data to be deconvolved.
  n_comp: list int
    numbers of population component.
  n_splits: int
    fold of cross-validation.

  Returns
  -------
  results: dict
    numbers of components, training errors and test errors.
  """

  results = {
      "n_comp":n_comp,
      "test_error":[[] for _ in range(len(n_comp))],
      "train_error":[[] for _ in range(len(n_comp))]
      }

  rng = [(idx, idy) for idx in range(B.shape[0]) for idy in range(B.shape[1])]
  random.Random(2019).shuffle(rng)

  kf = KFold(n_splits=n_splits)

  idx_fold = 0
  for train_index, test_index in kf.split(rng):
    idx_fold += 1

    rng_train = [rng[i] for i in train_index]
    rng_test = [rng[i] for i in test_index]

    M_test = np.zeros(B.shape)
    for r in rng_test:
      M_test[r[0],r[1]] = 1.0
    M_train = np.zeros(B.shape)
    for r in rng_train:
      M_train[r[0],r[1]] = 1.0

    for idx_trial in range(len(n_comp)):
      dim_k = results["n_comp"][idx_trial]

      args = {
          "dim_m":B.shape[0],
          "dim_n":B.shape[1],
          "dim_k":dim_k,
          "learning_rate":1e-5, #1e-4
          "weight_decay":0
          }

      nnd = NND(args)
      nnd.build()

      #C, F, l2_train, l2_test = nnd.train(B, M_train, M_test, max_iter=200000, inc=10000, verbose=True)
      C, F, l2_train, l2_test = nnd.train(B, M_train, M_test, max_iter=2000000, inc=20000)

      results["train_error"][idx_trial].append(l2_train)
      results["test_error"][idx_trial].append(l2_test)

      print("fold=%3d/%3d, dim_k=%2d, train=%.2e, test=%.2e"%(idx_fold, n_splits, dim_k, l2_train, l2_test))

  return results


def plot_cv_nnd(results):
  """ Plot the cross-validation results.

  Parameters
  ----------
  results: dict

  Returns
  -------
  dim_k: iint
    optimal numbers of components based on cross-validation
  """

  size_label = 18
  size_tick = 18
  sns.set_style("darkgrid")

  fig = plt.figure(figsize=(5,4))
  M_rst = []
  n_comp = results["n_comp"]
  M_test_error = np.asarray(results["test_error"])
  for idx, k in enumerate(n_comp):
    for v in M_test_error[idx]:
      M_rst.append([k, v])

  df = pd.DataFrame(
      data=M_rst,
      index=None,
      columns=["# comp", "test error"])
  avg_test_error = M_test_error.mean(axis=1)
  ax = sns.lineplot(x="# comp", y="test error", markers=True, data=df)

  idx_min = np.argmin(avg_test_error)
  plt.plot(n_comp[idx_min], avg_test_error[idx_min],"*", markersize=15)

  plt.ylabel("Test MSE", fontsize=size_label)
  plt.xlabel("# components", fontsize=size_label)
  plt.tick_params(labelsize=size_tick)
  plt.xlim([2, 7])
  plt.ylim([0.55,0.95])

  plt.show()
  ##fig.savefig("figures/fig1cvcomp.pdf", bbox_inches="tight")

  dim_k = n_comp[idx_min]

  return dim_k


def unmix(B, dim_k, max_iter=200000):
  """ Implement matrix factorization to unmix the bulk data.
  B \approx C F

  Parameters
  ----------
  B: matrix
    bulk data
  dim_k: int
    number of components
  max_iter: int
    maximum number of iterations
  """

  M = np.ones(B.shape)

  args = {
      "dim_m":B.shape[0],
      "dim_n":B.shape[1],
      "dim_k":dim_k,
      "learning_rate":1e-5,
      "weight_decay":0
      }

  nnd = NND(args)

  nnd.build()

  C, F, l2_train, l2_test = nnd.train(B, M, M, max_iter=max_iter, inc=20000, verbose=True)

  BCF = {"B":B, "C":C, "F":F}

  plt.figure()
  plt.hist(C.reshape(-1))
  plt.figure()
  plt.hist(F.reshape(-1))

  return BCF


def plot_B_CF(B, C, F):
  """ PCA plot of both B and CF. Each sample is a dot.

  """

  size_label = 18
  size_legend = 18
  size_tick = 18

  CF = np.dot(C, F)
  pca = PCA(n_components=50)
  pca.fit(np.concatenate((CF, B), axis=1).T)
  B_pca = pca.transform(B.T)
  CF_pca = pca.transform(CF.T)

  sns.set_style("white")
  fig = plt.figure(figsize=(5,5))

  for idx in range(B.shape[1]):
    plt.plot(
        [B_pca[idx,0],CF_pca[idx,0]],
        [B_pca[idx,1],CF_pca[idx,1]],
        "gray",alpha=0.5)

  plt.plot(CF_pca[:,0], CF_pca[:,1], "x", label="$\hat{B}=CF$")
  plt.plot(B_pca[:,0], B_pca[:,1], "+", label="$B$")

  plt.legend(prop={"size":size_legend})
  plt.xlabel("PCA 1", fontsize=size_label)
  plt.ylabel("PCA 2", fontsize=size_label)
  plt.tick_params(labelsize=size_tick)
  ##fig.savefig("figures/fig4pcannd.pdf", bbox_inches="tight")


def plot_F(F):
  """ Plot the distribution of deconvolved F.

  """
  size_label = 18
  size_tick = 18

  sns.set_style("darkgrid")
  fig = plt.figure()
  plt.hist(F.reshape(-1),bins=24)
  plt.xlabel("$F_{lj}^{\star}$", fontsize=size_label)
  plt.ylabel("Frequency", fontsize=size_label)
  plt.tick_params(labelsize=size_tick)
  ##fig.savefig("figures/fig5fdistribution.pdf", bbox_inches="tight")





