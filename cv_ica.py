""" Cross validation of matrix factorization and plot results.

"""

import random
import numpy as np
import pandas as pd
import pickle

from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

from DataProcessor import DataProcessor
from models import ICA

__author__ = "Yifeng Tao"


def cv_ica(B, n_comp, n_splits):
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

      ica = ICA(args)
      ica.build()

      #C, F, l2_train, l2_test = ica.train(B, M_train, M_test, max_iter=200000, inc=10000, verbose=True)
      C, F, l2_train, l2_test = ica.train(B, M_train, M_test, max_iter=2000000, inc=20000)

      results["train_error"][idx_trial].append(l2_train)
      results["test_error"][idx_trial].append(l2_test)

      print("fold=%3d/%3d, dim_k=%2d, train=%.2e, test=%.2e"%(idx_fold, n_splits, dim_k, l2_train, l2_test))

  return results


def plot_cv_ica(results):
  """ Plot the cross-validation results.

  Parameters
  ----------
  results: dict
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


# Load bulk data.
data_proc = DataProcessor()
df_modu, len_kegg = data_proc.load_modu_data()
B = df_modu.values

# Cross-validation
n_comp = [2,3,4,5,6,7]
n_splits = 20

#results = cv_ica(B, n_comp, n_splits)

#pickle.dump(results, open("data/ica/results_cv.pkl", "wb"))


# Plot the error vs. # components
results = pickle.load(open("data/ica/results_cv.pkl", "rb"))
plot_cv_ica(results)


