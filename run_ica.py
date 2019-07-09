import numpy as np
import pickle
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import seaborn as sns

from DataProcessor import DataProcessor
from models import ICA

__author__ = "Yifeng Tao"


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
      'dim_m':B.shape[0],
      'dim_n':B.shape[1],
      'dim_k':dim_k,
      'learning_rate':1e-5,
      'weight_decay':0
      }

  ica = ICA(args)

  ica.build()

  C, F, l2_train, l2_test = ica.train(B, M, M, max_iter=max_iter, inc=20000, verbose=True)

  BCF = {'B':B, 'C':C, 'F':F}

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

  sns.set_style('white')
  fig = plt.figure(figsize=(5,5))

  for idx in range(B.shape[1]):
    plt.plot(
        [B_pca[idx,0],CF_pca[idx,0]],
        [B_pca[idx,1],CF_pca[idx,1]],
        'gray',alpha=0.5)

  plt.plot(CF_pca[:,0], CF_pca[:,1], 'x', label='$\hat{B}=CF$')
  plt.plot(B_pca[:,0], B_pca[:,1], '+', label='$B$')

  plt.legend(prop={'size':size_legend})
  plt.xlabel('PCA 1', fontsize=size_label)
  plt.ylabel('PCA 2', fontsize=size_label)
  plt.tick_params(labelsize=size_tick)
  ##fig.savefig("figures/fig4pcaica.pdf", bbox_inches='tight')


def plot_F(F):
  """ Plot the distribution of deconvolved F.

  """
  size_label = 18
  size_tick = 18

  sns.set_style('darkgrid')
  fig = plt.figure()
  plt.hist(F.reshape(-1),bins=24)
  plt.xlabel('$F_{lj}^{\star}$', fontsize=size_label)
  plt.ylabel('Frequency', fontsize=size_label)
  plt.tick_params(labelsize=size_tick)
  ##fig.savefig("figures/fig5fdistribution.pdf", bbox_inches='tight')


data_proc = DataProcessor()
df_modu, len_kegg = data_proc.load_modu_data()
B = df_modu.values
dim_k = 5

#BCF = unmix(B, dim_k, max_iter=2000000)
#with open("data/ica/BCF.pkl", "wb") as f:
#  pickle.dump(BCF, f)
BCF = pickle.load(open('data/ica/BCF.pkl', 'rb'))

C, F = BCF['C'], BCF['F']
plot_B_CF(B, C, F)
plot_F(F)


