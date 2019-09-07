""" Cross validation of matrix factorization and plot results, and then conduct
the factorization using optimal number of components.

"""

import pickle
from DataProcessor import DataProcessor

from utils_nnd import cv_nnd, plot_cv_nnd, unmix, plot_B_CF, plot_F

__author__ = "Yifeng Tao"


## Load bulk data.
data_proc = DataProcessor()
df_modu, len_kegg = data_proc.load_modu_data()
B = df_modu.values

## Cross-validation
n_comp = [2,3,4,5,6,7]
n_splits = 20

#results = cv_nnd(B, n_comp, n_splits)

#pickle.dump(results, open("data/ica/results_cv.pkl", "wb"))


## Plot the error vs. # components
results = pickle.load(open("data/ica/results_cv.pkl", "rb"))
# Find optimal number of components
dim_k = plot_cv_nnd(results) #dim_k = 5

## Conduct deconvolution using optimal # component
#BCF = unmix(B, dim_k, max_iter=2000000)
#with open("data/ica/BCF.pkl", "wb") as f:
#  pickle.dump(BCF, f)
BCF = pickle.load(open("data/ica/BCF.pkl", "rb"))

C, F = BCF["C"], BCF["F"]
plot_B_CF(B, C, F)
plot_F(F)

