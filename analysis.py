""" Scripts for analyzing the results.

"""

import numpy as np
import pickle

from DataProcessor import DataProcessor
from utils_analysis import (
    hc_analysis, plot_3d_B, differential_pathway, component_portion,
    classify_patients, component_func, plot_phylo, plot_patients_F)

__author__ = "Yifeng Tao"


# Load data
data_proc = DataProcessor()
df_gene = data_proc.load_top_gene_data(top_k=3000)
df_modu, len_kegg = data_proc.load_modu_data()
BCF = pickle.load(open( "data/ica/BCF.pkl", "rb" ))
B, C, F = BCF["B"], BCF["C"], BCF["F"]
CF = np.dot(C, F)
df_modu = df_modu.drop(labels=["Pathways in cancer", "Glioma", "Breast cancer"], axis=0)
# 5, 24, 25
len_kegg -= 3
B = np.delete(B, [5, 24, 25], axis=0)
C = np.delete(C, [5, 24, 25], axis=0)
CF = np.delete(CF, [5, 24, 25], axis=0)


# Hierarchical clustering.
x = df_modu.values
hc_analysis(x, df_modu.columns, feature="Pathway")

x = df_gene.values
hc_analysis(x, df_modu.columns, feature="Genes")

# Differentially expressed pathways.
differential_pathway(df_modu, len_kegg, pval_threshold=1.0)

# Plot the PCA of bulk data
plot_3d_B(B, data_name="Pathway")

# Portions of components
comp_p = component_portion(F, plot_mode=True)

# Classify components of patients
list_patterns = classify_patients(F, threshold_0=2.5e-2)
plot_patients_F(F, threshold_0=2.5e-2)

# Functions of components
component_func(B, C, F, list(df_modu.index), len_kegg, threshold=0.05)

# Phylogeny of components.
for pattern in list_patterns:
  #pattern = list_patterns[0]
  plot_phylo(C, F, list(df_modu.index), len_kegg, comp_p, pattern, threshold=0.05)


