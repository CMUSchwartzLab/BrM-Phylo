# Phylogenies of Breast Cancer Brain Metastases

Contact: Yifeng Tao (yifengt@andrew.cmu.edu), Russell Schwartz (russells@andrew.cmu.edu)

## Introduction

This repository contains code for the following work:
**Phylogenies Derived from Matched Transcriptome Reveal the Evolution of Cell Populations and Temporal Order of Perturbed Pathways in Breast Cancer Brain Metastases**.

## Prerequisites

The code runs on `Python 2.7`.
* Common Python packages need to be installed: `os`, `random`, `numpy`, `pandas`, `pickle`, `scipy`, `sklearn`, `matplotlib`, `seaborn`, `cStringIO`, `collections`.
* These additional Python packages are required in some experiments: `statsmodels`, `networkx`, `skbio`, `Bio`, `PyTorch`.

## Usage

### Cross-validation

```python
python cv_ica.py
```

This step finds the optimal number of components for deconvolution. Results available at `data/ica/results_cv.pkl`.

### Deconvolution

```python
python run_ica.py
```

This unmixes the bulk data using the selected number of components (5 in our case). Results available at `data/ica/BCF.pkl`.

### Analysis

```python
python analysis.py
```


