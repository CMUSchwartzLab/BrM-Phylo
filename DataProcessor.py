import os
import numpy as np
import pandas as pd
import scipy

__author__ = "Yifeng Tao"


class DataProcessor():
  """" Data processor to preprocess the raw BrM data (brainMetPairs.salmon.cts.txt).
  Usage:
    dataprocessor = DataProcessor()
    df_gene = dataprocessor.load_top_gene_data(top_k=3000)
    df_modu, len_kegg = dataprocessor.load_modu_data()
  """

  def __init__(self, path='data/brainMetPairs.salmon.cts.txt'):
    """
    path: path to the raw transcriptome data.
    """
    # self.df: raw gene data
    self.df = pd.read_csv(path, sep='\t', index_col=0)
    self.ensg2gene = self.get_ensg2gene()
    self.build_data()


  def get_ensg2gene(self, path="data/genelist/list_gene_ensg.txt"):
    """ Load the mapping from ENSG to gene name.
    """
    ensg2gene = {}
    with open(path, "r") as f:
      for line in f:
        line = line.strip("\n").split(":")
        ensg = line[0][0:-5]
        gene = line[1]
        if gene == "no_match":
          gene = ensg
        ensg2gene[ensg] = gene
    return ensg2gene


  def load_prot_gene(self, path='data/genelist/mart_export.txt'):
    """ Load list of protein coding genes.
    """
    prot_gene = []
    with open(path,'r') as f:
      next(f)
      idx = 1
      for line in f:
        idx += 1
        line = line.strip()
        line = line.split('\t')
        if len(line) > 1:
          prot_gene.append(line[0])

    prot_gene = set(prot_gene)
    # 23,358 genes in total
    return prot_gene

  def quantile_normalize(self, df_input):
    """ Quantile normalization.
    """
    # https://github.com/ShawnLYU/Quantile_Normalize
    df = df_input.copy()
    # compute rank
    dic = {}
    for col in df:
      dic.update({col : sorted(df[col])})
    sorted_df = pd.DataFrame(dic)
    rank = sorted_df.mean(axis = 1).tolist()
    #sort
    for col in df:
      t = np.searchsorted(np.sort(df[col]), df[col])
      df[col] = [rank[i] for i in t]
    return df


  def build_data(self):
    """ Preprocess data.
    1. Remove unexpressed genes.
    2. Remove non-protein-coding genes.
    3. Transform into log scale.
    4. Quantile normalization.
    5. Clip extreme values and map into [0, 1].

    Returns
    -------
    Update self.df to be the normalized and processed data.
    """

    # step 1. remove genes that are never expressed -> 50,507 rows/genes.
    row_sums = self.df.values.sum(axis=1)
    idx_nonzero_row = [self.df.index[idx] for idx, r in enumerate(row_sums) if r >= 1]
    self.df = self.df.loc[idx_nonzero_row]

    # step 2. remove non-protein-coding genes -> 20,016 rows/genes.
    prot_gene = self.load_prot_gene()
    index_prot_gene = list(prot_gene.intersection(list(self.df.index)))
    self.df = self.df.loc[index_prot_gene]

    # step 3. take log of the gene expressions
    self.df[self.df.columns] = np.log(self.df.values+1)

    # step 4. adjust the gene expressions to have same median expression level
    # by multiplication in raw expression (or by shifting in log space)
    median_express = np.median(self.df.values, axis=0)
    mean_median = median_express.mean()

    self.df[self.df.columns] = self.df.values - (median_express - mean_median)

    # step 5. quantile normalization of gene expression across samples.
    self.df = self.quantile_normalize(self.df)

    # step 6. clip the log expression by the top and bottom 5% expression values.
    # scale the values into [0, 1].
    data = self.df.values
    ary = sorted(data[:,0])
    min_ary = ary[int(len(ary)*0.025)]
    max_ary = ary[-int(len(ary)*0.025)]
    data = np.clip(data, min_ary, max_ary)
    data = (data - min_ary)/(max_ary-min_ary)
    self.df[self.df.columns] = data

    # Note: somehow the mapping of 11 pairs of ENSG maps to the same gene names,
    # possibly due to the errors in Python mygene package.
    # We just keep it here since they just consist of a minor part of the
    # whole data.
    # ['GAGE12B', 'LOC102723655', 'PI4K2B', 'ARHGEF18', 'HOXD4', 'NPFF',
    # 'MATR3', 'TMSB15B', 'PINX1', 'IRS4', 'MPV17L']
    self.df = pd.DataFrame(
        data=self.df.values,
        index=[self.ensg2gene.get(ensg, "no_match") for ensg in self.df.index],
        columns=self.df.columns)


  def load_top_gene_data(self, top_k=3000):
    """
    Returns
    -------
    A top_k x n_samples dataframe, in the descending order of gene expression
    variance across samples.
    """

    # Process the variance of genes:
    var_genes = self.df.var(axis=1).values
    genes = list(self.df.index)

    ary_index = sorted(range(len(var_genes)), key=var_genes.__getitem__)
    ary_index = ary_index[::-1]

    sorted_top_genes = [ genes[idx] for idx in ary_index]

    return self.df.loc[sorted_top_genes[:top_k]]


  def parse_hsa(self, hsa, path):
    """ Parse the cancer pathway in a specific KEGG file.

    Returns
    -------
    pathway: str
      pathway name.
    list_genes: list of str
      all the genes in the pathway.
    """
    flag = False
    list_genes = []
    with open(path+hsa+'.txt', 'r') as f:
      for idx, line in enumerate(f):
        line = line.strip().split()
        if line[0] != 'NAME':
          continue
        else:
          pathway = " ".join(line[1:-4])
          break
      for idx, line in enumerate(f):
        line = line.strip().split()
        if not flag:
          if line[0] != 'GENE':
            continue
          else:
            g = line[2][:-1]
            if g in self.df.index:
              list_genes.append(g)
            flag = True
        else:
          if (line[0] != 'COMPOUND') and (line[0] != 'REFERENCE'):
            g = line[1][:-1]
            if g in self.df.index:
              list_genes.append(g)
          else:
            break

    return pathway, list_genes


  def get_kegg_list(self, path='data/kegg/hsas/'):
    """ Get the KEGG cancer-related pathways.

    Returns
    -------
    list_genes: list of list of str
      each element corresponds to the genes in a KEGG cancer pathway.
    list_funcs: list of str
      each element corresponds to a KEGG cancer pathway.
    """
    list_genes, list_funcs = [['RET']], ['RET']
    for file_name in os.listdir(path):
      if file_name != '.DS_Store':
        hsa = file_name[:-4]
        pathway, list_gene = self.parse_hsa(hsa, path)
        list_funcs.append(pathway)
        list_genes.append(list_gene)

    return list_genes, list_funcs


  def get_david_list(self, path='data/david/prot_gene_list_3000_cluster.txt'):
    """ Load gene function modules.
    Each module contains a list of genes and related functions.
    list_genes: list of list of genes
    list_funcs: list of list of functions
    """

    list_genes = []
    genes = []
    list_funcs = []
    funcs = []

    enrich = 0
    idx = 0
    with open(path,'r') as f:
      for line in f:
        idx += 1
        line = line.strip('\n')
        if line == '':
          list_funcs.append("; ".join(funcs))
          list_genes.append([self.ensg2gene.get(g, "no_match")for g in list(set(genes))])
          funcs = []
          genes = []
        elif line.startswith('Annotation Cluster'):
          enrich = float(line.split('\t')[1].split(': ')[1])
          if enrich <= 1.0:
            break
        elif line.startswith('Category'):
          pass
        else:
          line = line.split('\t')
          funcs.append(line[0]+' | '+line[1])
          genes = genes+line[5].split(', ')

    return list_genes, list_funcs


  def load_modu_data(self, mode='zscore'):
    """ Load z-scored gene module dataframe.

    Returns
    -------
    df_modu: matrix of z-scored data. Each column a sample, each row gene module.
      First len_kegg rows are kegg cancer-related pathways, and the remaining rows
      are compressed gene module from DAVID.
    """

    assert mode in ['zscore', 'mean']

    list_genes_kegg, list_funcs_kegg = self.get_kegg_list()
    list_genes_david, list_funcs_david = self.get_david_list()
    list_genes = list_genes_kegg + list_genes_david
    list_funcs = list_funcs_kegg + list_funcs_david
    self.len_kegg = len(list_genes_kegg)

    df_modu = np.zeros((len(list_funcs), self.df.shape[1]))

    for idx, genes in enumerate(list_genes):
      df_modu[idx, :] = self.df.loc[genes].mean()
    if mode == 'zscore':
      df_modu = scipy.stats.mstats.zscore(df_modu, axis=1)

    self.df_modu = pd.DataFrame(
        data=df_modu,
        index=list_funcs,
        columns=self.df.columns)

    return self.df_modu, self.len_kegg


