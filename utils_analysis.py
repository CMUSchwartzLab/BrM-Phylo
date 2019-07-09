""" Functions related to analysis called by `analysis.py`.

"""
from cStringIO import StringIO
from random import shuffle
import numpy as np
import pandas as pd
from collections import Counter
from collections import defaultdict as dd

from scipy import stats
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from statsmodels.stats.multitest import fdrcorrection

import networkx as nx
from skbio import DistanceMatrix
from skbio.tree import nj
from Bio import Phylo

import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Yifeng Tao"


def calculate_msd(distance, set_idx_p, set_idx_m, show=False):
  """ Calculate the MSD between two sets of nodes. It is used in the function `hc_analysis`.

  Parameters
  ----------
  distance: distance = nx.floyd_warshall(G)
  set_idx_p: set of int
    first set of nodes
  set_idx_m: set of int
    second set of nodes

  Returns
  -------
  positive-MSD and ratio of positive-MSD/negative-MSD
  """

  p2p, p2m = [], []
  for src in distance.keys():
    if src in set_idx_p:
      tgt2dist = distance[src]
      for tgt in tgt2dist.keys():
        if tgt in set_idx_p:
          if tgt > src:
            p2p.append(tgt2dist[tgt])
        elif tgt in set_idx_m:
          if tgt > src:
            p2m.append(tgt2dist[tgt])
        else:
          continue
    elif src in set_idx_m:
      tgt2dist = distance[src]
      for tgt in tgt2dist.keys():
        if tgt in set_idx_m:
          if tgt > src:
            p2p.append(tgt2dist[tgt])
        elif tgt in set_idx_p:
          if tgt > src:
            p2m.append(tgt2dist[tgt])
        else:
          continue
    else:
      continue

  msd, nmsd = np.mean([p**2 for p in p2p]), np.mean([p**2 for p in p2m])

  if show:
    print("MSD=%.2f, nMSD=%.2f"%(
        msd, nmsd))

  return msd, msd/nmsd


def hc_analysis(x, samples, feature="Pathway"):
  """ Implement hierarchical clustering of genes/gene modules and samples.

  Parameters
  ----------
  x: matrix
    each row a gene/gene module, each column a sample
  samples: list of str
    list of sample names
  """

  size_label = 18
  colors = {idx:"gray" for idx in range(50000)}
  print("feature: %s"%feature)

  fig = plt.figure(figsize=(10, 8))

  #ylabel
  ax1 = fig.add_axes([0.09,0.1,0.01,0.55])
  Y = linkage(x, method="ward")
  Z1 = dendrogram(Y, orientation="left", link_color_func=lambda k: colors[k], no_plot=True)
  ax1.set_xticks([])
  ax1.set_yticks([])
  ax1.axis("off")

  # xlabel
  # Compute and plot the dendrogram.
  ax2 = fig.add_axes([0.1,0.71,0.6,0.1])
  Y = linkage(x.T, method="ward")
  Z2 = dendrogram(Y, link_color_func=lambda k: colors[k])
  ax2.set_xticks([])
  ax2.set_yticks([])
  ax2.axis("off")

  # Plot distance matrix.
  axmatrix = fig.add_axes([0.1,0.1,0.6,0.55])
  idx1 = Z1["leaves"]
  idx2 = Z2["leaves"]
  tmp = x[idx1,:]
  tmp = tmp[:,idx2]
  im = axmatrix.matshow(1-tmp, aspect="auto", origin="lower", cmap=plt.cm.get_cmap("YlGnBu"))#cmap=pylab.cm.YlGnBu)#bwr

  samples = [samples[idx] for idx in Z2["leaves"]]
  plt.xticks([i+0.0 for i in range(len(samples))], samples, rotation=90)

  plt.ylabel(feature, fontsize=size_label)
  axmatrix.yaxis.set_label_position("right")
  axmatrix.xaxis.set_ticks_position("bottom")
  axmatrix.set_yticks([])

  # Plot the sample types
  axmatrix = fig.add_axes([0.1,0.66,0.6,0.04])

  list_pm = np.zeros((1,44),dtype=float)
  tmp = [(idx+1)%2 for idx in Z2["leaves"]] #1:primary, 0:metastatic
  list_pm[0] = tmp

  im = axmatrix.matshow(list_pm, aspect="auto", origin="lower", cmap=plt.cm.get_cmap("autumn"))

  for idx in range(44-1):
    axmatrix.plot([0.5+idx, 0.5+idx], [-0.5, 0.5], "gray")
  axmatrix.set_xticks([])
  axmatrix.set_yticks([])

  plt.show()

  #fig.savefig("figures/fig10hcpathway.pdf", bbox_inches="tight")
  #fig.savefig("figures/fig11hcgenes.pdf", bbox_inches="tight")

  # Statistical test.
  list_a = Y[:,0]
  list_b = Y[:,1]
  list_c = np.array([idx+x.shape[1] for idx in range(Y.shape[0])])

  n_nodes = 2*x.shape[1]-1

  G=nx.Graph()

  G.add_nodes_from([idx for idx in range(n_nodes)])

  edges = [(int(a), int(c)) for a, c in zip(list_a, list_c)]
  G.add_edges_from(edges)

  edges = [(int(b), int(c)) for b, c in zip(list_b, list_c)]
  G.add_edges_from(edges)

  distance = nx.floyd_warshall(G)

  idx_p = [idx for idx in range(44) if idx % 2 == 0]
  idx_m = [idx for idx in range(44) if idx % 2 == 1]

  set_idx_p = set(idx_p)
  set_idx_m = set(idx_m)

  msd, rmsd = calculate_msd(distance, set_idx_p, set_idx_m, show=True)

  list_rand_msd, list_rand_rmsd = [], []

  for _ in range(1000):
    list_pm = range(44)
    shuffle(list_pm)

    idx_p = list_pm[0:22]
    idx_m = list_pm[22:44]

    set_idx_p = set(idx_p)
    set_idx_m = set(idx_m)

    rand_msd, rand_rmsd = calculate_msd(distance, set_idx_p, set_idx_m)
    list_rand_msd.append(rand_msd)
    list_rand_rmsd.append(rand_rmsd)

  zmsd = (msd-np.mean(list_rand_msd))/np.std(list_rand_msd)
  zrmsd = (rmsd-np.mean(list_rand_rmsd))/np.std(list_rand_rmsd)

  #p_values = scipy.stats.norm.sf(16.1004606)
  print("Z_MSD=%.2f, Z_rMSD=%.2f"%(zmsd, zrmsd))


def plot_3d_B(B, data_name="B"):
  """ Plot the first three PCA of bulk data.

  """
  size_label = 15
  size_title = 20
  size_legend = 15
  size_tick = 12
  sns.set_style("white")

  pca = PCA(n_components=3)
  pca.fit(B.T)
  B_pca = pca.transform(B.T)

  idxp = [idx*2 for idx in range(B.shape[1]/2)]
  idxm = [idx*2+1 for idx in range(B.shape[1]/2)]

  fig = plt.figure()
  ax = plt.axes(projection="3d")
  for idx in range(len(idxp)):
    ax.plot3D(
        [B_pca[idxp[idx],0],B_pca[idxm[idx],0]],
        [B_pca[idxp[idx],1],B_pca[idxm[idx],1]],
        [B_pca[idxp[idx],2],B_pca[idxm[idx],2]],
        "gray",alpha=0.5)

  ax.plot3D(B_pca[idxp,0], B_pca[idxp,1], B_pca[idxp,2], "2",label="Primary", markersize=10)
  ax.plot3D(B_pca[idxm,0], B_pca[idxm,1], B_pca[idxm,2],"1",label="Metastatic", markersize=10)

  ax.legend(fancybox=True, framealpha=0.5, prop={"size":size_legend})

  plt.xlabel("PCA 1", fontsize=size_label)
  plt.ylabel("PCA 2", fontsize=size_label)
  ax.set_zlabel("PCA 3", fontsize=size_label)
  plt.tick_params(labelsize=size_tick)
  plt.title(data_name, fontsize=size_title)
  plt.xlim([-15,10])
  plt.ylim([0,15])
  ax.set_zlim(-8,4)
  ##fig.savefig("figures/fig12pcapathway.pdf", bbox_inches="tight")


def get_ary_pm_comp(F, threshold=0.05):
  idx_p = [idx*2+0 for idx in range(F.shape[1]/2)]
  idx_m = [idx*2+1 for idx in range(F.shape[1]/2)]
  is_p = F[:,idx_p].mean(axis=1) - F[:,idx_m].mean(axis=1) > threshold
  is_m = F[:,idx_m].mean(axis=1) - F[:,idx_p].mean(axis=1) > threshold
  return is_p, is_m


def get_labels_comp(F, is_p, is_m):
  """ Get labels of each component based on their relative portions.

  """
  labels = ["C"+str(idx+1)+"|P" if is_p[idx]
    else "C"+str(idx+1)+"|M" if is_m[idx]
    else "C"+str(idx+1) for idx in range(F.shape[0])]
  return labels


def differential_pathway(df_modu, len_kegg, pval_threshold = 0.01):
  """ Find the differentially expressed pathways between primary and metastatic
  samples.

  Parameters
  ----------
  df_modu: matrix
    each row a gene module, each column a sample
  len_kegg: int
    number of KEGG cancer pathways
  pval_threshold: float
    p-value above it will be filtered
  """

  list_funcs = list(df_modu.index)[0:len_kegg]
  B = df_modu.values[0:len_kegg]

  idx_p = [idx*2+0 for idx in range(B.shape[1]/2)]
  idx_m = [idx*2+1 for idx in range(B.shape[1]/2)]

  _, pvalues = stats.ttest_ind(B[:,idx_p], B[:,idx_m], axis=1)
  # bonferroni test
  _, pvalues_bon = fdrcorrection(pvalues)
  print( str(sum(pvalues_bon < pval_threshold))+"/"+str(len(pvalues_bon))+" modules selected" )

  idx_sel = sorted(range(len(pvalues_bon)), key=pvalues_bon.__getitem__)[0:sum(pvalues_bon < pval_threshold)]
  list_funcs_sel = [list_funcs[idx] for idx in idx_sel]
  list_pvalues_bon_sel = [pvalues_bon[idx] for idx in idx_sel]
  B_sel = B[idx_sel]
  is_metastasis = B_sel[:,idx_p].mean(axis=1) < B_sel[:,idx_m].mean(axis=1)

  # save results
  with open("data/function/function.txt", "w") as f:
    for idx, funcs in enumerate(list_funcs_sel):
      if is_metastasis[idx]:
        f.write("Metastasis Funcion "+str(idx)+"; pvalue="+"%.2e"%(list_pvalues_bon_sel[idx])+"; pathway = "+funcs+"\n")
        print("Metastatic & %.2e & %s\\\\"%(list_pvalues_bon_sel[idx], funcs))
      else:
        f.write("Primary Funcion "+str(idx)+"; pvalue="+"%.2e"%(list_pvalues_bon_sel[idx])+"; pathway = "+funcs+"\n")
        print("Primary & %.2e & %s \\\\"%(list_pvalues_bon_sel[idx], funcs))


def component_portion(F, plot_mode=True):
  """ Plot the change of component portions from primary to metastatic samples.

  Returns
  -------
  index of component that is most enriched in primary samples.
  """

  size_label = 18
  size_legend = 18
  size_tick = 18

  idx_p = [idx*2+0 for idx in range(F.shape[1]/2)]
  idx_m = [idx*2+1 for idx in range(F.shape[1]/2)]
  M_p = F[:,idx_p]
  M_m = F[:,idx_m]

  if plot_mode:
    list_cmp_p = ["C"+str(idx+1) for idx in range(M_p.shape[0]) for idy in range(M_p.shape[1])]
    list_cmp_m = ["C"+str(idx+1) for idx in range(M_m.shape[0]) for idy in range(M_m.shape[1])]

    list_val_p = [M_p[idx,idy] for idx in range(M_p.shape[0]) for idy in range(M_p.shape[1])]
    list_val_m = [M_m[idx,idy] for idx in range(M_m.shape[0]) for idy in range(M_m.shape[1])]

    list_typ = ["Primary"] * M_p.size + ["Metastatic"] * M_m.size
    list_cmp = list_cmp_p+list_cmp_m
    list_val = list_val_p+list_val_m

    df_clone = pd.DataFrame(data=np.array([list_typ,list_cmp,list_val]).T,
                            columns=["Type","Component","Portion"])
    df_clone["Portion"] = pd.to_numeric(df_clone["Portion"])

    sns.set_style("darkgrid")
    fig = plt.figure(figsize=(6,4))
    ax = sns.boxplot(x="Component", y="Portion", hue="Type", data=df_clone)

    for idx in range(1, F.shape[0]+1):
      y_p = list(
          df_clone["Portion"][(df_clone["Component"] == "C"+str(idx)) & (df_clone["Type"] == "Primary")]
          )

      y_m = list(
          df_clone["Portion"][(df_clone["Component"] == "C"+str(idx)) & (df_clone["Type"] == "Metastatic")]
          )
      # Add some random "jitter" to the x-axis
      x_p = np.random.normal(idx-1-0.2, 0.025, size=len(y_p))#0.025
      x_m = np.random.normal(idx-1+0.2, 0.025, size=len(y_m))#0.025

      for idx_p in range(len(y_p)):
        plt.plot([x_p[idx_p], x_m[idx_p]], [y_p[idx_p], y_m[idx_p]], "k-", alpha=0.2)
      for idx_p in range(len(y_p)):
        plt.plot(x_p[idx_p], y_p[idx_p], "b.", alpha=0.5)
      for idx_p in range(len(y_p)):
        plt.plot(x_m[idx_p], y_m[idx_p], "r.", alpha=0.5)

    ax.tick_params(axis="both", which="major", labelsize=size_tick)
    ax.xaxis.label.set_fontsize(size_label)
    ax.yaxis.label.set_fontsize(size_label)
    plt.legend(loc=2,fancybox=True, framealpha=0.5,prop={"size":size_legend-4})
    plt.show()
    ##fig.savefig("figures/fig3portion.pdf", bbox_inches="tight")
    sns.set_style("white")

  plt.show()

  return np.argmax(np.median(M_p,axis=1))


def classify_patients(F, threshold_0=1e-2):
  """ Classify components of patients.

  Parameters
  ----------
  threshold_0: float
    any value in F below this value is taken as 0.

  Returns
  -------
  list_pattern: list of list of int
    list of patterns of existed componets.
  """

  counts = []
  # component of primary clone
  F01 = np.array([["1" if f >= threshold_0 else "0" for f in line] for line in F])
  F01int = np.array([[1 if f >= threshold_0 else 0 for f in line] for line in F])
  for idxp in range(0, F01.shape[1], 2):
    p = F01[:,idxp]
    m = F01[:,idxp+1]
    p_int = F01int[:,idxp]
    m_int = F01int[:,idxp+1]
    strings = "".join(p)+"|"+"".join(m)
    counts.append("".join(["1" if i > 0 else "0" for i in (p_int+m_int)]))

    if True:
      strings += ": "+" ".join(["%.2f"%idx for idx in F[:,idxp]])+" | "+" ".join(["%.2f"%idx for idx in F[:,idxp+1]])
    print(strings)
  list_pattern = []
  counts = Counter(counts)
  for pattern in counts:
    print("%s: %d"%(pattern, counts[pattern]))
    list_pattern.append([idx for idx, v in enumerate(pattern) if v == "1"])

  return list_pattern


def pattern2case(pattern):
  """ From pattern to case number.
  Note: this is hard-coded and need to be revised once the data changes.

  """
  pattern = pattern.split("|")
  p, m = pattern[0], pattern[1]
  pattern = "".join(["0" if (p[idx] == "0" and m[idx] == "0") else "1" for idx in range(len(p))])
  if pattern == "11111":
    c = 1
  elif pattern == "01111":
    c = 2
  elif pattern == "10111":
    c = 3
  else:
    c = 4
  return c


def plot_patients_F(F, threshold_0=1e-2):
  """ Plot the classified patterns of patients.

  """
  size_label = 18
  size_title = 20
  size_legend = 18
  size_tick = 18

  n_clone = F.shape[0]
  pattern2df = dd(list)
  F01 = np.array([["1" if f >= threshold_0 else "0" for f in line] for line in F])
  for idxp in range(0, F01.shape[1], 2):
    pstr, mstr = F01[:,idxp],F01[:,idxp+1]
    pflt, mflt = F[:,idxp],F[:,idxp+1]
    strings = "".join(pstr)+"|"+"".join(mstr)
    pattern2df[strings].append([
        list(pflt)+list(mflt),
        ["Primary"]*n_clone+["Metastatic"]*n_clone,
        ["C"+str(idx+1) for idx in range(n_clone)]*2 ])
  for pattern in pattern2df.keys():
    data = [[],[],[]]
    df = pattern2df[pattern]
    for line in df:
      data[0] += line[0]
      data[1] += line[1]
      data[2] += line[2]
    data = pd.DataFrame(data=np.array(data).T,columns=["Portion","Type","Component"])
    data["Portion"] = pd.to_numeric(data["Portion"])
    pattern2df[pattern] = data

  case2pattern = dd(list)
  for pattern in pattern2df.keys():
    c = pattern2case(pattern)
    case2pattern[c].append(pattern)

  pp2alphabet = {0:"a",1:"b", 2:"c", 3:"d", 4:"e", 5:"f",
                 6:"g",7:"h",8:"i",9:"j",10:"k"}
  fig = plt.figure(figsize=(14, 9))
  idx_pattern = 0
  for idx_case in range(len(case2pattern.keys())):
    c = idx_case + 1
    patterns = case2pattern[c]
    for idx_pp, pattern in enumerate(patterns):
      idx_pattern += 1
      ax = plt.subplot(3, 4, idx_pattern)
      data = pattern2df[pattern]
      ax = sns.barplot(x="Component", y="Portion", hue="Type", data=data)
      plt.ylim([0, 0.6])
      plt.title("Case %s%s (#patient:%2d)"%(str(c), pp2alphabet[idx_pp], data.shape[0]/10), fontsize=size_title-4)
      ax.tick_params(axis="both", which="major", labelsize=size_tick)
      ax.xaxis.label.set_fontsize(size_label)
      ax.yaxis.label.set_fontsize(size_label)
      plt.legend(loc=1,fancybox=True, framealpha=0.5,prop={"size":size_legend-4})
      if idx_pattern != 1:
        ax.get_legend().remove()
      if idx_pattern <= 8:
        # Just hide the label but not ticks
        plt.xlabel("")
      if idx_pattern % 4 != 1:
        ax.get_yaxis().set_visible(False)
  plt.subplots_adjust(wspace=0.1, hspace=0.5)
  ##fig.savefig("figures/fig13instances.pdf", bbox_inches="tight")


def get_min_max_weight_edges(G):
  """ Return the minimum and maximum weight in the graph G.

  """
  min_weight = 1e10
  max_weight = 0
  for edge in G.edges(data=True):
    min_weight = min(1.0/edge[2]["weight"], min_weight)
    max_weight = max(1.0/edge[2]["weight"], max_weight)
  return min_weight, max_weight


def iter_func(root_name, root, set_traverse, list_funcs, G, strings,
              plot_nodes, cur_pos, xgrain, min_weight, max_weight):
  """ Recursive function to traverse over the phylogeny and generate positions
  of each node.

  """
  set_traverse.append(root)
  nbs = G.neighbors(root)
  nbs = G[root]

  plot_nodes.append(cur_pos)
  xgrain = xgrain/2.0

  flag_pn = -1
  for nb in nbs.keys():
    if nb in set_traverse:
      continue

    next_pos = [0, 0, 0]
    if root.name == root_name:
      next_pos[0] = cur_pos[0]
    else:
      next_pos[0] = cur_pos[0] + xgrain*flag_pn*( 0.8+0.2*(nbs[nb]["weight"]-1.0/max_weight)/(1.0/min_weight-1.0/max_weight) ) #* (nbs[nb]["weight"]-1.0/max_weight)/(1.0/min_weight-1.0/max_weight)
    next_pos[1] = cur_pos[1] + 3.0*(nbs[nb]["weight"]-1.0/max_weight)/(1.0/min_weight-1.0/max_weight)
    next_pos[2] = nb.name

    flag_pn = flag_pn*(-1)

    strings.append([root, nb])
    set_traverse, strings, plot_nodes = iter_func(root_name, nb, set_traverse, list_funcs, G, strings, plot_nodes, next_pos, xgrain, min_weight, max_weight)

  return set_traverse, strings, plot_nodes



def component_func(B, C, F, list_funcs, len_kegg, threshold=0.05):
  """ Plot the functions of cancer pathways in each component.

  Parameters
  ----------
  B: matrix
    gene module of each sample, each row a gene module, each column a sample.
  C: matrix
    gene module values of all components, each row a gene module, each column a component.
  F: matrix
    portions of each component in each sample, each row a component, each column a sample.
  list_funcs: list of str
    functions of each gene module.
  len_kegg: int
    number of KEGG cancer related pathways.
  threshold: float
    threshold to define whether a component is primary or metastatic.
  """

  size_label = 18
  size_tick = 18
  sns.set_style("darkgrid")

  list_funcs = list_funcs[0:len_kegg]
  B = B[0:len_kegg]
  C = C[0:len_kegg]

  idx_p = [idx*2+0 for idx in range(B.shape[1]/2)]
  idx_m = [idx*2+1 for idx in range(B.shape[1]/2)]
  is_p = F[:,idx_p].mean(axis=1) - F[:,idx_m].mean(axis=1) > threshold
  is_m = F[:,idx_m].mean(axis=1) - F[:,idx_p].mean(axis=1) > threshold
  is_p = [idx for idx, val in enumerate(is_p) if val]
  is_m = [idx for idx, val in enumerate(is_m) if val]
  is_n = [idx for idx in range(C.shape[1]) if (idx not in is_p) and (idx not in is_m)]

  fig = plt.figure(figsize=(6, 6))

  for idx_c in range(C.shape[1]):
    clone_num = idx_c +1
    if idx_c in is_p:
      color="green"
      label = "C"+str(clone_num)+"|P"
    elif idx_c in is_m:
      color="red"
      label = "C"+str(clone_num)+"|M"
    else:
      color = "royalblue"
      label = "C"+str(clone_num)

    if clone_num in [1]:
      marker = "$1$"
    elif clone_num in [2]:
      marker = "$2$"
    elif clone_num in [3]:
      marker = "$3$"
    elif clone_num in [4]:
      marker = "$4$"
    elif clone_num in [5]:
      marker = "$5$"
    elif clone_num in [6]:
      marker = "$6$"
    else:
      print("error")
    plt.plot(C[:,idx_c],range(len_kegg), color=color, marker=marker, markersize=10, linestyle="",label=label, alpha=0.5)

  plt.yticks(range(len_kegg), list_funcs)
  plt.xlabel("Pathway strength", fontsize=size_label)
  plt.legend()
  plt.tick_params(labelsize=size_tick-4)
  plt.show()
  ##fig.savefig("figures/fig7compfunc.pdf", bbox_inches="tight")


def plot_phylo(C_raw, F, list_funcs, len_kegg, comp_p, pattern, threshold=0.05):
  """ Build the phylogeny of components, analyze the pathway and plot them.

  Parameters
  ----------
  C_raw: matrix
    gene module values of all components, each row a gene module, each column a component.
  F: matrix
    portions of each component in each sample, each row a component, each column a sample.
  list_funcs: list of str
    functions of each gene module.
  len_kegg: int
    number of KEGG cancer related pathways.
  comp_p: int
    index of most abundant component in the primary samples.
  pattern: list of int
    components need to considered for constructing the phylogeney.
  threshold: float
    threshold to define whether a component is primary or metastatic.
  """
  assert comp_p in pattern
  is_p, is_m = get_ary_pm_comp(F, threshold=threshold)
  labels = get_labels_comp(F, is_p, is_m)

  labels = [labels[idx] for idx in pattern]
  C = C_raw[:, pattern]
  # build up phylogeny of components
  W = pairwise_distances(C.T)
  # for numerical stability
  W = (W+W.T)/2.0
  dm = DistanceMatrix(W, labels)
  newick_str = nj(dm, result_constructor=str)
  tree = Phylo.read(StringIO(newick_str), "newick")
  tree.ladderize() # Flip branches so deeper clades are displayed at top
  #Phylo.draw(tree)
  #Phylo.draw(tree, branch_labels=lambda c: c.branch_length)

  # initialize the graph:
  # pathway of leaves, name of steiner nodes, branch length of root
  G = Phylo.to_networkx(tree)
  idx = 1
  for node in G.nodes():
    if node.name != None:
      node.pathway = C_raw[:,int(node.name[1])-1]
    else:
      node.name = "S"+str(idx)
      idx += 1
      node.pathway = [0]
    if node.branch_length == None:
      node.branch_length = 0

  dim_path = C.shape[0]

  edges = G.edges(data=True)

  # number of steiner nodes
  n_s = C.shape[1] - 2
  mat_Q = np.zeros((n_s, n_s),dtype=float)
  ary_c = [np.zeros(n_s, dtype=float) for _ in range(dim_path)]

  for edge in edges:
    wt = edge[2]["weight"] # It's length, not weight!
    wt = 1.0/wt

    if (edge[0].name[0] == "S") and (edge[1].name[0] == "S"):
      nodes, nodet = edge[0], edge[1]
      ids, idt = int(nodes.name[1])-1, int(nodet.name[1])-1
      mat_Q[ids, ids] += wt
      mat_Q[ids, idt] -= wt
      mat_Q[idt, idt] += wt
      mat_Q[idt, ids] -= wt
    else:
      nodes, nodec = None, None
      if (edge[0].name[0] == "S") and (edge[1].name[0] == "C"):
        nodes, nodec = edge[0], edge[1]
      elif (edge[0].name[0] == "C") and (edge[1].name[0] == "S"):
        nodes, nodec = edge[1], edge[0]
      else:
        print("error")
      ids, idc = int(nodes.name[1])-1, int(nodec.name[1])-1
      mat_Q[ids, ids] += wt

      for idx_path in range(dim_path):
        ary_c[idx_path][ids] += wt * (nodec.pathway[idx_path])
  #for node in G.nodes():
  #  print(node.name, node.pathway[0])

  s_pathway = []
  for idx_path in range(dim_path):
    tmp = np.linalg.solve(mat_Q, ary_c[idx_path])
    s_pathway.append(tmp)
  # num_pathways x num_steiner_nodes
  S = np.asarray(s_pathway, dtype=float)

  for node in G.nodes():
    if node.name[0] == "S":
      node.pathway = S[:, int(node.name[1])-1]

  min_weight, max_weight = get_min_max_weight_edges(G)

  root_name = [l for l in labels if int(l[1])-1 == comp_p][0]
  for root in G.nodes():
    if root.name == root_name:
      break

  set_traverse = []
  plot_nodes = []
  cur_pos = [0, 0, root.name]
  xgrain = 1.0

  strings = []
  set_traverse, strings, plot_nodes = iter_func(
      root_name, root, set_traverse, list_funcs, G, strings, plot_nodes,
      cur_pos, xgrain, min_weight, max_weight)

  node2pos = {v[2]:[v[0],v[1]] for v in plot_nodes}

  sns.set_style("white")
  fig = plt.figure(figsize=(7,6))
  ax0 = plt.subplot(1,1,1)

  xmin = 0
  xmax = 0
  ymin = 0
  ymax = 0

  min_linewidth=8
  max_linewidth=20

  for edge in G.edges(data=True):
    ax0.plot([node2pos[edge[0].name][0],node2pos[edge[1].name][0]],
             [node2pos[edge[0].name][1],node2pos[edge[1].name][1]],
             "-", color="gray",
             linewidth=min_linewidth+(max_linewidth-min_linewidth)*(1.0/edge[2]["weight"]-min_weight)/max_weight,
             alpha=0.3)

  for node in node2pos.keys():
    pos = node2pos[node]
    if node[0] == "S":
      color = "gray"
    elif node[-1] == "P":
      color = "green"
    elif node[-1] == "M":
      color = "red"
    else:
      color = "royalblue"

    ax0.plot(pos[0], pos[1], "o", markersize=50, color=color, alpha=0.5)#markeredgecolor="k",
    ax0.annotate(
        s=node,
        xy=(pos[0], pos[1]),
        ha="center",
        va="center",
        size=22,
        fontweight="bold",
        )
    xmin = min(xmin, pos[0])
    xmax = max(xmax, pos[0])
    ymin = min(ymin, pos[1])
    ymax = max(ymax, pos[1])

  delta = 0.7
  xratio = (xmax-xmin)/(ymax-ymin)*delta
  plt.xlim(xmin-delta*xratio, xmax+delta*xratio)
  plt.ylim(ymin-delta, ymax+delta)

#  ax0.annotate("Progression", xy=(xmin-delta*xratio, ymax), xytext=(xmin-delta*xratio,ymin),
#               ha="center",
#               arrowprops=dict(facecolor="black",alpha=0.7),
#               size=22,
#               fontweight="bold",
#               rotation=0,
#               )

  plt.gca().invert_yaxis()
  ax0.spines["right"].set_visible(False)
  ax0.spines["top"].set_visible(False)
  ax0.spines["left"].set_visible(False)
  ax0.spines["bottom"].set_visible(False)
  ax0.get_xaxis().set_ticks([])
  ax0.get_yaxis().set_ticks([])
  plt.show()
  ##fig.savefig("figures/fig8phylo3.pdf", bbox_inches="tight")

  for tmp in strings:
    node_src, node_tgt = tmp[0], tmp[1]
    print("\colrule")
    print("$%s \\rightarrow %s$"%(node_src.name, node_tgt.name))
    delta_pathway = node_tgt.pathway - node_src.pathway
    delta_pathway = delta_pathway[0:len_kegg]

    idx_sel = sorted(range(len(delta_pathway)), key=delta_pathway.__getitem__)

    threshold = 1.0
    list_pos, list_neg = [], []
    max_ct = 5
    for ct, idx in enumerate(idx_sel[::-1]):
      if delta_pathway[idx] > threshold and ct < max_ct:
        list_pos.append([delta_pathway[idx], list_funcs[idx] ])
    for ct, idx in enumerate(idx_sel):
      if delta_pathway[idx] < -threshold and ct < max_ct:
        list_neg.append([delta_pathway[idx], list_funcs[idx] ])

    for idx in range(max(len(list_pos), len(list_neg))):
      if idx+1 <= len(list_pos):
        fun = list_pos[idx][1]
        if fun in ["RET", "PI3K-Akt signaling pathway", "ErbB signaling pathway"]:
          fun = "\\textbf{"+fun+"}"
        print("& $+%.2f$ & "%(list_pos[idx][0])+fun)
      else:
        print("& & ")
      if idx+1 <= len(list_neg):
        fun = list_neg[idx][1]
        if fun in ["RET", "PI3K-Akt signaling pathway", "ErbB signaling pathway"]:
          fun = "\\textbf{"+fun+"}"
        print("& $%.2f$ & "%(list_neg[idx][0])+fun)
      else:
        print("& & ")
      print("\\\\")
    if max(len(list_pos), len(list_neg)) == 0:
      print("& $<1.0$ & $\emptyset$ & $<1.0$ & $\emptyset$ \\\\")


