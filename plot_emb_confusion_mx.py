import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from IPython import display



device = 'cuda' if torch.cuda.is_available() else 'cpu'

"""


"""
#torch.set_default_device(device)


def prim_tree(adj_matrix):
    infty = torch.max(adj_matrix) + 10
    ones = torch.ones(adj_matrix.shape[0], device=device)
    dst = ones * infty
    visited = torch.zeros(adj_matrix.shape[0], dtype=bool, device=device)
    ancestor = -torch.ones(adj_matrix.shape[0], dtype=int, device=device)


    v, s = 0, torch.tensor(0.0)
    s = s.to(device)
    for i in range(adj_matrix.shape[0] - 1):
        visited[v] = 1
        ancestor[dst > adj_matrix[v]] = v
        dst = torch.minimum(dst, adj_matrix[v])
        dst[visited] = infty
        
        v = torch.argmin(dst)
        
        s += adj_matrix[v][ancestor[v]]
    return s.item()

def rtd_r(a, b):
    with torch.no_grad():
        r1 = torch.cdist(a, a)
        r2 = torch.cdist(b, b)
        r12 = torch.minimum(r1, r2)  # probably should add normaization by distance quantile ?
    s1 = prim_tree(r1)
    s2 = prim_tree(r2)
    s12 = prim_tree(r12)
    return 0.5 * (s1 + s2 - 2 * s12) / a.shape[0]  # maybe remove averaging  / a.shape[0]   ?

# Approximated version of RTD for faster computation
# 



def get_average_similarity(emb1, emb2):
    return rtd_r(emb1, emb2)


def plot_emb_confusion(layer_embeddings, base_layer_numbers, save_path=None):
    figsize = (6, 6)
    fig, ax = plt.subplots(figsize=figsize, dpi=400)
    num_l = len(base_layer_numbers)
    heatmap = np.zeros((num_l, num_l))
    for i in range(num_l):
        for j in range(num_l):
            if j >= i:
                sim = get_average_similarity(layer_embeddings[i][i], layer_embeddings[i][j])
            else:
                sim = get_average_similarity(layer_embeddings[j][j], layer_embeddings[j][i])
            heatmap[i,j] = sim 

    
    s = sns.heatmap(heatmap.T, annot=True, cmap='YlGnBu', \
                    xticklabels=base_layer_numbers, yticklabels=base_layer_numbers,
                    ax=ax, fmt='.2f', vmin=0.4, vmax=1.0,
                    cbar_kws={"orientation": "horizontal", "pad":0.02},
                    annot_kws={"size": 12})
    s.set(xlabel='Layer num', ylabel='Layer num')


    ax.set_title(f"Similarity of layer embeddings", fontsize=16)
    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')
    ax.tick_params(axis='y', rotation=0)
    hdisplay = display.display("", display_id=True)
    hdisplay.update(fig)
    if save_path:
        fig.savefig(save_path)