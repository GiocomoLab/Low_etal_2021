import os
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression

import matplotlib

matplotlib.rcParams["axes.titlesize"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["xtick.major.size"] = 2
matplotlib.rcParams["ytick.major.size"] = 2
matplotlib.rcParams["xtick.major.pad"] = 2
matplotlib.rcParams["ytick.major.pad"] = 2
matplotlib.rcParams["xtick.labelsize"] = 7
matplotlib.rcParams["ytick.labelsize"] = 7
matplotlib.rcParams["savefig.transparent"] = True

POSBINS = 80
MICE = [
    "Pisa_0502_1",
    "Seattle_1006_1",
    "Mumbai_1130_1",
    "Goa_1211_1",
    "Vancouver_1114_1",
    "Goa_1210_1",
    "Seattle_1007_1",
    "Punjab_1217_1",
    "Pisa_0501_1",
    "Seattle_1005_1",
    "Mumbai_1201_1",
    "Vancouver_1118_1",
    "Salvador_1202_1",
    "Mumbai_1129_1",
    "Hanover_0615_2",
    "Goa_1209_1",
    "Punjab_1214_1",
    "Toronto_1115_1",
    "Kerala_1207_1",
    "Toronto_1112_1",
    "Pisa_0430_1",
    "Quebec_1007_1",
    "Toronto_1113_1",
    "Calais_0713_2",
    "Toronto_1114_1",
    "Toronto_1111_1",
    "Toronto_1117_1",
    "Portland_1005_2"
]

CUE_POOR_MICE = [
    'Pisa_0502_1',
    'Pisa_0501_1',
    'Pisa_0430_1',
    'Hanover_0615_2',
    'Calais_0713_2',
]

CUE_RICH_MICE = [
    'Goa_1209_1',
    'Goa_1210_1',
    'Goa_1211_1',
    'Kerala_1207_1',
    'Mumbai_1129_1',
    'Mumbai_1130_1',
    'Mumbai_1201_1',
    'Portland_1005_2',
    'Punjab_1214_1',
    'Punjab_1217_1',
    'Quebec_1007_1',
    'Salvador_1202_1',
    'Seattle_1005_1',
    'Seattle_1006_1',
    'Seattle_1007_1',
    'Toronto_1111_1',
    'Toronto_1112_1',
    'Toronto_1113_1',
    'Toronto_1114_1',
    'Toronto_1115_1',
    'Toronto_1117_1',
    'Vancouver_1114_1',
    'Vancouver_1118_1'
]

# Sanity checks
assert(len(CUE_POOR_MICE) == len(np.unique(CUE_POOR_MICE)))
assert(len(CUE_RICH_MICE) == len(np.unique(CUE_RICH_MICE)))
assert(len(CUE_POOR_MICE + CUE_RICH_MICE) == len(np.unique(CUE_POOR_MICE + CUE_RICH_MICE)))
assert(len(MICE) == len(np.unique(CUE_POOR_MICE + CUE_RICH_MICE)))



def load_mouse(mouse):

    X = np.load("../data/{}_MEC_FRtensor.npy".format(mouse))
    cell_idx_file = "../data/{}_MEC_idx.npy".format(mouse)
    if os.path.exists(cell_idx_file):
        cell_idx = np.load(cell_idx_file)
        X = np.copy(X[:, :, cell_idx])

    return X

def circdist(i, j):
    if i < j:
        return circdist(j, i)
    elif i == j:
        return 0.0
    else:
        return 2 * min(i - j, j - i + POSBINS) / POSBINS

def compute_entanglement(manifold):
    n_bins, n_neurons = manifold.shape
    
    ex_dists = squareform(pdist(manifold, metric="euclidean"))
    ex_dists /= np.mean(ex_dists[np.triu_indices_from(ex_dists, k=1)])

    neighbor_dists = np.full(n_bins, np.nan)
    idx = (np.arange(n_bins - 1), np.arange(1, n_bins))
    neighbor_dists[:-1] = ex_dists[idx]
    neighbor_dists[-1] = ex_dists[0, -1]
    
    in_dists = np.zeros((n_bins, n_bins))
    for i, j in zip(*np.triu_indices_from(in_dists, k=1)):
        assert j > i
        in_dists[i, j] = min(
            np.sum(neighbor_dists[i:j]),
            np.sum(neighbor_dists[j:]) + np.sum(neighbor_dists[:i]),
        )
        in_dists[j, i] = in_dists[i, j]
    
    D = in_dists[np.triu_indices_from(in_dists, k=1)]
    d = ex_dists[np.triu_indices_from(ex_dists, k=1)]
    raw_score = np.max(D / d)
    upper_bound = np.max(D) / np.min(d)
    lower_bound = 1 # np.min(D) / np.max(d)
    assert all((D / d) >= lower_bound)
    assert all((D / d) <= upper_bound)
    return (raw_score - lower_bound) / (upper_bound - lower_bound)


def make_entanglements_panel():
    # Compute entanglements for all mice.
    E = {}
    for mouse in tqdm(MICE):
        X = load_mouse(mouse)
        M = X.reshape(X.shape[0], -1)  # trials x (position * units)
        kmeans = KMeans(n_clusters=2, n_init=50, init="random")
        kmeans.fit(M)
        tuning_curves = kmeans.cluster_centers_.reshape(2, X.shape[1], X.shape[2])
        E[mouse] = [
            np.median(compute_entanglement(tuning_curves[0])),
            np.median(compute_entanglement(tuning_curves[1]))
        ]

    # Separate entanglements for cue rich vs cue poor mice
    cue_poor_entanglements = np.concatenate([E[m] for m in CUE_POOR_MICE])
    cue_rich_entanglements = np.concatenate([E[m] for m in CUE_RICH_MICE])


def make_alignment_panel():
    scales = []
    rmse_raw = []
    rmse_aligned = []
    rmse_shuff_mean = []
    remap_dims = []

    for mouse in tqdm(MICE):
        X = load_mouse(mouse)
        M = X.reshape(X.shape[0], -1)  # trials x (position * units)
        kmeans = KMeans(
            n_clusters=2, n_init=100, init="random", tol=1e-6
        )
        kmeans.fit(M)
        tuning_curves = kmeans.cluster_centers_.reshape(2, X.shape[1], X.shape[2])
        
        remap_dims.append(
            np.mean(tuning_curves[0], axis=0) - np.mean(tuning_curves[1], axis=0)
        )
        
        # 1) Mean-center cluster centroids.
        m1 = tuning_curves[0] - np.mean(tuning_curves[0], axis=0, keepdims=True)
        m2 = tuning_curves[1] - np.mean(tuning_curves[1], axis=0, keepdims=True)
        
        m1_norm = np.linalg.norm(m1)
        m2_norm = np.linalg.norm(m2)
        
        scales.append(
            abs(m1_norm - m2_norm) / np.mean([m1_norm, m2_norm])
        )
        
        m1 /= m1_norm
        m2 /= m2_norm

        # 2) Compute Raw RMSE
        rmse_raw.append(np.sqrt(np.mean((m1 - m2) ** 2)))

        # 3) Compute RMSE after best rotational alignment
        u, _, vt = np.linalg.svd(m1.T @ m2)
        rmse_aligned.append(np.sqrt(np.mean((m1 @ (u @ vt) - m2) ** 2)))

        # 4) Compute Null Distribution of RMSE's by random rotations
        rmse_2 = []
        for _ in range(100):
            Q = np.linalg.qr(np.random.randn(m1.shape[1], m1.shape[1]))[0]
            # error corrected Feb. 2022, this line originally read:
            # Q = np.linalg.qr(np.random.randn(m1.shape[1], m1.shape[1]))
            rmse_2.append(np.sqrt(np.mean((m1 @ Q - m2) ** 2)))

        rmse_shuff_mean.append(np.percentile(rmse_2, 2.5))

    rmse_raw = np.array(rmse_raw)
    rmse_aligned = np.array(rmse_aligned)
    rmse_shuff_mean = np.array(rmse_shuff_mean)

    fig, ax = plt.subplots(1, 1, figsize=(2.1, 1.6))
    print(np.max((rmse_raw - rmse_aligned) / (rmse_shuff_mean - rmse_aligned)))
    ax.hist(
        (rmse_raw - rmse_aligned) / (rmse_shuff_mean - rmse_aligned),
        np.linspace(0.0, 1.0, 30),
        color="gray", lw=1, edgecolor="k")
    ax.set_xlabel("manifold misalignment")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # ax.axvline(1.0, dashes=[2, 2], color="k")
    ax.plot([1, 1], [0, 5.5], dashes=[2, 2], color="k")
    ax.text(1, 5, "shuffle", fontsize=7, horizontalalignment='center')

    ax.set_xlim([0, 1.2])
    ax.set_xticks([0, .2, .4, .6, .8, 1., 1.2])
    ax.set_yticks([0, 3, 6])
    ax.spines["left"].set_bounds(0, 6)
    ax.set_ylabel("count")

    fig.tight_layout()
    fig.savefig("alignment_panel.pdf")

if __name__ == "__main__":
    make_entanglements_panel()
    make_alignment_panel()

