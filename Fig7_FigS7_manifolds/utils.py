
from matplotlib.colors import LinearSegmentedColormap, colorConverter
import matplotlib

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist, squareform
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LogisticRegression

matplotlib.rcParams["axes.titlesize"] = 8
matplotlib.rcParams["axes.labelsize"] = 8
matplotlib.rcParams["xtick.major.size"] = 2
matplotlib.rcParams["ytick.major.size"] = 2
matplotlib.rcParams["xtick.major.pad"] = 2
matplotlib.rcParams["ytick.major.pad"] = 2
matplotlib.rcParams["xtick.labelsize"] = 6
matplotlib.rcParams["ytick.labelsize"] = 6
matplotlib.rcParams["savefig.transparent"] = True


def simple_cmap(*colors, name='none'):
    """Create a colormap from a sequence of rgb values.
    cmap = simple_cmap((1,1,1), (1,0,0)) # white to red colormap
    cmap = simple_cmap('w', 'r')         # white to red colormap
    """

    # check inputs
    n_colors = len(colors)
    if n_colors <= 1:
        raise ValueError('Must specify at least two colors')

    # make sure colors are specified as rgb
    colors = [colorConverter.to_rgb(c) for c in colors]

    # set up colormap
    r, g, b = colors[0]
    cdict = {'red': [(0.0, r, r)], 'green': [(0.0, g, g)], 'blue': [(0.0, b, b)]}
    for i, (r, g, b) in enumerate(colors[1:]):
        idx = (i+1) / (n_colors-1)
        cdict['red'].append((idx, r, r))
        cdict['green'].append((idx, g, g))
        cdict['blue'].append((idx, b, b))

    return LinearSegmentedColormap(name, {k: tuple(v) for k, v in cdict.items()})


CMAP = simple_cmap("r", "b")
circ_c = np.cos(np.linspace(0, np.pi * 2, 80))

def load_mouse(mouse, env_id=None, mean_center=True):

    # Handle two track mouse.
    mousename = mouse.split("_")[0]
    if mousename in ("Seoul", "Degu", "Inchon", "Busan", "Ulsan"):
        if env_id is None:
            raise ValueError("Must specifying env_id for two track mice.")
        else:
            return load_2track_mouse(mouse, env_id, mean_center=mean_center)

    # Load single track mouse.
    X = np.load("../data/{}_MEC_FRtensor.npy".format(mouse))
    cell_idx_file = "../data/{}_MEC_idx.npy".format(mouse)
    if os.path.exists(cell_idx_file):
        cell_idx = np.load(cell_idx_file)
        X = np.copy(X[:, :, cell_idx])

    return X


def load_2track_mouse(mouse, env_id, mean_center=True):
    """
    Parameters
    ---------
    mouse : str
        Session id.
    env_id : int
        For cue poor, cue == 0. For cue rich, cue == 1.
    mean_center : bool
        If true, subtract mean off of each block and renormalize.
    """

    trial_types_file = "../data/{}_trial_types.npy".format(mouse)
    X = np.load("../data/{}_MEC_FRtensor.npy".format(mouse))
    idx = np.load("../data/{}_trial_types.npy".format(mouse))
    cell_idx_file = "../data/{}_MEC_idx.npy".format(mouse)
    if os.path.exists(cell_idx_file):
        cell_idx = np.load(cell_idx_file)
        X = np.copy(X[:, :, cell_idx])

    if mean_center:

        # Should be an array of ints {0, 1, 2, 3}, designating trial blocks.
        trial_block_id = np.concatenate(([0], np.cumsum(np.diff(idx.astype(float)) != 0)))

        # blocks[0] should hold the trial indices for the first block of trials in specified environment.
        # blocks[1] should hold the trial indices for the second block of trials in specified environment.
        blocks = [(env_id == idx) & (trial_block_id == b) for b in np.unique(trial_block_id[env_id == idx])]

        # subtract off mean firing rate from each neuron in two trial blocks.
        X[blocks[0]] -= np.mean(X[blocks[0]], axis=(0,1))
        X[blocks[1]] -= np.mean(X[blocks[1]], axis=(0,1))

        # Restrict to either cue rich or cue poor trials
        X = X[idx == env_id]

        # renormalized between zero and one
        return (X - np.min(X, axis=-1, keepdims=True)) / (np.max(X, axis=-1, keepdims=True) - np.min(X, axis=-1, keepdims=True))

    else:
        return X[idx == env_id]


def load_manifolds(mouse, env_id=None):
    X = load_mouse(mouse, env_id=env_id)
    M = X.reshape(X.shape[0], -1)
    kmeans = KMeans(n_clusters=2, n_init=50, random_state=1234)
    kmeans.fit(M)
    return kmeans.cluster_centers_.reshape(2, X.shape[1], X.shape[2])


def plot_one_manifold(
        mouse, env_id=None, add_supervised_dimension=False,
        axlim=2, reflect_x=False, reflect_y=False, reflect_z=False):

    m, _ = load_manifolds(mouse, env_id=env_id)
    m -= np.mean(m, axis=0)
    num_posbins, num_neurons = m.shape

    if add_supervised_dimension:

        # Find dimension that maximally separates first and second half of the track.
        lgr = LogisticRegression(C=0.1).fit(m, np.repeat([0, 1], num_posbins//2))
        p = (lgr.coef_ / np.linalg.norm(lgr.coef_)).ravel()

        # Find two more PCs
        pca = PCA(n_components=2)
        x_, z_ = pca.fit_transform(m - (m @ (p[:, None] * p[None, :]))).T
        y_ = m @ p

    else:
        # Find two more PCs
        pca = PCA(n_components=3)
        x_, y_, z_ = pca.fit_transform(m).T

    # Reflect axes if desired
    if reflect_x:
        x_ *= -1
    if reflect_y:
        y_ *= -1
    if reflect_z:
        z_ *= -1

    fig = plt.figure(figsize=(2, 1.4))
    ax = plt.axes([0, 0, .6, 1.2], projection='3d')

    ax.scatter(
        x_, y_, z_, c=circ_c,
        alpha=.9, lw=0, s=13, cmap=CMAP)
    print(x_.size)
    print(circ_c.size)

    ax.plot(x_, y_, np.full(num_posbins, -axlim), color="k", alpha=.1, lw=4)
    ax.set_xlim(-axlim, axlim)
    ax.set_ylim(-axlim, axlim)
    ax.set_zlim(-axlim, axlim)

    ax.plot([-axlim, axlim], [axlim, axlim], [-axlim, -axlim], color="k", alpha=.7)
    ax.plot(
        [axlim, axlim], [-axlim, axlim], [-axlim, -axlim], color="k", alpha=.7,
        dashes=([2, 2] if add_supervised_dimension else []))
    ax.plot([axlim, axlim], [axlim, axlim], [-axlim, axlim], color="k", alpha=.7)

    ax.view_init(azim=135, elev=40)
    ax.axis("off")

    ax2 = plt.axes([.6, .2, .3, .5])
    ax2.imshow(squareform(pdist(m)))
    ax2.set_xticks([0, num_posbins//2, num_posbins])
    ax2.set_yticks([0, num_posbins//2, num_posbins])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xlabel("position", fontsize=6.5, labelpad=0)
    ax2.set_ylabel("position", fontsize=6.5, labelpad=0)
    ax2.set_title("within-manifold\ndistances", fontsize=6.5, pad=3)

    return fig, ax, ax2


def plot_two_manifolds(
        mouse, env_id=None, add_supervised_dimension=False, shuffle=False,
        axlim=2, reflect_x=False, reflect_y=False, reflect_z=False):

    m1, m2 = load_manifolds(mouse, env_id=env_id)
    num_posbins, num_neurons = m1.shape

    if shuffle:
        q1 = np.linalg.qr(np.random.randn(num_neurons, num_neurons))[0]
        q2 = np.linalg.qr(np.random.randn(num_neurons, num_neurons))[0]

        m1 = ((m1 - np.mean(m1, axis=0)) @ q1) + np.mean(m1, axis=0)
        m2 = ((m2 - np.mean(m2, axis=0)) @ q2) + np.mean(m2, axis=0)

    m1m2 = np.row_stack((m1, m2))
    m1m2 -= np.mean(m1m2, axis=0)

    if add_supervised_dimension:
        # Find dimension that maximally separates first and second half of the track.
        lgr = LogisticRegression(C=0.1).fit(m1m2, np.repeat([0, 1], num_posbins))
        p = (lgr.coef_ / np.linalg.norm(lgr.coef_)).ravel()

        # Find two more PCs
        pca = PCA(n_components=2)
        x_, z_ = pca.fit_transform(m1m2 - (m1m2 @ (p[:, None] * p[None, :]))).T
        y_ = m1m2 @ p

    else:
        # Find two more PCs
        pca = PCA(n_components=3)
        x_, y_, z_ = pca.fit_transform(m1m2).T

    # Reflect axes if desired
    if reflect_x:
        x_ *= -1
    if reflect_y:
        y_ *= -1
    if reflect_z:
        z_ *= -1

    if add_supervised_dimension:
        y_ *= 1.5 * np.linalg.norm(x_) / np.linalg.norm(y_)

    fig = plt.figure(figsize=(2, 1.4))
    ax = plt.axes([0, 0, .6, 1.2], projection='3d')

    ax.scatter(
        x_[:num_posbins], y_[:num_posbins], z_[:num_posbins],
        c=circ_c, alpha=1, lw=0, s=13, cmap=CMAP)
    ax.scatter(
        x_[num_posbins:], y_[num_posbins:], z_[num_posbins:],
        c=circ_c, alpha=1, lw=0, s=13, cmap=CMAP)

    ax.plot(
        x_[:num_posbins], y_[:num_posbins], np.full(num_posbins, -axlim),
        color="k", alpha=.1, lw=4)
    ax.plot(
        x_[num_posbins:], y_[num_posbins:], np.full(num_posbins, -axlim),
        color="k", alpha=.1, lw=4)

    ax.set_xlim(-axlim, axlim)
    ax.set_ylim(-axlim, axlim)
    ax.set_zlim(-axlim, axlim)

    ax.plot([-axlim, axlim], [axlim, axlim], [-axlim, -axlim], color="k", alpha=.7)
    ax.plot(
        [axlim, axlim], [-axlim, axlim], [-axlim, -axlim], color="k", alpha=.7,
        dashes=([2, 2] if add_supervised_dimension else []))
    ax.plot([axlim, axlim], [axlim, axlim], [-axlim, axlim], color="k", alpha=.7)

    ax.view_init(azim=135, elev=40)
    ax.axis("off")

    ax2 = plt.axes([.6, .2, .3, .5])
    ax2.imshow(cdist(m1, m2))
    ax2.set_xticks([0, num_posbins//2, num_posbins])
    ax2.set_yticks([0, num_posbins//2, num_posbins])
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    ax2.set_xlabel("position", fontsize=6.5, labelpad=0)
    ax2.set_ylabel("position", fontsize=6.5, labelpad=0)
    ax2.set_title("across-manifold\ndistances", fontsize=6.5, pad=3)

    return fig, ax, ax2
