import numpy as np
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import interp1d
from scipy.stats import sem
import scipy.io
from scipy.spatial import distance

from tqdm import trange

def tuning_curve(x, Y, dt, b, smooth=True, l=2, SEM=False):
    '''
    Params
    ------
    x : ndarray
        variable of interest by observation; shape (n_obs, )
        e.g. position for spatial tuning curve
    Y : ndarray
        spikes per observation; shape (n_obs, n_cells)
    dt : int
        time per observation in seconds
    b : int
        bin size
    smooth : bool
        apply gaussian filter to firing rate; optional, default is True
    l : int
        smoothness param for gaussian filter; optional, default is 2
    SEM : bool
        return SEM for FR; optional, default is False

    Returns
    -------
    firing_rate : ndarray
        trial-averaged, binned firing rate for each cell; shape (n_bins, n_cells)
    centers : ndarray
        center of each bin
    spike_sem : ndarray
        if SEM==True, returns the binned SEM for each cell; shape (n_bins, n_cells)
    '''

    edges = np.arange(0, np.max(x) + b, b)
    centers = (edges[:-1] + edges[1:])/2
    b_idx = np.digitize(x, edges)
    if np.max(x) == edges[-1]:
        b_idx[b_idx==np.max(b_idx)] = np.max(b_idx) - 1
    unique_bdx = np.unique(b_idx)
    
    # find FR in each bin
    firing_rate = np.zeros((unique_bdx.shape[0], Y.shape[1]))
    spike_sem = np.zeros((unique_bdx.shape[0], Y.shape[1]))
    for i in range(unique_bdx.shape[0]):
        spike_ct = np.sum(Y[b_idx == unique_bdx[i], :], axis=0)
        occupancy = dt * np.sum(b_idx==unique_bdx[i])
        spike_sem[i, :] = sem(Y[b_idx == unique_bdx[i], :]/dt, axis=0)
        firing_rate[i, :] = spike_ct / occupancy
    if smooth:
        firing_rate = gaussian_filter1d(firing_rate, l, axis=0, mode='wrap')
        spike_sem = gaussian_filter1d(spike_sem, l, axis=0, mode='wrap')
    
    if SEM:
        return firing_rate, centers, spike_sem
    else:
        return firing_rate, centers


def get_coordinates(tip, entry, distances):
    '''
    Convert position along probe to 3D brain coordinates
    in microns, relative to MEC landmarks (see STAR Methods)

    Params:
    ------
    tip : ndarray
        coordinates of probe tip; shape(3)
    entry : ndarray
        coordinates of probe entry into MEC; shape(3)
    distances : ndarray
        distance of each cell from probe tip; shape(n_cells)
        
    Returns:
    -------
    coords : ndarray
        coordinates for each cell; shape(n_cells, 3)
        (ML, AP, DV)
    '''
    probe_vec = (entry - tip) # vector connecting tip coords to entry coords
    l = np.linalg.norm(probe_vec) # length of the probe
    coords = tip[None, :] + (distances[:, None]/l) * probe_vec[None, :]
    
    return(coords)


def similarity(Y):
    '''
    Compute the trial-trial similarity. Compare the position-binned 
    spatial coding across all neurons for each pair of trials.

    Params:
    ------
    Y : ndarray
        normalized firing rates for each cell, position, trial
        shape (n_trials, n_pos_bins, n_cells)

    Returns:
    -------
    sim : ndarray
        trial-trial matrix of correlations, 0-1
        shape (n_trials, n_trials)


    '''
    Y_unwrapped = np.reshape(Y, (Y.shape[0], -1))
    sim_vec = np.abs(distance.pdist(Y_unwrapped, 'correlation')-1)
    sim = distance.squareform(sim_vec)

    return sim


def get_remap_idx(W, MIN_TRIALS=5):
    '''
    Get the index for each trial preceding a remap event

    Params:
    ------
    W : ndarray
        k-means model output, map ID for each trial
        shape (n_trials, n_maps)
    MIN_TRIALS : int, odd
        minimum number of trials that activity must reside in a map before the next remap event
        prevents over-counting if activity bounces back and forth before settling
    '''
    trials = np.arange(0, W.shape[0]-1)
    near_N = MIN_TRIALS//2

    # find all remaps
    remap_idx = np.where(np.abs(np.diff(W[:, 0])))[0]

    # find stable periods meeting trial min
    for i in range(near_N):
        if i == 0:
            near_remaps = np.append(remap_idx, remap_idx+(i+1))
        elif i < near_N:
            near_remaps = np.append(np.append(near_remaps, remap_idx-i), remap_idx+(i+1))
    near_remaps = np.sort(near_remaps)
    stable_idx = np.setdiff1d(trials, near_remaps)

    # keep only remaps at least 5 trials from last remap
    boundary_trials = np.insert(remap_idx, 0, 0)
    remap_idx = np.setdiff1d(remap_idx, remap_idx[np.diff(boundary_trials) < MIN_TRIALS])

    return remap_idx


def map_idx_by_obs(A, W):
    '''
    Get the k-means map for each observation.
    Set "map 1" to be the map with slower running speed.

    Params:
    ------
    A : ndarray
        behavioral variables
    W : ndarray
        k-means model output, map ID for each trial
        shape (n_trials, n_maps)

    Returns:
    -------
    map1_idx : ndarray
        k-means map for each observation: 1 = map 1, 0 = map 2
        shape (n_obs, )
    map1_slower : int
        1 if True, 0 if False

    '''
    # get map indices
    map_idx = W[:, 0].astype(bool)
    trials = A[:, 2]
    map1_idx = np.zeros_like(trials)
    for i, t in enumerate(np.unique(trials)):
        if map_idx[i]:
            map1_idx[trials == t] = 1
    map1_idx = map1_idx.astype(bool)

    # get running speed in each map and see which is slower
    speed = A[:, 1]
    speed_1 = np.nanmean(speed[map1_idx])
    speed_2 = np.nanmean(speed[~map1_idx])
    if speed_2 < speed_1: # swap labels
        map1_idx = ~map1_idx
        map1_slower = 0
    else:
        map1_slower = 1

    return map1_idx, map1_slower
