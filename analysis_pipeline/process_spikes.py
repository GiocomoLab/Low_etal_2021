import numpy as np
import matplotlib.pyplot as plt
from lvl.factor_models import KMeans

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


'''
Population Analyses:
Functions to explore remapping at the population level.
'''
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


def map_similarity(sim, map_idx):
    '''
    calculate the average withing vs. across map similarity
    count each trial pair once and exclude the diagonal (i.e. autocorrelation) 
    by indexing into the upper triangle of each within/across maps matrix

    Params:
    ------
    sim : ndarray
        trial-by-trial similarity matrix
    map_idx : bool
        True if map 1, False if map 2; shape (n_trials, )

    Returns:
    -------
    avg_within : float
        average within map trial-trial correlation
    avg_across : float
        average across map trial-trial correlation
    '''
    # get the within map similarity
    sim_0 = sim[map_idx, :]
    sim_0 = sim_0[:, map_idx]
    sim_1 = sim[~map_idx, :]
    sim_1 = sim_1[:, ~map_idx]
    within = np.append(sim_0[np.triu_indices(n=sim_0.shape[0], k=1)], 
                       sim_1[np.triu_indices(n=sim_1.shape[0], k=1)])
    avg_within = np.mean(within)

    # get the across map similarity
    sim_across = sim[map_idx, :]
    sim_across = sim_across[:, ~map_idx]
    across = sim_across[np.triu_indices(n=sim_across.shape[0], k=1, m=sim_across.shape[1])]
    avg_across = np.mean(across)

    return avg_within, avg_across

''' K-means '''
from lvl.factor_models import KMeans as lvl_kmeans
from lvl.factor_models import NMF as lvl_soft_kmeans

def fit_kmeans(Y, **kwargs):
    '''
    Params:
    ------
    Y : ndarray
        normalized firing rates for each cell, position, trial
        shape (n_trials, n_pos_bins, n_cells)
    kwargs : passed to lvl_kmeans

    Returns:
    -------
    kmeans_dict : dict
        contains the k-means model outputs:
        W : map ID for each trial; shape (n_trials, n_maps)
        H : tuning curve estimates; shape (n_maps, n_cells*n_pos_bins)
        Y_hat : predicted firing rates; shape (n_trials, n_pos_bins, n_cells)
    '''
    # reshape Y into (n_trials, n_cells*n_pos_bins)
    Y = Y.transpose(0, 2, 1)
    Y_unwrapped = np.reshape(Y, (Y.shape[0], -1))
    
    # fit model
    model_kmeans = KMeans(**kwargs)
    model_kmeans.fit(Y_unwrapped)

    # store params
    kmeans_dict = {}
    W, H = model_kmeans.factors
    Y_hat = model_kmeans.predict()
    kmeans_dict['W'] = W
    kmeans_dict['H'] = H
    kmeans_dict['Y_hat'] = Y_hat
    
    return kmeans_dict


'''
Single Cell Analyses:
Functions the assess the impact of remapping on single cells
'''
def FR_diff(FR_1, FR_2):
    '''
    Compute the absolute percent change in peak firing rate.

    Params:
    ------
    FR_1, FR_2 : ndarray
        trial-averaged, binned firing rate for each cell; shape (n_bins, n_cells)

    Returns:
    -------
    pct_dFR : ndarray
        percent change in peak firing rate; shape (n_cells, )
    '''
    peak_FR_1 = np.max(FR_1, axis=0)
    peak_FR_2 = np.max(FR_2, axis=0)
    pct_dFR = (np.abs(peak_FR_2 - peak_FR_1) / ((peak_FR_2 + peak_FR_1)/2))*100
    return pct_dFR


def spatial_similarity(FR_1, FR_2):
    '''
    Compute the cosine similarity between two sets of spatial tuning curves.

    Params:
    ------
    FR_1, FR_2 : ndarray
        trial-averaged, binned firing rate for each cell; shape (n_bins, n_cells)

    Returns:
    -------
    spatial_sim : ndarray
        cosine similarity; shape (n_cells, )
    '''
    norms_1 = np.linalg.norm(FR_1, axis=0)
    norms_2 = np.linalg.norm(FR_2, axis=0)
    spatial_sim = np.zeros(FR_1.shape[1])
    for i, n1 in enumerate(norms_1):
        n2 = norms_2[i]
        normalized_FR_1 = FR_1[:, i]/n1
        normalized_FR_2 = FR_2[:, i]/n2   
        spatial_sim[i] = normalized_FR_1 @ normalized_FR_2
    return spatial_sim


def spatial_info(posx, B, bin_size, dt):
    '''
    compute spatial information (bits per second) averaged across all trials
    
    Params:
    -------
    posx : ndarray
        position at each observation; shape (n_obs, )
    B : ndarray
        number of spikes per observation; shape (n_obs, n_cells)
    b : int
        bin size (cm)
    dt : int
        time per observation (s)
        
    Returns:
    -------
    SI : ndarray
        spatial information for each cell; shape (n_cells, )
    '''
    # get position bins
    edges = np.arange(0, np.max(posx) + bin_size, bin_size)
    centers = (edges[:-1] + edges[1:])/2
    b_idx = np.digitize(posx, edges)
    if np.max(posx) == edges[-1]:
        b_idx[b_idx==np.max(b_idx)] = np.max(b_idx) - 1
    unique_bdx = np.unique(b_idx)
    
    # get params
    T = dt * posx.shape[0] # total time
    L = np.sum(B, axis=0) / T # overall mean FR
    
    SI = np.zeros(B.shape[1]) # spatial info
    for i, b in enumerate(unique_bdx):
        occupancy = dt * np.sum(b_idx == b)
        t = occupancy / T # occupancy ratio (time in bin/total time)
        spike_ct = np.sum(B[b_idx == b, :], axis=0)
        l = spike_ct / occupancy # mean FR for that bin
        SI += t * l * np.log2((l / L) + 0.0001)
    SI[L == 0] = 0
        
    return SI


def shuffle_SI(posx, B, bin_size, dt, n_repeats=100, max_shift=50):
    '''
    Get shuffled spatial information:
        for each repeat, spikes are shifted in time by a random amount, up to max_shift * dt
        spikes for all cells are shifted together, maintaining relationship between cells
        temporal firing pattern is preserved
    '''
    n_cells = B.shape[1]
    shuffle = np.zeros((n_cells, n_repeats))
    shifts = (np.random.rand(n_repeats, 1)*max_shift).astype(int)
    for i, k in enumerate(shifts):
        shuff_B = np.roll(B.copy(), k, axis=0)
        shuffle[:, i] = spatial_info(posx, shuff_B, bin_size, dt)    
    return shuffle


def change_SI(SI_1, SI_2):
    '''
    Compute the absolute fold change in spatial information for each cell.
    '''
    # remove zeros
    map0 = SI_map0[SI_map0 > 0]
    map1 = SI_map1[SI_map0 > 0]
    
    # get the change in SI
    return np.abs((map0 - map1)/map0)


def percent_spatial(d, THRESH=95):
    '''
    Get the number of cells that are spatial in 
    both maps, one map, or neither.

    Params:
    ------
    d : dict
        data[mouse][session]
    THRESH : int
        percentile of shuffle to be considered significant

    Returns:
    -------
    int, number of cells in each session that are:
    retain : spatial in both maps
    gain : spatial in one map
    neither : not spatial in either map

    total cells : total number of cells in each session
    '''
    # get data
    SI_map0 = d['SI'][0, :].copy()
    SI_map1 = d['SI'][1, :].copy()
    shuff_SI_map0 = d['shuff'][0].copy()
    shuff_SI_map1 = d['shuff'][1].copy()

    # get cells that are significantly spatial in each map
    flat_shuff_SI_map0 = shuff_SI_map0.flatten()
    flat_shuff_SI_map1 = shuff_SI_map1.flatten()        
    sig_0_idx = SI_map0 > np.percentile(flat_shuff_SI_map0, THRESH)
    sig_1_idx = SI_map1 > np.percentile(flat_shuff_SI_map1, THRESH)
    
    # get number of cells
    total_cells = SI_map0.shape[0]
    retain = np.sum([sig_1_idx & sig_0_idx])
    gain = np.sum([sig_1_idx & ~sig_0_idx]) + np.sum([~sig_1_idx & sig_0_idx])
    neither = np.sum([~sig_1_idx & ~sig_0_idx])

    return retain, gain, neither, total_cells

'''
Distance to Cluster:
functions to calculate the distance to k-means cluster along different axes
see STAR Methods for more details
'''
def clu_distance_population(Y, H, map_idx):
    '''
    Calculate the distance between the population activity and 
    the k-means cluster centroids on each trial

    Params:
    ------
    Y : ndarray
        normalized firing rate by 5cm position bins by trial for each cell
        shape (n_trials, n_pos_bins, n_cells)
    H : ndarray
        k-means tuning curve estimates for each cluster/map
        shape (n_maps, n_cell*n_pos_bins)
    map_idx : int
        index for map 1

    Returns:
    -------
    dist : ndarray
        distance to cluster on each trial; shape (n_trials, )
        1 = in map 1 centroid
        -1 = in map 2 centroid
        0 = exactly between the two maps
    '''
    # reshape Y to get a trial x neurons*positions matrix
    Y = Y.transpose(0, 2, 1)
    Y_unwrapped = np.reshape(Y, (Y.shape[0], -1))
    n_trials, n_cells, n_pos = Y.shape

    # get kmeans centroids
    c1 = H[map_idx]
    c2 = H[map_idx-1]
    
    # project everything down to a vector connecting the two centroids
    proj = (c1 - c2) / np.linalg.norm(c1 - c2)
    projc1 = c1 @ proj # cluster 1
    projc2 = c2 @ proj # cluster 2
    projY = Y_unwrapped @ proj # activity on each trial
    
    # get distance to cluster on each trial
    dd = (projY - projc2) / (projc1 - projc2)
    return 2 * (dd - .5) # classify -1 or 1

def clu_distance_cells(Y, H, map_idx, W):
    '''
    Calculate the distance to k-means cluster for each cell on each trial.
    Also computes the log-likelihood that each cell is in each map or the "remap score."

    Params:
    ------
    Y : ndarray
        normalized firing rate by 5cm position bins by trial for each cell
        shape (n_trials, n_pos_bins, n_cells)
    H : ndarray
        k-means tuning curve estimates for each cluster/map
        shape (n_maps, n_cell*n_pos_bins)
    map_idx : int
        index for map 1
    W : ndarray
        k-means cluster label for each trial; shape (n_trials, n_maps)

    Returns:
    -------
    dd_by_cells : ndarray
        distance to cluster for each cell on each trial; shape (n_cells, n_trials)
        1 = in map 1 centroid
        -1 = in map 2 centroid
        0 = exactly between the two maps
    ll_cells : ndarray
        log likelihood that each cell is in each map; shape (n_cells, n_trials)
        0 = at the midpoint between clusters
        1 = in either cluster centroid
    '''
    # reshape and get the dimensions 
    Y = Y.transpose(0, 2, 1)
    n_trials, n_cells, n_pos = Y.shape
    n_maps = H.shape[0]
    H_tens = H.reshape((n_maps, n_cells, n_pos))

    # get each cluster
    c1 = H_tens[map_idx, :, :]
    c2 = H_tens[map_idx-1, :, :]
    
    # find the unit vector in direction connecting c1 and c2 in state space
    proj = (c1 - c2) / np.linalg.norm(c1 - c2, axis=1, keepdims=True)

    # project everything onto the same line
    projc1 = np.sum(c1 * proj, axis=1)[None, :]
    projc2 = np.sum(c2 * proj, axis=1)[None, :]
    projY = np.sum(Y * proj[None, :, :], axis=2)

    # distance to cluster for each cell on each trial
    # assign 1 for in map 1 and -1 for in map 2
    dd_by_cells = (projY - projc2) / (projc1 - projc2)
    dd_by_cells = 2 * (dd_by_cells - .5)
    dd_by_cells = dd_by_cells.T
    
    # get the ideal distribution (k-means label for each trial)
    n_cells = dd_by_cells.shape[0]
    K = np.tile(W[:, map_idx-1], (n_cells, 1))

    # calculate log likelihood - this is the "remap score"
    ll_cells = K * np.log(1 + np.exp(dd_by_cells)) + (1 - K) * np.log(1 + np.exp(-dd_by_cells))

    return dd_by_cells, ll_cells


'''
Histology Functions:
Functions related to the anatomical location of cells in MEC.
'''
def get_coordinates(tip, entry, distances):
    '''
    Convert position along probe to 3D brain coordinates
    in microns, relative to MEC landmarks (see STAR Methods)

    Params:
    ------
    tip : ndarray
        coordinates of probe tip; shape(3)
        this comes from the histology
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

def remapper_locations(avg_ll_cell_all, \
                        axis_coords, min_coord, max_coord, \
                        BIN=50, THRESH=1, print_results=True):
    '''
    Compute the number of consistent remappers in each anatomical location.

    Params:
    ------
    avg_ll_cell_all : ndarray
        average log-likelihood distance to cluster for all cells
        comes from clu_distance_cells
        shape (n_total_cells, )
    axis_coords: ndarray
        DV, ML, or AP coordinates of each cell
    min_coord : int
        the minimum edge of the region in um (+ BIN)
        paper uses DV=50, ML=-200, AP=50
    max_coord : in
        the maximum edge of the region in um (+ BIN)
        paper uses DV=2050, ML=350, AP=700
    BIN : int
        bin size (um)
        default is 50um
    THRESH : int
        maximum average log-likelihood for a cell to be considered a "consistent remapper"
        default is 1

    Returns:
    -------
    (Use to produce fig 3f.)
    axis_pcts : ndarray
        Percent of cells in each anatomical bin that are consistent remappers.
    loc_remappers : ndarray
        Coords for each bin with consistent remappers in it.
    '''
    # get bin indices
    edges = np.arange(min_coord, max_coord, BIN)
    idx = np.digitize(axis_coords, edges)

    # get the total units in each bin and remappers in each bin
    axis_unique, ct_all = np.unique(idx, return_counts=True) # total cells per bin
    loc_remappers, ct_remappers = np.unique(idx[avg_ll_cell_all < THRESH], return_counts=True) # remappers per bin
    if ct_all.shape[0] == ct_remappers.shape[0]: # check that we didn't lose any cells
        axis_pcts = (ct_remappers/ct_all)*100
    else:
        axis_pcts = (ct_remappers/ct_all[axis_unique!=np.setdiff1d(axis_unique, loc_remappers)])*100

    # print the results
    if print_results:
        print(f'N cells total = {np.sum(ct_all)}')
        print(f'N cells strong remappers = {np.sum(ct_remappers)}')
        print(f'strong remappers overall = {(np.sum(ct_remappers)/np.sum(ct_all)*100):.2f}%')
    
    return axis_pcts, loc_remappers