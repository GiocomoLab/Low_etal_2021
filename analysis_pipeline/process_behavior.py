import numpy as np
from scipy import stats

def avg_speed(speed, trials):
    '''
    Calculate the average running speed on each trial.

    Params:
    ------
    speed in each time bin; shape (n_obs,)
    trial number for each time bin; shape (n_obs,)
    '''
    avg_speed = np.zeros_like(np.unique(trials))
    for t in np.unique(trials).astype(int):
        avg_speed[t] = np.mean(speed[trials==t])
    return avg_speed

def remap_vs_stable_speed(data, mice, sessions, \
                            MIN_REMAPS=3, print_output=True):
    '''
    Calculate the average speed during remap trials and stable blocks for each session.

    Params:
    ------
    MIN_REMAPS : int
        minimum number of remap events for a session to be included
    print_output : bool
        prints excluded sessions and total number of mice/sessions included

    Returns:
    -------
    mean_speed_remap, mean_speed_stable : list
        average speed for remap trials or stable blocks in each session; len(n_mice,)
        each list element is an array of shape (n_sessions,) for each mouse
    sem_remaps, sem_stable : list
        as above, but for SEM 
    '''

    # lists to store output
    mean_speed_remap = []
    mean_speed_stable = []
    sem_remaps = []
    sem_stable = []
    N_mice_included = 0
    N_sess_included = 0

    ct = -1
    if print_output:
        print(f'excluded sessions with fewer than {MIN_REMAPS} remapping events:')
    for m, session in zip(mice, sessions):
        ct += 1
        
        # add an array for each session for this mouse
        mean_speed_remap.append(np.zeros(len(session)))
        mean_speed_stable.append(np.zeros(len(session)))
        sem_remaps.append(np.zeros(len(session)))
        sem_stable.append(np.zeros(len(session)))   
        
        for i, s in enumerate(session):
            d = data[m][s]
            stable_idx = d['remap_stable_idx'][0]
            remap_idx = d['remap_stable_idx'][1]
            
            # keep only sessions with at least MIN_REMAPS
            if remap_idx[::2].shape[0] < MIN_REMAPS:
                if print_output:
                    print(f'{m}, session {s}')
                continue
                
            # get average speed on remap vs. stable trials
            avg_speed_remaps = d['avg_speeds'][remap_idx]
            avg_speed_stable = d['avg_speeds'][stable_idx]

            # store mean speed and SEM
            mean_speed_remap[-1][i] = np.mean(avg_speed_remaps)
            sem_remaps[-1][i] = stats.sem(avg_speed_remaps)
            mean_speed_stable[-1][i] = np.mean(avg_speed_stable)
            sem_stable[-1][i] = stats.sem(avg_speed_stable)
            
        # count the number of mice/sessions that were included
        idx = mean_speed_stable[-1] > 0
        if any(idx):
            N_mice_included += 1
            N_sess_included += np.sum(idx)

    if print_output:
        print(f'\nn mice >{MIN_REMAPS} remaps = {N_mice_included}')
        print(f'n sessions >{MIN_REMAPS} remaps = {N_sess_included}')

    return mean_speed_remap, mean_speed_stable, sem_remaps, sem_stable

def speed_by_block(remap_trials, stable_blocks, avg_speeds):
    '''
    Calculate the average speed in each pair of remap trials and each stable block
    for a given session.

    Params:
    ------
    remap_trials : ndarray
        trial numbers for trials bookending each remap event
    stable_blocks : ndarray
        trial numbers for stable trials between remap events
    
    Returns:
    -------
    '''
    # remap trial speeds
    remap_speeds_both = avg_speeds[remap_trials]
    remap_speeds = (remap_speeds_both[::2] + remap_speeds_both[1::2])/2

    # stable block speeds
    bdx = np.digitize(stable_blocks, remap_trials[::2])
    stable_speeds = np.zeros(remap_trials[::2].shape[0])
    for b in np.unique(bdx):
        if b < stable_speeds.shape[0]:
            stable_speeds[b] = np.mean(avg_speeds[stable_blocks[bdx==b]])

    return remap_speeds, stable_speeds

def speed_by_pos_bin(speed, pos, trials):
    '''
    Calculate the average running speed for each position bin on each trial.

    Params:
    ------
    speed : ndarray
        running speed in each time bin; shape (n_obs,)
    pos : ndarray
        position in each time bin; shape (n_obs,)
    trials : ndarray
        trial number at each timepoint; shape (n_obs,)

    Returns:
    -------
    '''
    n_trials = np.max(trials+1).astype(int)
    edges = np.arange(0, 400+bin_size, bin_size)
    bdx = np.digitize(pos, edges)

    binned_speed = np.zeros((n_trials, n_pos))
    for t in np.unique(trials).astype(int):
        for i, b in enumerate(np.unique(bdx)):
            idx = (trials==t) & (bdx==b)
            binned_speed[t, i] = np.mean(speed[idx])

    return binned_speed


''' helpers '''
def moving_avg(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n