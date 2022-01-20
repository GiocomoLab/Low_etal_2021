import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

def draw_raster(cell_IDs, ax, posx, trials, Y, color='k', minimalist=False):
    '''
    Make raster plot of spikes by position and trial.
    Can either input a single cell ID and one subplot axis,
    or a series of cell IDs and the corresponding 1D axis object.
    
    Params
    ------
    cell_IDs : tuple
        to index into the spike matrix
    ax : subplot axes object
        single subplot axis for one raster or multiple axes for many rasters
    posx : ndarray
        position at each observation; shape (n_obs)
    trial : ndarray
        trial number at each observation; shape (n_obs)
    Y : ndarray
        matrix of spike occurances; shape (n_obs, n_cells)
    color : string or 3-element list
        color to plot raster; optional, default is 'k'
    minimalist : bool
        exclude axis labels and spike count from title; optional, default is False
    '''
    if len(cell_IDs) > 1:
        for i, c in enumerate(cell_IDs):
            sdx = Y[:, np.where(cells==c)[0][0]].astype(bool)
            ax[i].plot(posx[sdx], trials[sdx], '.', c=color, markersize=6, alpha=.3)
            if minimalist:
                ax[i].set_title('cell ' + str(c))
                ax[i].tick_params(labelleft=False, labelbottom=False)
            else:
                ax[i].set_title('cell ' + str(c) + ', ' + str(int(np.sum(Y[:, np.where(cells==c)[0][0]]))) + ' spikes')
            ax[i].set_xlim((0, 400))
            ax[i].set_ylim((0, np.max(trial)))
    else:
        c = cell_IDs[0]
        sdx = Y[:, np.where(cells==c)[0][0]].astype(bool)
        ax.plot(posx[sdx], trials[sdx], '.', c=color, markersize=0.5, alpha=.3)
        if minimalist:
            ax.set_title('cell ' + str(c))
            ax.tick_params(labelleft=False, labelbottom=False)
        else:
            ax.set_title('cell ' + str(c) + ', ' + str(int(np.sum(Y[:, np.where(cells==c)[0][0]]))) + ' spikes')
        ax.set_xlim((0, 400))
        ax.set_ylim((0, np.max(trial)))

''' Define plot colors:

Assign colors based on track type
see: https://personal.sron.nl/~pault/#sec:qualitative Fig. 4
cue poor : green
cue rich : rose
'''
cp_color = [17/255, 119/255, 51/255, 1]
cr_color = [204/255, 102/255, 119/255, 1]

colors = [cp_color, cr_color]

'''
single cell colors for Figure 4
red, orange, yellow, green, blue, purple
'''
cell_colors = np.asarray([[165, 23, 14, 255], [241, 147, 45, 255], [1, 1, 1, 1], 
                          [78, 178, 101, 255], [82, 137, 199, 255], [153, 79, 136, 255]])/255
cell_colors = list(cell_colors)
for i, c in enumerate(cell_colors):
    cell_colors[i] = tuple(c)
cell_colors[2] = 'xkcd:gold'


''' FIGURE 1 '''

''' FIGURE 2 '''
def plot_fig2b_f(data, mouse, session, cell_IDs, \
                    f, gs, PT_SIZE=0.2):
    '''
    Make panels B-F
    (note that panel A is produced above as Fig. 1I)

    Params:
    ------
    data : dict
        dictionary of data indexed by mouse and session
    mouse : string
    session : string
    cell_IDs : ndarray
        cell IDs for 3 example cells
    f : figure object
    gs : gridspec axis object for subplots
    PT_SIZE : float
        point size for rasters (make larger for shorter sessions)

    Returns:
    -------
    f : figure object
    gs : gridspec axis object for subplots
    ''' 
    d = data[mouse][session]
    A = d['A'] # behavior
    B = d['B'] # spikes
    cells = d['cells'] # all cell IDs
    sim = d['similarity'] # trial-trial similarity

    ''' plot rasters '''
    ax_start = 0
    for i, cell_ID in enumerate(cell_IDs):
        # set axis
        ax_end = ax_start+2
        ax0 = plt.subplot(gs[ax_start:ax_end])
        
        # get spike index by observation for this example cell
        sdx = B[:, np.where(cells==cell_ID)[0][0]].astype(bool)
        
        # plot spikes
        ax0.scatter(A[:, 0][sdx], A[:, 2][sdx], \
                    color='k', lw=0, s=PT_SIZE, alpha=.3)
            
        # set axis params
        ax0.set_xlim([0, 400])
        ylim_ax = [0, np.max(A[:, 2])]
        ax0.set_ylim(ylim_ax[::-1])
        

        # label axes
        ax0.set_title('unit {}'.format(cell_ID), fontsize=10, pad=16)
        if i==0:
            ax0.set_ylabel('trial number', fontsize=9, labelpad=1)
        else:
            ax0.tick_params(labelleft=False)
        if i==1:
            if mouse == 'Seattle':
                # account for 4 rasters
                ax0.set_xlabel('position (cm)', fontsize=9, labelpad=1, \
                               horizontalalignment='left', x=0.5)
            else:
                ax0.set_xlabel('position (cm)', fontsize=9, labelpad=1)
        ax0.set_xticks(np.arange(0, 450, 200))
        ax0.tick_params(which='major', labelsize=7.5, pad=0.5)
        
        # set up for next subplot
        ax_start = ax_end

    ''' plot similarity '''
    ax1 = plt.subplot(gs[ax_end:])
    im = ax1.imshow(sim, clim=[0.1, 0.6], \
                    aspect='auto', cmap='Greys')

    # label axes
    ax1.set_xlabel('trial number', fontsize=9, labelpad=1)
    ax1.tick_params(labelleft=False, which='major', labelsize=7.5, pad=0.5)
    ax1.set_title('network\nsimilarity', fontsize=10, pad=5)
    cbar = f.colorbar(im, shrink=0.5)
    cbar.ax.tick_params(labelsize=7.5)

    return f, gs

''' FIGURE 3 '''
def plot_fig3b(data, mouse='Pisa', session='0430_1'):
    '''
    To plot network-wide trail-trial similarity, population distance to 
    k-means cluster and cell-by-cell distance to cluster.
    '''
    # get data
    d = data[mouse][session]
    A = d['A']
    normalized_dd_all = d['dist']
    sim = d['similarity']
    dd_by_cell = d['cells_dist']

    # figure params
    gs = gridspec.GridSpec(20, 1, hspace=1.5)
    f = plt.figure(figsize=(2.3, 4))
    c1 = cp_color
    c2 = 'k'

    # plot distance to cluster
    ax0 = plt.subplot(gs[11:15])
    ax0.plot(normalized_dd_all, color=c1, lw=1.5, alpha=1)
    ax0.set_xlim([0, normalized_dd_all.shape[0]])
    ax0.set_yticks([-1, 1])
    ax0.set_ylabel('distance\nscore', fontsize=9, labelpad=1)
    ax0.tick_params(labelbottom=False, which='major', labelsize=7.5, pad=0.5)

    # plot Euclidean similarity score
    ax1 = plt.subplot(gs[:11])
    im = ax1.imshow(sim, clim=[0.1, 0.7], aspect='auto', 
                    cmap='Greys', interpolation='none')
    ax1.tick_params(labelbottom=False, which='major', labelsize=7.5, pad=0.5)
    ax1.set_ylabel('trial number', fontsize=9, labelpad=1)
    ax1.set_title('network similarity', fontsize=10, pad=3)

    # plot distance to cluster by cells
    ax2 = plt.subplot(gs[15:])
    im = ax2.imshow(dd_by_cell, clim=[-1, 1], aspect='auto', 
                    cmap='binary', interpolation='none')
    ax2.set_ylim(ax2.get_ylim()[::-1])
    ax2.tick_params(which='major', labelsize=7.5, pad=0.5)
    ax2.set_xlabel('trial number', fontsize=9, labelpad=1)
    ax2.set_ylabel('cells', fontsize=9, labelpad=1)

    # set axes
    ax0.set_xticks([0, 200, 400])
    ax1.set_yticks([0, 200, 400])
    ax1.set_xticks([0, 200, 400])
    ax2.set_xticks([0, 200, 400])

    return f, gs


def plot_fig3e(all_within, all_across, N_cue_poor, N_cue_rich):
    '''
    within vs. across map similarity for all 2-map sessions
    '''
    # set figure params
    f, ax = plt.subplots(1, 1, figsize=(1.2, 1.4))
    DOT_SIZE = 20
    DOT_LW = 1
    BAR_SIZE = 10
    BAR_WIDTH = 2.8
    j = np.random.randn(all_within.shape[0]) * .08

    # plot connector lines
    for k, w in enumerate(all_within):
        a = all_across[k]
        x_vals = [1.08+j[k], 1.92+j[k]]
        y_vals = [w-0.005, a+0.005]
        ax.plot(x_vals, y_vals, '-', color='xkcd:gray', lw=DOT_LW, zorder=1,  alpha=1)

    # within map similarity, colored by track type
    ax.scatter(np.full(N_cue_poor, 1)+j[:N_cue_poor], all_within[:N_cue_poor], \
               facecolors=cp_color, edgecolors='k', alpha=0.7, \
               s=DOT_SIZE, lw=DOT_LW, zorder=2, label='poor') 
    ax.scatter(np.full(N_cue_rich, 1)+j[N_cue_poor:], all_within[N_cue_poor:], \
               facecolors=cr_color, edgecolors='k', alpha=0.7, \
               s=DOT_SIZE, lw=DOT_LW, zorder=2, label='rich') 

    # across map similarity, colored by track type
    ax.scatter(np.full(N_cue_poor, 2)+j[:N_cue_poor], all_across[:N_cue_poor], \
               facecolors=cp_color, edgecolors='k', alpha=0.7, \
               s=DOT_SIZE, lw=DOT_LW, zorder=2) 
    ax.scatter(np.full(N_cue_rich, 2)+j[N_cue_poor:], all_across[N_cue_poor:], \
               facecolors=cr_color, edgecolors='k', alpha=0.7, \
               s=DOT_SIZE, lw=DOT_LW, zorder=2) 

    # plot means
    ax.plot(1, np.mean(all_within), '_', c='k', \
            markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=3)
    ax.plot(2, np.mean(all_across), '_', c='k', \
            markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=3)

    # label axes etc.
    labels = ['within', 'across']
    ax.set_xlim([0.5, 2.5])
    ax.set_xticks([1, 2])
    ax.set_yticks([0.1, 0.3, 0.5])
    ax.set_xticklabels(labels, rotation=45)
    ax.tick_params(which='major', labelsize=8, pad=0.8)
    ax.set_ylabel('avg. correlation', fontsize=10, labelpad=1)
    plt.legend(bbox_to_anchor=(1,1,0,0), fontsize=8)

    return f, ax

def plot_fig3f(DV_pcts, ML_pcts, AP_pcts, \
                DV_loc, ML_loc, AP_loc, \
                THRESH=1):
    '''
    % of all cells that are consistent remappers for different anatomical locations
    see STAR Methods for more details

    Params:
    ------
    DV_pcts, ML_pcts, AP_pcts : ndarray
        the % of cells that are consistent remappers in each
        DV, ML, and AP bin (from remapper_locations())
        shape (n_total_cells, )
    DV_loc, ML_loc, AP_loc
        anatomical location of each bin center (from from remapper_locations())
        shape (n_bins, )
    THRESH : int
        maximum average log-likelihood for a cell to be considered a "consistent remapper"
        default is 1, as in the paper
    '''
    gs  = gridspec.GridSpec(1, 3, wspace=0.2)
    f = plt.figure(figsize=(3, 1.2))

    # plot scores by DV coords:
    ax0 = plt.subplot(gs[0])
    ax0.bar(DV_loc, DV_pcts, width=1, color='k')
    ax0.set_xticks([0, 1000//50])
    ax0.set_xticklabels([0, 1000])
    ax0.set_yticks(np.arange(0, 125, 50))
    ax0.tick_params(which='major', labelsize=7.5, pad=1)
    ax0.set_title('DV', fontsize=10, pad=1.5)
    ax0.set_ylabel('% of cells', fontsize=10, labelpad=1)

    # plot scores by ML coords:
    ax1 = plt.subplot(gs[1])
    ax1.bar(ML_loc, ML_pcts, width=1, color='k')
    ax1.set_xticks([0, 250//50, 500//50])
    ax1.set_xticklabels([-250, 0, 250])
    ax1.set_yticks(np.arange(0, 125, 50))
    ax1.tick_params(labelleft=False, which='major', labelsize=7.5, pad=1)
    ax1.set_title('ML', fontsize=10, pad=1.5)
    ax1.set_xlabel('distance to reference ($\mu$m)', fontsize=10, labelpad=1)

    # plot scores by AP coords:
    ax2 = plt.subplot(gs[2])
    ax2.bar(AP_loc, AP_pcts, width=1, color='k')
    ax2.set_xticks([200//50, 500//50])
    ax2.set_xticklabels([200, 500])
    ax2.set_yticks(np.arange(0, 125, 50))
    ax2.tick_params(labelleft=False, which='major', labelsize=7.5, pad=1)
    ax2.set_title('AP', fontsize=10, pad=1.5)

    return f, gs



''' FIGURE 4 '''
''' set example cell colors '''
# red, orange, green, blue, purple
cell_colors = np.asarray([[165, 23, 14, 255], [241, 147, 45, 255], [1, 1, 1, 1], 
                          [78, 178, 101, 255], [82, 137, 199, 255], [153, 79, 136, 255]])/255
cell_colors = list(cell_colors)
for i, c in enumerate(cell_colors):
    cell_colors[i] = tuple(c)
cell_colors[2] = 'xkcd:gold'

def plot_fig4a(data, mouse, session, cell_IDs, \
                binned_pos, FR_naive, FR_0, FR_1, \
                FR_sem, FR_0_sem, FR_1_sem):
    '''
    Examples of remapping cells.

    Params:
    ------
    cell_IDs : ndarray
        ID numbers for the example cells.
    binned_pos : ndarray
        position bin centers for each firing rate measurement
        shape (n_pos_bins,)
    FR_naive, FR_0, FR_1 : ndarray
        firing rate by position across maps or within each map
        shape (n_pos_bins, n_cells)
    FR_sem, FR_0_sem, FR_1_sem : ndarray
        SEM for the firing rate arrays; shape (n_pos_bins, n_cells)
    '''
    # get data
    d = data[mouse][session] 
    A = d['A']
    B = d['B']  
    cells = d['cells']

    # figure params:
    gs = gridspec.GridSpec(19, len(cell_IDs), hspace=1.5, wspace=0.3)
    f = plt.figure(figsize=(6, 2.3)) 
    PT_SIZE = 1
    LW_MEAN = 1
    LW_SEM = 0.3

    for i, cell_ID in enumerate(cell_IDs):
        # draw raster plot
        ax0 = plt.subplot(gs[:11, i])
        sdx_0 = B[map0_idx, np.where(cells==cell_ID)[0][0]].astype(bool)
        ax0.scatter(A[map0_idx, 0][sdx_0], A[map0_idx, 2][sdx_0], color='k', lw=0, s=PT_SIZE, alpha=.1)
        sdx_1 = B[map1_idx, np.where(cells==cell_ID)[0][0]].astype(bool)
        ax0.scatter(A[map1_idx, 0][sdx_1], A[map1_idx, 2][sdx_1], color=cell_colors[i], lw=0, s=PT_SIZE, alpha=.1)
        ax0.set_xlim((0, 400))
        ylim_ax = [0, np.max(A[:, 2])]
        ax0.set_ylim(ylim_ax[::-1])
        ax0.set_title('cell ' + str(cell_ID), fontsize=10, pad=3)

        # plot tuning curves with SEM
        sdx = (np.where(cells==cell_ID)[0][0]).astype(int)
        ax1 = plt.subplot(gs[11:15, i])
        ax1.plot(FR_0[:, sdx], 'k', lw=LW_MEAN, alpha=0.9)
        ax1.fill_between(binned_pos/2, FR_0[:, sdx] + FR_0_sem[:, sdx], FR_0[:, sdx] - FR_0_sem[:, sdx],
                         color='k', linewidth=LW_SEM, alpha=0.3)
        ax1.plot(FR_1[:, sdx], color=cell_colors[i], lw=LW_MEAN, alpha=1)
        ax1.fill_between(binned_pos/2, FR_1[:, sdx] + FR_1_sem[:, sdx], FR_1[:, sdx] - FR_1_sem[:, sdx],
                         color=cell_colors[i], linewidth=LW_SEM, alpha=0.4)
     
        # plot naive tuning curve with SEM
        sdx = (np.where(cells==cell_ID)[0][0]).astype(int)
        ax2 = plt.subplot(gs[15:, i])
        ax2.plot(FR_naive[:, sdx], 'k', lw=LW_MEAN, alpha=0.6)
        ax2.fill_between(binned_pos/2, FR_naive[:, sdx] + FR_sem[:, sdx], \
                            FR_naive[:, sdx] - FR_sem[:, sdx],
                            color='k', linewidth=LW_SEM, alpha=0.1)
        
        if i == 0:
            ax0.set_ylabel('trial number', fontsize=10, labelpad=2)
            ax1.set_ylabel('FR (Hz)', fontsize=10, labelpad=7, horizontalalignment='right', y=0.8)
        else:
            ax0.tick_params(labelleft=False)
            ax1.tick_params(labelleft=False)
            ax2.tick_params(labelleft=False)
        if i == len(cell_IDs)//2:
            ax2.set_xlabel('track position (cm)', fontsize=10, labelpad=2, horizontalalignment='right', x=0.8)
        ax0.set_xticks(np.arange(0, 425, 200))
        ax0.tick_params(labelbottom=False, which='major', labelsize=8, pad=0.5)
        ax1.set_xlim([0, 200])
        ax1.set_yticks([0, 20])
        ax1.set_ylim([0, 27])
        ax1.set_xticks(np.arange(0, 225, 100))
        ax1.tick_params(labelbottom=False, which='major', labelsize=8, pad=0.5)
        ax2.set_xlim([0, 200])
        ax2.set_yticks([0, 20])
        ax2.set_ylim([0, 27])
        ax2.set_xticks(np.arange(0, 225, 100))
        ax2.set_xticklabels(np.arange(0, 450, 200))
        ax2.tick_params(which='major', labelsize=8, pad=0.5)

    return f, gs

def plot_fig4b(dissim, pct_dFR):
    '''
    Plot fold-change in firing rate versus spatial dissimilarity across maps

    Params:
    ------
    dissim : ndarray
        1 - cosine similarity across maps; shape (n_cells, )
    pct_FR : ndarray
        percent change in peak firing rate; shape (n_cells, )
    '''
    # figure params
    gs = gridspec.GridSpec(6, 6, hspace=0, wspace=0)
    f = plt.figure(figsize=(3, 3)) 
    PT_SIZE = 5
    LW_THRESH = 1.5
    LW_HIST = 1.5

    # plot change in FR vs. dissimilarity, all cells
    ax0 = plt.subplot(gs[2:, :-2])
    ax0.scatter(dissim, pct_dFR, color='k', s=PT_SIZE, lw=0, alpha=0.2)
    ax0.set_xlim(0, 1.05)
    ymax = ax0.get_ylim()[1] + 25
    ax0.set_xlabel('spatial dissimilarity', fontsize=10, labelpad=1)
    ax0.set_ylabel('fold change in\npeak firing rate', fontsize=10, labelpad=1)

    # plot density, spatial
    ax1 = plt.subplot(gs[:2, :-2])
    n1, bins1, _ = ax1.hist(dissim, bins=50, density=True, histtype='stepfilled', 
                          lw=LW_HIST, edgecolor='k', facecolor='xkcd:light gray', alpha=0.7)
    ax1.set_xlim(0, 1.05)
    ymax_ax1 = ax1.get_ylim()[1]
    ax1.tick_params(labelbottom=False, which='major', labelsize=8, pad=0.5)
    ax1.set_ylabel('% cells', fontsize=10, labelpad=2)

    # plot density, firing rate
    ax2 = plt.subplot(gs[2:, -2:])
    n2, bins2, _ = ax2.hist(pct_dFR, bins=50, density=True, histtype='stepfilled', orientation='horizontal', 
                          lw=LW_HIST, edgecolor='k', facecolor='xkcd:light gray', alpha=0.7)
    ax2.tick_params(labelleft=False, which='major', labelsize=8, pad=0.5)
    ax2.set_xlabel('% cells', fontsize=10, labelpad=2)
    xmax = ax2.get_xlim()[1]

    # plot medians etc
    ax0.vlines(np.median(dissim), 0, ymax, colors='xkcd:vermillion', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax1.vlines(np.median(dissim), 0, ymax_ax1, colors='xkcd:vermillion', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax0.vlines(np.percentile(dissim, 95), 0, ymax, colors='xkcd:gold', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax1.vlines(np.percentile(dissim, 95), 0, ymax_ax1, colors='xkcd:gold', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax0.hlines(np.median(pct_dFR), 0, 1.05, colors='xkcd:vermillion', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax2.hlines(np.median(pct_dFR), 0, xmax, colors='xkcd:vermillion', lw=LW_THRESH, 
               linestyles='dashed', alpha=1, label='median')
    ax0.hlines(np.percentile(pct_dFR, 95), 0, 1.05, colors='xkcd:gold', lw=LW_THRESH, 
               linestyles='dashed', alpha=1)
    ax2.hlines(np.percentile(pct_dFR, 95), 0,  xmax, colors='xkcd:gold', lw=LW_THRESH, 
               linestyles='dashed', alpha=1, label='95$^{th}$ pct')

    # lims and labels
    ax0.set_xticks([0, 0.5, 1])
    ax0.set_xticklabels([0, 0.5, 1])
    ax0.set_yticks([0, 100, 200])
    ax0.set_yticklabels([1, 2, 3])
    ax0.set_ylim(-2, ymax)
    ax2.set_ylim(-2, ymax)
    ax0.tick_params(which='major', labelsize=8, pad=0.5)

    # make the hist ticks meaningful
    b1_width = np.unique(np.round(np.diff(bins1), 8))
    vals = np.arange(0, 35, 10)
    t1s = np.zeros(vals.shape)
    for i, v in enumerate(vals):
        t1s[i] = v/(100*b1_width)
    ax1.set_yticks(t1s)
    ax1.set_yticklabels(vals)

    b2_width = np.unique(np.round(np.diff(bins2), 8))
    vals = np.arange(2, 9, 2)
    t2s = np.zeros(vals.shape)
    for i, v in enumerate(vals):
        t2s[i] = v/(100*b2_width)
    ax2.set_xticks(t2s)
    ax2.set_xticklabels(vals)
    ax2.legend(bbox_to_anchor=(0.8,0.9,0,0), fontsize=7.5)

    return f, gs

def plot_fig4c(dissim, pct_dFR, cells, cell_IDs):
    '''
    Plot change in firing rate vs. spatial dissimilarity for an example session.

    Params
    ------
    dissim : ndarray
        1 - cosine similarity across maps; shape (n_cells, )
    pct_dFR : ndarray
        percent change in peak firing rate; shape (n_cells, )
    cells : ndarray
        all cell IDs for this session
    cell_IDs : ndarray
        example cell IDs
    '''
    # figure params
    f, ax = plt.subplots(1, 1, figsize=(1.7, 1.7))
    POINT_SIZE = 28

    # plot change in firing rate vs. dissimilarity
    ax.scatter(dissim, pct_dFR, color='k', s=POINT_SIZE, lw=0, alpha=0.2)

    # plot examples
    for i, cell_ID in enumerate(cell_IDs):
        cdx = np.where(cells==cell_ID)[0][0]
        ax.scatter(dissim[cdx], frac_dFR[cdx], \
                    edgecolors='k', facecolors=cell_colors[i], \
                    lw=0.3, s=POINT_SIZE+2, alpha=1)

    # labels and lims
    ax.set_xlim(-0.02, ax.get_xlim()[1])
    ax.set_xlabel('spatial dissimilarity', fontsize=10, labelpad=1)
    ax.set_ylabel('change in FR', fontsize=10, labelpad=1)
    ax.set_yticks(np.arange(0, 175, 50))
    ax.set_yticklabels(np.arange(1, 3, 0.5))
    ax.set_xticks([0, 0.25, 0.5])
    ax.set_xticklabels([0, 0.25, 0.5])
    ax.tick_params(which='major', labelsize=8, pad=0.8)

    return f, ax


def plot_fig4d(SI_map1, SI_map2, SI_map1_shuff, SI_map2_shuff, cells, cell_IDs):
    '''
    Plot the spatial information for each cell in each map for an example session.

    Params
    ------
    SI_map1, SI_map2 : ndarray
        spatial information for each cell in each map
    SI_map1_shuff, SI_map2_shuff : ndarray
        shuffled spatial information for each map
    cells : ndarray
        all cell IDs for this session
    cell_IDs : ndarray
        example cell IDs
    '''
    f, ax = plt.subplots(1, 1, figsize=(1.7, 1.7))
    POINT_SIZE = 28
    UNITY_WIDTH = 2
    SHUFF_WIDTH = 1.5

    ax.scatter(SI_map2, SI_map1, color='k', lw=0, s=POINT_SIZE, alpha=0.2)
    ax.set_ylabel('spatial info.\nmap 1', fontsize=10, labelpad=1)
    ax.set_xlabel('spatial info.\nmap 2', fontsize=10, labelpad=1)

    # plot examples
    for i, c in enumerate(cell_IDs):
        cell_ID = c
        cdx = np.where(cells==cell_ID)[0][0]
        ax.scatter(SI_map1[cdx], SI_map0[cdx], \
                    edgecolors='k', facecolors=cell_colors[i], lw=0.3, s=POINT_SIZE+1, alpha=1)

    # plot unity
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    min_lim = np.min([xlims[0], ylims[0]])
    max_lim = np.max([xlims[1], ylims[1]])
    ax.plot([min_lim, max_lim], [min_lim, max_lim], \
                '--k', lw=SHUFF_WIDTH, alpha=1)

    # plot significance threshold
    shuff_0 = SI_map1_shuff.flatten()
    shuff_0_thresh = np.percentile(shuff_0, 95)
    ax.plot(ax.get_xlim(), [shuff_0_thresh, shuff_0_thresh], \
            ':k', lw=SHUFF_WIDTH, alpha=1)
    shuff_1 = SI_map2_shuff.flatten()
    shuff_1_thresh = np.percentile(shuff_1, 95)
    ax.plot([shuff_1_thresh, shuff_1_thresh], ax.get_ylim(), \
            ':k', lw=SHUFF_WIDTH, alpha=1)

    ax.set_xlim([min_lim, max_lim])
    ax.set_ylim([min_lim, max_lim])
    ax.set_xticks([0, 2.5, 5])
    ax.set_yticks([0, 2.5, 5])
    ax.tick_params(which='major', labelsize=8, pad=0.8)

    return f, ax


def plot_fig4e(percents, label=True):
    '''
    Plot the percent of all cells that are spatial in both maps, 
    one map, or neither map.

    Params:
    ------
    percents : ndarray
        percent of cells that are spatial in both, one, or neither map.
        shape (3, )
    labels : bool
        if True, includes category and percentage labels on the plot
    '''
    # figure params
    f, ax = plt.subplots(1, 1, figsize=(2, 2))
    labels = ['both', 'one', 'neither']

    # plot percentages
    if label:
        p = ax.pie(pcts, labels=labels, autopct='%1.1f%%', \
              colors = ['k', 'k', 'w'], \
              wedgeprops=dict(linewidth=0.5, edgecolor='k'))
    else:
        p = ax.pie(percents, labels=None, \
                      colors = ['k', 'k', 'w'], \
                      wedgeprops=dict(linewidth=0.5, edgecolor='k'))
    
    # adjust the colors
    p[0][1].set_alpha(0.2)
    p[0][0].set_alpha(0.8)

    return f, ax


def plot_fig4f(data, mice, sessions):
    # figure params
    f, ax = plt.subplots(1, 1, figsize=(1.7, 1.7))
    PT_SIZE = 15
    LW_THRESH = 1.5

    for m, session in zip(mice, sessions):
        for s in session:
            d = data[m][s]
            ll_cells = np.mean(d['ll_cells'].copy(), axis=1)
            
            # get data
            SI_map0 = d['SI'][0, :].copy()
            SI_map1 = d['SI'][1, :].copy()
            shuff_SI_map0 = d['shuff'][0].copy()
            shuff_SI_map1 = d['shuff'][1].copy()
            SI_keep = np.max(d['SI'].copy(), axis=0) # we will plot whichever SI is higher

            # get sig indices for each map
            flat_shuff_SI_map0 = shuff_SI_map0.flatten()
            flat_shuff_SI_map1 = shuff_SI_map1.flatten()        
            sig_0_idx = SI_map0 > np.percentile(flat_shuff_SI_map0, 95)
            sig_1_idx = SI_map1 > np.percentile(flat_shuff_SI_map1, 95)
            both_idx = [sig_0_idx & sig_1_idx]
            one_idx = [(sig_0_idx & ~sig_1_idx) | (~sig_0_idx & sig_1_idx)]
            not_idx = [~sig_0_idx & ~sig_1_idx]
            
            # plot the data
            ax.scatter(SI_keep, ll_cells, color='k', 
                       s=PT_SIZE, lw=0, alpha=0.3, zorder=1)

    # add threshold for consistent remapper
    y_lims = ax.get_ylim()
    x_lims = ax.get_xlim()
    ax.hlines(THRESH, x_lims[0], x_lims[1], colors='k', lw=LW_THRESH, linestyles='dashed', alpha=1, zorder=4)

    # labels and lims
    ax.set_xlim([-0.1, x_lims[1]])
    ax.set_ylim(y_lims)
    ax.set_xticks([0, 4, 8, 12])
    ax.set_yticks(np.arange(5))
    ax.tick_params(which='major', labelsize=8, pad=0.8)
    ax.set_ylabel('distance to\nmap center', fontsize=10, labelpad=1)
    ax.set_xlabel('spatial information', fontsize=10, labelpad=1)

    return f, ax


''' FIGURE 5 '''
''' set colors for each cell type
orange = putative grid cells
blue = putative border cells
'''
cell_colors = np.asarray([[241, 147, 45, 255], [25, 101, 176, 255]])
cell_colors = cell_colors/255
cell_colors = list(cell_colors)
for i, c in enumerate(cell_colors):
    cell_colors[i] = tuple(c)
GC_COLOR = cell_colors[0]
BD_COLOR = cell_colors[1]

def plot_fig5b(n_grid, n_border, n_spatial, n_total_cells, label_plot=False):
    '''
    plot the proportion of grid, border, non-grid/non-border spatial cells, 
    and non-spatial cells across all gain manipulation sessions
    '''
    # set figure params
    f, ax = plt.subplots(1, 1, figsize=(0.9, 1))
    pie_colors = [cell_colors[0], cell_colors[1], 'k', 'w']
    labels = ['grid', 'border', 'other spatial', 'other']
    
    pcts = np.asarray([n_grid, n_border, n_spatial, n_total_cells]) / n_total_cells * 100
    if label_plot:
        p = ax.pie(pcts, labels=labels, autopct='%1.1f%%', 
                    colors = pie_colors,
                    wedgeprops=dict(linewidth=0.5, edgecolor='k'))
    else:   
        p = ax.pie(pcts, labels=None, 
                    colors = pie_colors,
                    wedgeprops=dict(linewidth=0.5, edgecolor='k'))
    for i in range(3):
        p[0][i].set_alpha(0.8)

    return f, ax


def plot_fig5cd(data, mouse, session, cell_IDs, \
                FR_0, FR_1, FR_0_sem, FR_1_sem, binned_pos):
    '''
    plot example rasters and tuning curves for 3 example grids and 2 example borders
    last 5 normal trials and 5 gain manipulation trials
    
    Params:
    ------
    cell_IDs : ndarray
        ID numbers for the example cells.
    FR_0, FR_1 : ndarray
        firing rate by position for last 5 normal, 5 gain manipulation trials
        shape (n_pos_bins, n_cells)
    FR_0_sem, FR_1_sem : ndarray
        SEM for the firing rate arrays; shape (n_pos_bins, n_cells)
    binned_pos : ndarray
        position bin centers for the firing rate arrays; shape (n_pos_bins,)
    '''
    # get data
    d = data[mouse][session] 
    A = d['A_gm']
    B = d['B_gm']  
    cells = d['cells']
    [manip_idx, normal_idx] = d['gain_idx']

    # get correlation/similarity for each cell
    corr_gain_manip = d['corr_gain_manip']
    cell_idx = np.isin(cells, cell_IDs)
    correlations = corr_gain_manip[cell_idx]

    # set figure params
    gs = gridspec.GridSpec(2, len(cell_IDs), hspace=0.2, wspace=0.6)
    f = plt.figure(figsize=(4.8, 1.2)) 
    PT_SIZE = 8
    PT_W = 0.7
    LW_FR = 0.75
    LW_SEM = 0.5
    
    for i, cell_ID in enumerate(cell_IDs):
        if i < 3:
            COLOR = GC_COLOR
        else:
            COLOR = BD_COLOR
        
        # draw raster plot
        ax0 = plt.subplot(gs[0, i])
        sdx = B[normal_idx, :][:, np.where(cells==cell_ID)[0][0]].astype(bool)
        ax0.scatter(A[:, 0][normal_idx][sdx], A[:, 2][normal_idx][sdx], 
                    marker='|', c='k', lw=PT_W, s=PT_SIZE, alpha=.4)
        sdx = B[manip_idx, :][:, np.where(cells==cell_ID)[0][0]].astype(bool)
        ax0.scatter(A[:, 0][manip_idx][sdx], A[:, 2][manip_idx][sdx], 
                    marker='|', c=COLOR, lw=PT_W, s=PT_SIZE, alpha=.4)

        if i == 0:
            ax0.set_ylabel('trial', fontsize=9, labelpad=1)
            ax0.set_title('unit {}'.format(cell_ID), fontsize=9, pad=15)
            ax0.text(-180, 293, 'corr = {}'.format(np.round(correlations[i], 2)), fontsize=8)
        else:
            ax0.set_title('unit {}'.format(cell_ID), fontsize=9, pad=15)
            ax0.text(100, 293, '{}'.format(np.round(correlations[i], 2)), fontsize=8)
        ax0.tick_params(which='major', labelbottom=False, labelsize=7.5, pad=0.5)
        ax0.set_xlim((0, 400))
        ax0.set_xticks([0, 200, 400])
        ylim_ax = [np.min(A[:, 2])+4.5, np.max(A[:, 2])+0.5]
        ax0.set_ylim(ylim_ax[::-1])

        # plot tuning curve with SEM
        sdx = (np.where(cells==cell_ID)[0][0]).astype(int)
        ax1 = plt.subplot(gs[1, i])
        ax1.plot(FR_0[:, sdx], 'k', linewidth=LW_FR)
        ax1.fill_between(binned_pos/2, FR_0[:, sdx] + FR_0_sem[:, sdx], FR_0[:, sdx] - FR_0_sem[:, sdx], 
                         color='k', lw=LW_SEM, alpha=0.1)
        ax1.plot(FR_1[:, sdx], color=COLOR, linewidth=LW_FR)
        ax1.fill_between(binned_pos/2, FR_1[:, sdx] + FR_1_sem[:, sdx], FR_1[:, sdx] - FR_1_sem[:, sdx], 
                         color=COLOR, lw=LW_SEM, alpha=0.1)

        if i == 0:
            ax1.set_ylabel('FR (Hz)', fontsize=9, labelpad=5)
        ax1.set_xlim([0, 200])
        ax1.set_xticks(np.arange(0, 225, 100))
        ax1.set_xticklabels(np.arange(0, 450, 200))
        ax1.tick_params(which='major', labelsize=7.5, pad=0.5)
        if i == 1:
            ax1.set_xlabel('position (cm)', fontsize=9, labelpad=1)
        elif i == 3:
            ax1.set_xlabel('position (cm)', fontsize=9, labelpad=1, horizontalalignment='left', x=0.5)

    return f, gs

def plot_fig5e(pct_dFR, all_angles, all_grid, all_border, nongc_nonbd_stable):
    '''
    plot the change in firing rate and spatial dissimilarity across maps
    for grid, border, and non-grid/non-border spatial cells

    Params
    ------
    pct_dFR : ndarray
        percent change in firing rate across maps; shape (n_total_cells,)
    all_angles : ndarray
        spatial similarity across maps; shape (n_total_cells,)
    all_grid : ndarray
        indices for grid cells across all sessions; shape (n_total_cells,)
    all_border : ndarray
        indices for border cells across all sessions; shape (n_total_cells,)
    nongc_nonbd_stable : ndarray
        indices for other spatial cells across all sessions; shape (n_total_cells,)
    '''
    # get the data for each cell type
    dFR_gc = pct_dFR[all_grid]
    dFR_bd = pct_dFR[all_border]
    dFR_sp = pct_dFR[nongc_nonbd_stable]

    angle_gc = 1 - all_angles[all_grid]
    angle_bd = 1- all_angles[all_border]
    angle_sp = 1- all_angles[nongc_nonbd_stable]

    n_gc = dFR_gc.shape[0]
    n_bd = dFR_bd.shape[0]
    n_sp = dFR_sp.shape[0]

    # plotting params
    gs = gridspec.GridSpec(1, 2, wspace=0.9)
    f = plt.figure(figsize=(2.2, 1.2))
    PLOT_COLORS = [GC_COLOR, BD_COLOR, 'k']
    PT_SIZE = 5
    PT_LW = 0
    BAR_SIZE = 4
    BAR_WIDTH = 0.8
    LW_PCT = 0.6
    SIG_SIZE = 5
    LW_SIG = 0.8
    POSITIONS = np.asarray([1, 3, 5])

    # set jitter
    JIT = 0.15
    j_gc = np.random.randn(n_gc) * JIT
    j_bd = np.random.randn(n_bd) * JIT
    j_sp = np.random.randn(n_sp) * JIT

    # CHANGE IN FIRING RATE
    ax0 = plt.subplot(gs[0])
    ax0.scatter(np.full(n_gc, POSITIONS[0])+j_gc, dFR_gc, s=PT_SIZE, lw=PT_LW, 
                color=GC_COLOR, alpha=0.3, zorder=1)
    ax0.scatter(np.full(n_bd, POSITIONS[1])+j_bd, dFR_bd, s=PT_SIZE, lw=PT_LW, 
                color=BD_COLOR, alpha=0.3, zorder=1)
    ax0.scatter(np.full(n_sp, POSITIONS[2])+j_sp, dFR_sp, s=PT_SIZE, lw=PT_LW, 
                color='k', alpha=0.3, zorder=1)

    # mark median and 95th percentiles
    dFR_medians = np.asarray([np.median(dFR_gc), np.median(dFR_bd), np.median(dFR_sp)])
    dFR_5 = np.asarray([np.percentile(dFR_gc, 5), np.percentile(dFR_bd, 5), np.percentile(dFR_sp, 5)])
    dFR_95 = np.asarray([np.percentile(dFR_gc, 95), np.percentile(dFR_bd, 95), np.percentile(dFR_sp, 95)])
    ax0.plot(POSITIONS[:-1], dFR_medians[:-1], '_k', markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=2, alpha=0.7)
    ax0.plot(POSITIONS[-1], dFR_medians[-1], '_w', markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=2, alpha=0.7)
    ax0.vlines(POSITIONS, dFR_5, dFR_95, lw=LW_PCT, colors=['k', 'k', 'w'], linestyles='solid', zorder=2, alpha=0.7)

    # mark significance
    xvals = np.asarray([POSITIONS[1]-0.3, POSITIONS[1], POSITIONS[1]+0.3, np.mean(POSITIONS[:-1])])
    yvals = np.append(np.full(3, 198), [178])
    ax0.hlines([170, 190], [POSITIONS[0], POSITIONS[0]], POSITIONS[1:], 
                colors='k', linestyles='solid', lw=LW_SIG)
    ax0.scatter(xvals, yvals, c='k', marker='*', lw=0, s=SIG_SIZE)

    # set axis params
    ax0.tick_params(which='major', labelsize=7.5, pad=0.5)
    ax0.set_xticks(POSITIONS)
    ax0.set_xticklabels(['grid', 'border', 'spatial'], rotation=90)
    ax0.set_xlim(POSITIONS[0] - 1, POSITIONS[2] + 1)
    ax0.set_yticks(np.arange(0, 210, 100))
    ax0.set_yticklabels([0, 1, 2])
    ax0.set_ylabel('fold change in FR', fontsize=9, labelpad=1)

    # CHANGE IN SPATIAL CODING
    ax1 = plt.subplot(gs[1])
    ax1.scatter(np.full(n_gc, POSITIONS[0])+j_gc, angle_gc, s=PT_SIZE, lw=PT_LW, color=GC_COLOR, alpha=0.3, zorder=1)
    ax1.scatter(np.full(n_bd, POSITIONS[1])+j_bd, angle_bd, s=PT_SIZE, lw=PT_LW, color=BD_COLOR, alpha=0.3, zorder=1)
    ax1.scatter(np.full(n_sp, POSITIONS[2])+j_sp, angle_sp, s=PT_SIZE, lw=PT_LW, color='k', alpha=0.3, zorder=1)

    # mark median and 95th percentiles
    angle_medians = np.asarray([np.median(angle_gc), np.median(angle_bd), np.median(angle_sp)])
    angle_5 = np.asarray([np.percentile(angle_gc, 5), np.percentile(angle_bd, 5), np.percentile(angle_sp, 5)])
    angle_95 = np.asarray([np.percentile(angle_gc, 95), np.percentile(angle_bd, 95), np.percentile(angle_sp, 95)])
    ax1.plot(POSITIONS[:-1], angle_medians[:-1], '_k', markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=2, alpha=0.7)
    ax1.plot(POSITIONS[-1], angle_medians[-1], '_w', markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=2, alpha=0.7)
    ax1.vlines(POSITIONS, angle_5, angle_95, lw=LW_PCT, colors=['k', 'k', 'w'], linestyles='solid', zorder=2, alpha=0.7)

    # mark significance
    xvals = np.asarray([POSITIONS[1]-0.3, POSITIONS[1], POSITIONS[1]+0.3, 
                        np.mean(POSITIONS[:-1])-0.3, np.mean(POSITIONS[:-1]), np.mean(POSITIONS[:-1])+0.3])
    yvals = np.append(np.full(3, 0.83), np.full(3, 0.73))
    ax1.hlines([0.7, 0.8], [POSITIONS[0], POSITIONS[0]], POSITIONS[1:], 
                colors='k', linestyles='solid', lw=LW_SIG)
    ax1.scatter(xvals, yvals, c='k', marker='*', lw=0, s=SIG_SIZE)

    # set axis params
    ax1.tick_params(which='major', labelsize=7.5, pad=0.5)
    ax1.set_xticks(POSITIONS)
    ax1.set_xticklabels(['grid', 'border', 'spatial'], rotation=90)
    ax1.set_xlim(POSITIONS[0] - 1, POSITIONS[2] + 1)
    ax1.set_yticks([0, 0.3, 0.6])
    ax1.set_yticklabels([0, 0.3, 0.6])
    ax1.set_ylabel('spatial dissimilarity', fontsize=9, labelpad=1)

    return f, gs

def plot_fig5fg(data, mouse, session, cell_IDs, \
                binned_pos, FR_naive, FR_0, FR_1, \
                FR_sem, FR_0_sem, FR_1_sem):
    '''
    Examples of remapping cells.

    Params:
    ------
    cell_IDs : ndarray
        ID numbers for the example cells.
    binned_pos : ndarray
        position bin centers for each firing rate measurement
        shape (n_pos_bins,)
    FR_naive, FR_0, FR_1 : ndarray
        firing rate by position across maps or within each map
        shape (n_pos_bins, n_cells)
    FR_sem, FR_0_sem, FR_1_sem : ndarray
        SEM for the firing rate arrays; shape (n_pos_bins, n_cells)
    '''
    # get data
    d = data[mouse][session] 
    A = d['A']
    B = d['B']  
    cells = d['cells']
    W = d['kmeans']['W']
    map0_idx = d['idx']
    map1_idx = ~map0_idx

    # figure params
    gs = gridspec.GridSpec(5, 5, hspace=0.5, wspace=0.6)
    f = plt.figure(figsize=(4.8, 2)) 
    PT_SIZE = 1
    LW_MEAN = 1
    LW_SEM = 0.3

    for i, cell_ID in enumerate(cell_IDs):
        # set axes
        ax0 = plt.subplot(gs[:3, i])
        ax1 = plt.subplot(gs[3:4, i])
        ax2 = plt.subplot(gs[4:, i])
        
        # set color
        if i >= 3:
            cell_color = cell_colors[1]
        else:
            cell_color = cell_colors[0]
        
        # draw raster
        sdx = B[map0_idx, np.where(cells==cell_ID)[0][0]].astype(bool)
        ax0.scatter(A[map0_idx, 0][sdx], A[map0_idx, 2][sdx], 
                    color=cell_color, lw=0, s=PT_SIZE, alpha=.1)
        sdx = B[map1_idx, np.where(cells==cell_ID)[0][0]].astype(bool)
        ax0.scatter(A[map1_idx, 0][sdx], A[map1_idx, 2][sdx], 
                    color='k', lw=0, s=PT_SIZE, alpha=.1)
        
        # plot tuning curves with SEM
        sdx = (np.where(cells==cell_ID)[0][0]).astype(int)
        ax1.plot(FR_0[:, sdx], color=cell_color, lw=LW_MEAN, alpha=1)
        ax1.fill_between(binned_pos/2, FR_0[:, sdx] + FR_0_sem[:, sdx], 
                         FR_0[:, sdx] - FR_0_sem[:, sdx],
                         color=cell_color, linewidth=LW_SEM, alpha=0.2)
        ax1.plot(FR_1[:, sdx], color='k', lw=LW_MEAN, alpha=1)
        ax1.fill_between(binned_pos/2, FR_1[:, sdx] + FR_1_sem[:, sdx], 
                         FR_1[:, sdx] - FR_1_sem[:, sdx],
                         color='k', linewidth=LW_SEM, alpha=0.2)    
        
        # plot naive tuning curve with SEM
        sdx = (np.where(cells==cell_ID)[0][0]).astype(int)
        ax2.plot(FR_naive[:, sdx], 'k', lw=LW_MEAN, alpha=0.5)
        ax2.fill_between(binned_pos/2, FR_naive[:, sdx] + FR_sem[:, sdx], FR_naive[:, sdx] - FR_sem[:, sdx],
                         color='k', linewidth=LW_SEM, alpha=0.1)

        # format axes
        ax0.set_xlim((0, 400))
        ax0.set_xticks(np.arange(0, 425, 200))
        ax0.tick_params(labelbottom=False)
        ylim_ax = [0, np.max(A[:, 2])]
        ax0.set_ylim(ylim_ax[::-1])
        ax0.set_yticks(np.arange(0, 350, 100))
        ax0.set_title("cell {}".format(cell_ID), fontsize=10, pad=3)
         
        if i == 0:
            ax0.set_ylabel('trial number', fontsize=9, labelpad=1)
            ax2.set_ylabel('FR (Hz)', horizontalalignment='left', fontsize=9, labelpad=5)
            
        ax1.set_xlim([0, 200])
        ax1_lims = ax1.get_ylim()
        ax1.set_xticks(np.arange(0, 225, 100))
        ax1.tick_params(labelbottom=False)
        ax2.set_xlim([0, 200])
        ax2.set_ylim(ax1_lims)
        ax2.set_xticks(np.arange(0, 225, 100))
        ax2.set_xticklabels(np.arange(0, 450, 200))
        ax0.tick_params(which='major', labelsize=7.5, pad=0.5)
        ax1.tick_params(which='major', labelsize=7.5, pad=0.5)
        ax2.tick_params(which='major', labelsize=7.5, pad=0.5)

        if i == 1:
            ax2.set_xlabel('position (cm)', fontsize=9, labelpad=1)
        elif i == 3:
            ax2.set_xlabel('position (cm)', fontsize=9, labelpad=1, horizontalalignment='left', x=0.5)

    return f, gs

''' FIGURE 6 '''
def plot_fig6d(avg_scores, avg_shuffle, N_sessions, session_colors):
    N_sessions = avg_scores.shape[0]

    # figure params
    f, ax = plt.subplots(1, 1, figsize=(1.2, 1.2))
    DOT_SIZE = 10
    DOT_LW = 0.5
    BAR_SIZE = 7
    BAR_WIDTH = 2
    j = np.random.randn(avg_scores.shape[0]) * .08 # jitter

    # get the average scores for each case
    avg_0to0 = avg_scores[:, 0]
    avg_1to1 = avg_scores[:, 2]
    avg_rand = avg_scores[:, -1] # train/test both maps

    # plot connector lines
    for k, a0 in enumerate(avg_0to0): # plot connector lines
        a1 = avg_1to1[k]
        r = avg_rand[k]
        sh = avg_shuffle[k]
        x_vals = [1+j[k], 2+j[k], 3+j[k], 4+j[k]]
        y_vals = [a0, a1, r, sh]
        ax.plot(x_vals, y_vals, '-', color='xkcd:gray', lw=DOT_LW, zorder=1,  alpha=1)

    # train on map 0, test on map 0
    ax.scatter(np.full(N_sessions, 1)+j, avg_0to0,
               facecolors=session_colors, edgecolors='k', 
               s=DOT_SIZE, lw=DOT_LW, zorder=2, alpha=0.7)   
    
    # train on map 1, test on map 1
    ax.scatter(np.full(N_sessions, 2)+j, avg_1to1,
               facecolors=session_colors, edgecolors='k', 
               s=DOT_SIZE, lw=DOT_LW, zorder=2, alpha=0.7)
    
    # train on all, test on all
    ax.scatter(np.full(N_sessions, 3)+j, avg_rand,
               facecolors=session_colors, edgecolors='k', 
               s=DOT_SIZE, lw=DOT_LW, zorder=2, alpha=0.7)
    
    # shuffled "remap"
    ax.scatter(np.full(N_sessions, 4)+j, avg_shuffle,
               facecolors=session_colors, edgecolors='k', 
               s=DOT_SIZE, lw=DOT_LW, zorder=2, alpha=0.7)

    # plot means
    ax.plot(1, np.mean(avg_0to0), '_', c='k', markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=3)
    ax.plot(2, np.mean(avg_1to1), '_', c='k', markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=3)
    ax.plot(3, np.mean(avg_rand), '_', c='k', markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=3)
    ax.plot(4, np.mean(avg_shuffle), '_', c='k', markersize=BAR_SIZE, markeredgewidth=BAR_WIDTH, zorder=3)

    # labels and axes
    labels = ['map 1', 'map 2', 'both', 'shuffle']
    ax.set_xlim([0.5, 4.5])
    ax.set_ylim([-0.1, 1.1])
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels([0, 0.5, 1])
    ax.set_xticklabels(labels, rotation=90)
    ax.set_ylabel('avg model score', fontsize=9, labelpad=1)
    ax.tick_params(which='major', labelsize=7.5, pad=0.5)
    ax.set_title('train/test same', fontsize=10, pad=4)

    return f, ax

def plot_fig6efg(avg_scores, pct_75, pct_25, pct_best, session_colors):
    '''
    plot performance for models trained and tested on the same vs. different maps

    inputs are the mean, 75th, and 25th percentiles for each train/test case
        train/test on map 0/0, 0/1, 1/1, 1/0
    pct_best : across map performance relative to shuffle (0%) and within map (100%)
    '''
    N_sessions = avg_scores.shape[0]

    # figure params:
    gs = gridspec.GridSpec(1, 15, wspace=10000)
    f = plt.figure(figsize=(4.1, 1.2)) 
    DOT_SIZE = 10
    DOT_LW = 0.5
    UNITY_WIDTH = 1.25
    SEM_WIDTH = 0.8
    j = np.random.randn(pct_best.shape[0]) * .08 # jitter

    ax0 = plt.subplot(gs[:5])
    ax1 = plt.subplot(gs[6:-4])
    ax2 = plt.subplot(gs[-2:])

    # get the average scores for each case
    avg_0to0 = avg_scores[:, 0]
    avg_0to1 = avg_scores[:, 1]
    avg_1to1 = avg_scores[:, 2]
    avg_1to0 = avg_scores[:, 3]

    # plot 25th and 75th percentiles
    ax0.plot([avg_0to1, avg_0to1], [pct_75[:, 0], pct_25[:, 0]], '-k', lw=SEM_WIDTH, alpha=0.2, zorder=1)
    ax0.plot([pct_75[:, 1], pct_25[:, 1]], [avg_0to0, avg_0to0], '-k', lw=SEM_WIDTH, alpha=0.2, zorder=2)
    ax1.plot([pct_75[:, 2], pct_25[:, 2]], [avg_1to0, avg_1to0], '-k', lw=SEM_WIDTH, alpha=0.2, zorder=1)
    ax1.plot( [avg_1to1, avg_1to1], [pct_75[:, 3], pct_25[:, 3]], '-k', lw=SEM_WIDTH, alpha=0.2, zorder=2)

    # train on map 0, test on map 0 vs. test on map 1
    ax0.scatter(avg_0to1, avg_0to0,
                facecolors=session_colors, edgecolors='k', 
                s=DOT_SIZE, lw=DOT_LW, zorder=3, alpha=0.7)   
    
    # train on map 1, test on map 1 vs. test on map 0
    ax1.scatter(avg_1to1, avg_1to0,
                facecolors=session_colors, edgecolors='k', 
                s=DOT_SIZE, lw=DOT_LW, zorder=3, alpha=0.7)
    
    # relative performance vs. shuffle
    ax2.scatter(np.full(N_sessions, 1)+j, pct_best[:, 0]*100, 
                facecolors=session_colors, edgecolors='k', 
                s=DOT_SIZE, lw=DOT_LW, zorder=3, alpha=0.7)
    ax2.scatter(np.full(N_sessions, 1)-j, pct_best[:, 1]*100, 
                facecolors=session_colors, edgecolors='k', 
                s=DOT_SIZE, lw=DOT_LW, zorder=3, alpha=0.7)

    # plot unity lines + shuffle/best lines
    ax0.plot([0, 1], [0, 1], '--k', lw=UNITY_WIDTH, alpha=1)
    ax1.plot([0, 1], [0, 1], '--k', lw=UNITY_WIDTH, alpha=1)
    xlims = ax2.get_xlim()
    ax2.hlines([0, 100], xlims[0], xlims[1], linestyles='dotted', colors='k', lw=UNITY_WIDTH)

    # label everything
    ax0.set_title('train map 1', fontsize=10, pad=4)
    ax0.set_ylabel('test map 1', fontsize=9, labelpad=1)
    ax0.set_xlabel('test map 2', fontsize=9, labelpad=1)
    ax0.set_ylim([-0.05, 1.05])
    ax0.set_xlim([-0.05, 1.05])
    ax0.set_xticks([0, 0.5, 1])
    ax0.set_xticklabels([0, 0.5, 1])
    ax0.set_yticks([0, 0.5, 1])
    ax0.set_yticklabels([0, 0.5, 1])

    ax1.set_title('train map 2', fontsize=10, pad=4)
    ax1.set_ylabel('test map 1', fontsize=9, labelpad=1)
    ax1.set_xlabel('test map 2', fontsize=9, labelpad=1)
    ax1.set_ylim([-0.05, 1.05])
    ax1.set_xlim([-0.05, 1.05])
    ax1.set_xticks([0, 0.5, 1])
    ax1.set_xticklabels([0, 0.5, 1])
    ax1.set_yticks([0, 0.5, 1])
    ax1.set_yticklabels([0, 0.5, 1])

    ax2.set_xticks([1])
    ax2.set_title('train/test\ndifferent', fontsize=10, pad=4)
    ax2.set_ylabel('relative\nscore (%)', fontsize=9, labelpad=1)

    ax0.tick_params(which='major', labelsize=7.5, pad=0.5)
    ax1.tick_params(which='major', labelsize=7.5, pad=0.5)
    ax2.tick_params(labelbottom=False, bottom=False, which='major', labelsize=7.5, pad=0.5)

    return f, gs

''' FIGURE 7 '''

''' FIGURE 8 '''
def plot_fig8a(stable_speeds, remap_speeds, mouse_ID):
    # figure params
    f, ax = plt.subplots(1, 1, figsize=(1.2, 1.2))
    UNITY_W = 1
    PT_SIZE = 15
    PT_LW = 0.5

    # stable vs. remaps
    ax.scatter(stable_speeds, remap_speeds, s=PT_SIZE, 
               facecolors=cp_color, edgecolors='k', lw=PT_LW, alpha=0.7)

    # plot unity
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    min_speed = np.min([xlims[0], ylims[0]])
    max_speed = np.max([xlims[1], ylims[1]])
    ax.plot([min_speed, max_speed], [min_speed, max_speed], 
            '--k', lw=UNITY_W, alpha=1)

    # labels etc
    ax.set_xticks([25, 50])
    ax.set_yticks([25, 50])
    ax.tick_params(which='major', labelsize=7.5, pad=0.5)
    ax.set_xlabel('stable block\nspeed (cm/s)', fontsize=9, labelpad=1)
    ax.set_ylabel('remap trials\nspeed (cm/s)', fontsize=9, labelpad=1)
    ax.set_title(f'mouse {mouse_ID}', fontsize=10, pad=3)

    return f, ax

def plot_fig8b(mice, sessions, \
                med_speed_stable, med_speed_remap, \
                sem_stable, sem_remaps, \
                print_speeds=True):
    # figure params
    f, ax = plt.subplots(1, 1, figsize=(1.2, 1.2))
    DOT_SIZE = 15
    PT_LW = 0.5
    UNITY_WIDTH = 1
    SEM_WIDTH = 1

    i = -1
    if print_speeds:
        print('average speeds per session (0 indicates excluded sessions)')
    for m, session in zip(mice, sessions):
        i += 1
        
        # get data
        avg_stable = med_speed_stable[i]
        avg_remap = med_speed_remap[i]
        sem_s = sem_stable[i]
        sem_r = sem_remaps[i]

        # if avg_stable > 0, session has enough remap events to be included
        idx = avg_stable > 0
        
        # show data
        if print_speeds:
            print(f'{m} stable block speeds: {avg_stable}')
            print(f'{m} remap trial speeds: {avg_remap}\n')
        
        # choose color for each mouse
        if m in ['Pisa', 'Hanover', 'Calais']:
            session_color = cp_color
        else:
            session_color = cr_color
        
        # plot mean and sem
        ax.vlines(avg_stable[idx], avg_remap[idx] - sem_r[idx], avg_remap[idx] + sem_r[idx], 
                  colors='k', lw=SEM_WIDTH, linestyles='solid', alpha=0.2, zorder=1)
        ax.hlines(avg_remap[idx], avg_stable[idx] - sem_s[idx], avg_stable[idx] + sem_s[idx], 
                  colors='k', lw=SEM_WIDTH, linestyles='solid', alpha=0.2, zorder=1)
        ax.scatter(avg_stable[idx], avg_remap[idx], s=DOT_SIZE, 
                   facecolors=session_color, edgecolors='k', lw=PT_LW, zorder=2)

    # set ticks
    ax.set_xticks([25, 50])
    ax.set_yticks([25, 50])
    ax.tick_params(which='major', labelsize=7.5, pad=0.5)
    
    # plot unity
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    min_speed = np.min([xlims[0], ylims[0]])
    max_speed = np.max([xlims[1], ylims[1]])
    plt.plot([min_speed, max_speed], [min_speed, max_speed], '--k', lw=UNITY_WIDTH, alpha=1)

    # label plot
    ax.set_xlabel('stable block\nspeed (cm/s)', fontsize=9, labelpad=1)
    ax.set_ylabel('remap trials\nspeed (cm/s)', fontsize=9, labelpad=1)
    ax.set_title('all 2-map mice', fontsize=10, pad=3)

    return f, ax

def plot_fig8c(mouse_ID, session, ex_blocks, \
                stable_idx, speed_stable, pos_stable):
    n_trials = speed_stable.shape[0]

    # figure params
    gs  = gridspec.GridSpec(15, 2, hspace=0, wspace=0.2)
    f = plt.figure(figsize=(2.5, 3))
    LW_SPEED = 1.5
    MAX_SPEED = 90

    # plot speed and distance to cluster for 8 stable trials
    xvals = np.arange(0, 4*400, 5)                       
    for r, i in enumerate(ex_blocks):
        # set subplots
        if r == 0:
            ax01 = plt.subplot(gs[:2, :])
            ax11 = plt.subplot(gs[2, :])
            ax01.set_title(f'mouse {mouse_ID}, session {session}', \
                           fontsize=10, pad=5)
        else:
            ax01 = plt.subplot(gs[-3:-1, :])
            ax11 = plt.subplot(gs[-1, :])
        
        # get trial numbers
        X_LABELS = []
        j = -1
        for t in range(12):
            if np.isin(t, np.asarray([1, 3, 5, 7])):
                X_LABELS.append(f'{stable_idx[i] + j}')
                j += 1
            else:
                X_LABELS.append('')
        
        # plot running speed
        ax01.plot(xvals, speed_stable[i], '-k', lw=LW_SPEED)
        ax01.set_ylim([0, MAX_SPEED])
        ax01.set_xticks(np.arange(0, 4*400+5, 200))
        ax01.set_xlim([0, 4*400])
        ax01.tick_params(labelbottom=False, which='major', labelsize=7.5, pad=0.5)

        # plot distance to midpoint
        ax11.imshow(pos_stable[None, i, :], clim=[-1, 1], aspect='auto', cmap='binary')
        ax11.set_xticks(np.arange(0, 4*80+5, 40))
        ax11.set_xticklabels(X_LABELS)
        ax11.set_xlim([0, 4*80])
        ax11.tick_params(labelleft=False, which='major', labelsize=7.5, pad=3)
        ax11.tick_params(axis = "y", which = "both", left = False)
        
    # zoom in on an example trial from each block
    xvals = np.arange(0, 400, 5)                       
    for r, i in enumerate(to_plot):
        if r == 0:
            ax01 = plt.subplot(gs[5:8, 1])
            ax11 = plt.subplot(gs[8:9, 1])
            idx = slice(-80, None)
            ax01.tick_params(labelleft=False)
            ax11.set_xlabel('position (cm)', fontsize=9, labelpad=1, 
                            horizontalalignment='right', x=0.5)             
        else:
            ax01 = plt.subplot(gs[5:8, 0])
            ax11 = plt.subplot(gs[8:9, 0])
            idx = slice(80)
            ax01.set_ylabel('running speed (cm/s)', fontsize=9, labelpad=3, 
                            verticalalignment='bottom', y=0.3)
        
        # plot running speed
        ax01.plot(xvals, speed_stable[i, idx], '-k', lw=LW_SPEED)
        ax01.set_ylim([0, MAX_SPEED])
        ax01.set_yticks([0, 40, 80])
        ax01.set_xticks(np.arange(0, 405, 200))
        ax01.set_xlim([0, 400])
        ax01.tick_params(labelbottom=False, which='major', labelsize=7.5, pad=0.5)

        # plot dd_pos
        ax11.imshow(pos_stable[None, i, idx], clim=[-1, 1], aspect='auto', cmap='binary')
        ax11.set_xticks(np.arange(0, 80+5, 40))
        ax11.set_xticklabels(np.arange(0, 405, 200))
        ax11.set_xlim([0, 80])
        ax11.tick_params(labelleft=False, which='major', labelsize=7.5, pad=0.5)
        ax11.tick_params(axis="y", which ="both", left=False)

        return f, gs
