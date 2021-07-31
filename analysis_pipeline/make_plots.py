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


''' FIGURE 1 '''

''' FIGURE 2 '''
def plot_fig2b_f(data, mouse, session, cell_IDs, \
                    f, gs, PT_SIZE=0.2):
    '''
    Make panels B-F

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

''' FIGURE 4 '''

''' FIGURE 5 '''

''' FIGURE 6 '''

''' FIGURE 7 '''

''' FIGURE 8 '''