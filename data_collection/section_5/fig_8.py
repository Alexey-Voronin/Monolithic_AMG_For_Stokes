import numpy as np
import itertools

import sys, os
sys.path.append(os.path.abspath("../plot_common"))
from common import set_figure, fig_size
from dataloader import load_data
from palettes import get_hero_plot_names, get_color, get_marker

# plotting
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, FormatStrFormatter,
                               AutoMinorLocator)
import matplotlib.ticker as ticker

marker_size = 3

######################################################
# READ IN DATA
# P2/P1
COMM_PREAM  = "th/split_artery/"
p2p1_dir    = "amg_p2p1/unstructured/3D/"
isop2p1_dir = "amg_isop2p1/ho/unstructured/3D/"
paths       = [COMM_PREAM+p2p1_dir, COMM_PREAM+isop2p1_dir]

p2p1_names    = get_hero_plot_names("p2p1")
p2p1_problems = load_data(p2p1_names, paths)
# P2/P1disc
COMM_PREAM  = "sv/"
p2p1_dir    = "uzawa/unstructured/2D/"
isop2p1_dir = "defect_correction/unstructured/2D/"
paths     = [COMM_PREAM+p2p1_dir, COMM_PREAM+isop2p1_dir]
p2p1disc_names    = get_hero_plot_names("p2p1disc")
p2p1disc_problems = load_data(p2p1disc_names, paths)

########################################################################
# PLOT DATA
########################################################################
linewidth = 1

ncols = 3
nrows = 2
fs    = fig_size.singlefull
set_figure(width=fs['width'], height=0.65*fs['width'])
fig, AXS_ALL = plt.subplots(nrows, ncols,
                            sharex='col')

xmin = np.ones((nrows,))*1e100
xmax = np.zeros((nrows,))
for j, (data_dict, disc_name, names) in enumerate(
                                 zip([p2p1_problems, p2p1disc_problems],
                                     ['3D Artery', '2D Airfoil'],
                                     [p2p1_names, p2p1disc_names]
                                     )):
    print(names)
    #names      = attr_dict['names']
    axs_all = AXS_ALL[j,:]

    if j == 0:
        # Ylabels (titles of plots - one per row)
        for jj, title in enumerate([
                                    'iterations',
                                    'relative time',
                                    'rel. time per iter.']):
            axs_all[jj].set_title(title)

    # Ylabels (titles of plots - one per row)
    axs_all[0].set_ylabel(disc_name)

    #######################################################
    for r, ax in enumerate(axs_all):
        # x-ticks
        nticks = 9
        maj_loc = ticker.LogLocator(numticks=nticks)
        min_loc = ticker.LogLocator(subs='all', numticks=nticks)
        #ax.xaxis.set_major_locator(maj_loc)
        #ax.xaxis.set_minor_locator(min_loc)
        ax.tick_params(axis='x', which='major')
        ax.tick_params(axis='x', which='minor')
        ax.grid(None)

    axs_row = axs_all
    ########################################################################
    # Iterations
    i  = 0
    ax = axs_row[i]
    # Plots should share the same data ranges (x is known in advance)
    for (k,v) in data_dict['residuals'].items():
        size, resid_hist = np.array(list(v.keys())), list(v.values())
        iters      = np.array([len(it) for it in resid_hist])

        startidx = 0
        if j==0:
            startidx = 2
        ax.semilogx(size[startidx:], iters[startidx:], label=k,
                    linestyle='-',
                    marker=get_marker(k),
                    color=get_color(k),
                    markersize=marker_size,
                    linewidth=linewidth,
                    clip_on=False, zorder=10)

    ax.set_xlim(5e4, 1.1e7)
    ax.set_xticks([1e5, 1e6, 1e7])

    # y-ticks
    if j == 0:
        ax.yaxis.set_minor_locator(MultipleLocator(1))
    else: # SV results have more max iters
        ax.yaxis.set_minor_locator(MultipleLocator(5))

    #ax.set_box_aspect(1)

    ########################################################################
    # Time to convergence
    i  = 1
    ax = axs_row[i]
    # Plots should share the same data ranges (x is known in advance)
    ho_id = list(data_dict['timings'].keys())[0]
    ref_dofs = np.array(list(data_dict['timings'][ho_id].keys()))
    ref_time = np.array([d['mg:solve']['0'] for d in
                         data_dict['timings'][ho_id].values()])

    for (k,v) in data_dict['timings'].items():

        dofs       = np.array(list(v.keys()))
        solve_time = np.array([d['mg:solve']['0'] for d in v.values()])
        dofs       = dofs
        rel_time   = solve_time/ref_time

        startidx = 0
        if j==0:
            startidx = 2
        ax.semilogx(dofs[startidx:], rel_time[startidx:], label=k,
                    linestyle='-',
                    marker=get_marker(k),
                    color=get_color(k),
                    markersize=marker_size,
                    linewidth=linewidth,
                    clip_on=False, zorder=10)

    ax.set_xlim(5e4, 1.1e7)
    ax.set_xticks([1e5, 1e6, 1e7])

    # y-ticks
    #ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

    #ax.set_box_aspect(1)
    ########################################################################
    # Timer per iteration
    i  = 2
    ax = axs_row[i]
    # Plots should share the same data ranges (x is known in advance)
    ho_id    = names[0]
    ref_dofs = np.array(list(data_dict['timings'][ho_id].keys()))
    ref_time = np.array([d['mg:solve']['0'] for d in
                         data_dict['timings'][ho_id].values()])

    resid_hist = list(data_dict['residuals'][ho_id].values())
    iters      = np.array([len(it) for it in resid_hist])
    ref_time   = ref_time/iters

    for (k,v), resid_hist in zip(data_dict['timings'].items(), data_dict['residuals'].values()):

        resid_hist = list(resid_hist.values())
        iters      = np.array([len(it) for it in resid_hist])
        dofs       = np.array(list(v.keys()))
        solve_time = np.array([d['mg:solve']['0'] for d in v.values()])
        rel_time   = (solve_time/iters)/ref_time

        startidx = 0
        if j==0:
            # skip really small problems
            startidx = 2

        ax.semilogx(dofs[startidx:], rel_time[startidx:], label=k,
                    linestyle='-',
                    marker=get_marker(k),
                    color=get_color(k),
                    markersize=marker_size,
                    linewidth=linewidth,
                    clip_on=False, zorder=10)


    ax.set_xlim(5e4, 1.1e7)
    ax.set_xticks([1e5, 1e6, 1e7])

    # y-ticks
    #ax.set_yticks([0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

from matplotlib.lines import Line2D
names = ['HO-AMG', 'DC-skip1', 'Uzawa', 'DC-all']
custom_lines = [ Line2D([0], [0],
                        color=get_color(k),
                        marker=get_marker(k),
                        lw=1, markersize=marker_size) for k in names]
AXS_ALL[0,0].legend(custom_lines, names,
                    loc='lower left',
                    bbox_to_anchor=(0.3, 1.2, 2.8, 0.2),
                    mode="expand", borderaxespad=0,
                    ncol=4)

AXS_ALL[1, 0].set_xlabel("\# DoFs")
AXS_ALL[1, 1].set_xlabel("\# DoFs")
AXS_ALL[1, 2].set_xlabel("\# DoFs")

if '--savefig' in sys.argv:
    plt.savefig('fig_8.pdf')
else:
    plt.show()
