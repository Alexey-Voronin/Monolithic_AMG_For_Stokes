import sys, os

sys.path.append(os.path.abspath("../../plot_common"))
from dataloader import load_data
from common import set_figure, fig_size
from palettes import get_robustness_plot_names, get_color, get_marker

import numpy as np
import sys
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as ticker

########################################################################
# Define AMG labels and attributes (colors, etc.)
# If you change their order, you also need to change the order of
# paths[ ] below.
########################################################################
names = list(get_robustness_plot_names("p2p1disc"))
colors = [get_color(n) for n in names]
markers = [get_marker(n) for n in names]
names = ["BTP", "DC-all(Vanka)", "DC-all(Vanka($\ell=0$),LSC-DGS($\ell>0$))"]
markers[-1] = "*"  # '|'

ms = 3

########################################################################
# LOAD DATA
########################################################################
COMM_PREAM = "./"  # data_bf/robustness_problems/sv/"
PREAM_HO = COMM_PREAM + "uzawa/"
PREAM_LO_0 = COMM_PREAM + "defect_correction/vanka/"
# PREAM_LO_1  = COMM_PREAM + "defect_correction/lsc_dgs/"
PREAM_LO_2 = COMM_PREAM + "defect_correction/lsc_dc/"

for i in range(1):
    names.append(names[-1])
    colors.append(colors[-1])
    markers.append(markers[-1])
names[-2] = f"{names[-2]}(Vanka)"
names[-1] = f"{names[-1]}(Vanka($\\ell=0$),LSC-DC($\\ell>0$))"
linestyle = ["solid", "solid", "dashed"]

all_problems = dict.fromkeys(
    [
        "structured/2D_bfs/",
        "unstructured/2D/",
    ],
    None,
)

for problem in all_problems.keys():
    paths = [
        PREAM_HO + problem,
        PREAM_LO_0 + problem,
        # PREAM_LO_1 + problem,
        PREAM_LO_2 + problem,
    ]

    all_problems[problem] = load_data(names, paths)

########################################################################
# PLOT DATA
########################################################################
linewidth = 1

nrows = len(list(all_problems.keys()))  # 2
ncols = 3
fs = fig_size.singlefull
set_figure(width=fs["width"], height=0.6 * fs["width"])
fig, axs_all = plt.subplots(nrows, ncols, sharex="col")  # sharey='row',

# Ylabels (titles of plots - one per row)
for i, title in enumerate(["iterations", "relative time", "rel. time per iter."]):
    axs_all[0][i].set_title(title)
########################################################################
# Iterations
axs = axs_all[:, 0]
# Plots should share the same data ranges (x is known in advance)
ymin = 1e4
ymax = -1
for i, ((prob_path, data_dict), ax) in enumerate(zip(all_problems.items(), axs)):
    for j, (k, v) in enumerate(data_dict["residuals"].items()):
        size, resid_hist = list(v.keys()), list(v.values())
        iters = np.array([len(it) for it in resid_hist])

        # names[0] has more data points than need
        # if prob_path.split('/')[0] == 'structured' and k == names[0]:
        #    iters = iters[:-1]
        #    size = size[:-1]

        ax.semilogx(
            size,
            iters,
            label=k,
            # linestyle='-',
            color=colors[j],
            marker=markers[j],
            linestyle=linestyle[j],
            markersize=ms,
            linewidth=linewidth,
            clip_on=False,
            zorder=10,
        )

        ymax = max(ymax, max(iters))
        ymin = min(ymin, min(iters))
    # y-ticks
    ax.yaxis.set_minor_locator(MultipleLocator(5))
    #
    msh_type, dim = prob_path.split("_")[0].split("/")[:2]
    msh_type = msh_type.capitalize()
    ax.set_ylabel("%s %s" % (msh_type, dim))

for ax in axs:
    ax.set_ylim((ymin - 1, ymax + 1))
    ax.set_box_aspect(1)

########################################################################
# Time to convergence
axs = axs_all[:, 1]

ymin = 1e4
ymax = -1
for i, ((prob_path, data_dict), ax) in enumerate(zip(all_problems.items(), axs)):
    ho_id = list(data_dict["timings"].keys())[0]
    ref_dofs = np.array(list(data_dict["timings"][ho_id].keys()))
    ref_time = np.array(
        [d["mg:solve"]["0"] for d in data_dict["timings"][ho_id].values()]
    )

    for j, (k, v) in enumerate(data_dict["timings"].items()):
        dofs = np.array(list(v.keys()))
        solve_time = np.array([d["mg:solve"]["0"] for d in v.values()])
        # since the relative timings need to be computed and some AMG types
        # have run larger problems we should only use the same/matching problem
        # sizes for each AMG type
        r_idx = np.in1d(ref_dofs, dofs)
        idx = np.in1d(dofs, ref_dofs)
        rel_time = solve_time[idx] / ref_time[r_idx]
        # names[0] has more data points than need
        # if prob_path.split('/')[0] == 'structured' and k == names[0]:
        #    rel_time = rel_time[:-1]
        #    dofs = dofs[idx][:-1]

        ax.semilogx(
            dofs,
            rel_time,
            label=k,
            # linestyle='-',
            color=colors[j],
            marker=markers[j],
            linestyle=linestyle[j],
            markersize=ms,
            linewidth=linewidth,
            clip_on=False,
            zorder=10,
        )
        ymax = max(ymax, max(rel_time))
        ymin = min(ymin, min(rel_time))

    # y-ticks
    ax.set_yticks([0.4, 0.6, 0.8, 1.0, 1.2])
    ax.yaxis.set_minor_locator(MultipleLocator(0.05))

for ax in axs:
    ax.set_ylim((ymin * 0.95, ymax * 1.025))
    ax.set_box_aspect(1)
########################################################################
# Time per iteration
axs = axs_all[:, 2]
ymin = 1e4
ymax = -1
for (prob_path, data_dict), ax in zip(all_problems.items(), axs):
    ho_id = names[0]
    ref_dofs = np.array(list(data_dict["timings"][ho_id].keys()))
    ref_time = np.array(
        [d["mg:solve"]["0"] for d in data_dict["timings"][ho_id].values()]
    )

    resid_hist = list(data_dict["residuals"][ho_id].values())
    iters = np.array([len(it) for it in resid_hist])
    ref_time = ref_time / iters

    for j, ((k, v), resid_hist) in enumerate(
        zip(data_dict["timings"].items(), data_dict["residuals"].values())
    ):
        resid_hist = list(resid_hist.values())
        iters = np.array([len(it) for it in resid_hist])
        dofs = np.array(list(v.keys()))
        solve_time = np.array([d["mg:solve"]["0"] for d in v.values()])

        r_idx = np.in1d(ref_dofs, dofs)
        idx = np.in1d(dofs, ref_dofs)
        rel_time = (solve_time[idx] / iters[idx]) / ref_time[r_idx]
        # names[0] has more data points than need
        # if prob_path.split('/')[0] == 'structured' and k == names[0]:
        #    rel_time = rel_time[:-1]
        #    dofs = dofs[idx][:-1]

        ax.semilogx(
            dofs,
            rel_time,
            label=k,
            # linestyle='-',
            color=colors[j],
            marker=markers[j],
            linestyle=linestyle[j],
            markersize=ms,
            linewidth=linewidth,
            clip_on=False,
            zorder=10,
        )

        ymax = max(ymax, max(rel_time))
        ymin = min(ymin, min(rel_time))

    # y-ticks
    ax.set_yticks([1.0, 2.0, 3.0, 4.0])
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

for ax in axs:
    ax.set_ylim((ymin * 0.85, ymax * 1.05))
    ax.set_box_aspect(1)

#######################################################
xmin = 1e100
xmax = 0  # np.zeros((3,))
for r, axs in enumerate(axs_all):
    for ax in axs:
        # x-ticks
        nticks = 9
        maj_loc = ticker.LogLocator(numticks=nticks)
        min_loc = ticker.LogLocator(subs="all", numticks=nticks)
        ax.xaxis.set_major_locator(maj_loc)
        ax.xaxis.set_minor_locator(min_loc)
        ax.tick_params(axis="x", which="major")
        ax.tick_params(axis="x", which="minor")
        ax.grid(None)

        # if r < 2:
        #    ax.tick_params(labelbottom=False)

    # fix x-axis alignment
    for (i, data_dict), data in zip(
        enumerate(all_problems.values()), all_problems.values()
    ):
        ndofs = np.array([int(i) for i in data_dict["timings"][names[0]].keys()])
        xmin = min(xmin, np.min(ndofs))
        # xmax[i] = max(xmax[i], np.max(ndofs))
        # i=1/column=1 -> unstructured grids, one fewer prob size than expected
        xmax = max(xmax, np.max(ndofs))  # [:-1]) if i == 1 else np.max(ndofs))

for axs in axs_all:
    for j, ax in enumerate(axs):
        ax.set_xlim((xmin * 0.9, xmax * 1.1))

# axs_all[0,1].legend(loc='lower left',
#                    bbox_to_anchor=(0.0, 1.1, 1.0, 0.2),
#                    ncol=ncols)
axs_all[0, 0].legend(
    loc="lower left",
    bbox_to_anchor=(0.2, 1.2, 3.6, 0.2),  # 0.8,  # 2.8,
    mode="expand",
    borderaxespad=0,
    ncol=3,
)  # ncols)
# axs_all[0,1].legend(#custom_lines, names,
#                    loc='lower left',
#                    bbox_to_anchor=(-0.5, 1.2, 1.2, 0.2),
#                    mode="expand",
#                    borderaxespad=0,
#                    ncol=2)
axs_all[1, 0].set_xlabel("\# DoFs")
axs_all[1, 1].set_xlabel("\# DoFs")
axs_all[1, 2].set_xlabel("\# DoFs")

if "--savefig" in sys.argv:
    plt.savefig("p2p1disc_results.pdf")
else:
    plt.show()
