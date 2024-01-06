import sys, os

sys.path.append(os.path.abspath("../plot_common"))
from common import set_figure, fig_size
from dataloader import load_data
from palettes import get_robustness_plot_names, get_color, get_marker

import numpy as np
import sys
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as ticker

########################################################################
########################################################################
names = get_robustness_plot_names("p2p1")
colors = [get_color(n) for n in names]
markers = [get_marker(n) for n in names]

ms = 3

########################################################################
# LOAD DATA
########################################################################
COMM_PREAM0 = "./th/"
uzawa_dir = "uzawa/"
COMM_PREAM1 = COMM_PREAM0
p2p1_dir = "amg_p2p1/"
isop2p1_dir = "amg_isop2p1/"

all_problems = dict.fromkeys(
    [
        "structured/2D_bfs/",
        "unstructured/2D/",
        "structured/3D/",
        "unstructured/3D/",
    ],
    None,
)

for problem in all_problems.keys():
    # the order of paths below needs to patch the order of names
    paths = [
        COMM_PREAM0 + uzawa_dir + problem,
        COMM_PREAM1 + p2p1_dir + problem,
        COMM_PREAM1 + isop2p1_dir + "ho/" + problem,
        COMM_PREAM1 + isop2p1_dir + "lo/" + problem,
        COMM_PREAM1 + isop2p1_dir + "hlo/" + problem,
    ]

    all_problems[problem] = load_data(names, paths)


########################################################################
# PLOT DATA
########################################################################
linewidth = 1

ncols = len(list(all_problems.keys()))  # 4
nrows = 3
fs = fig_size.singlefull
set_figure(width=fs["width"], height=0.8 * fs["width"])
fig, axs_all = plt.subplots(nrows, ncols, sharey="row", sharex="col")

# Ylabels (titles of plots - one per row)
for i, title in enumerate(["iterations", "rel. time", "rel. time per iter."]):
    axs_all[i][0].set_ylabel(title)

########################################################################
# Iterations
axs = axs_all[0]
# Plots should share the same data ranges (x is known in advance)
ymin = 1e4
ymax = -1
for i, ((prob_path, data_dict), ax) in enumerate(zip(all_problems.items(), axs)):
    for j, (k, v) in enumerate(data_dict["residuals"].items()):
        size, resid_hist = list(v.keys()), list(v.values())
        iters = [len(it) for it in resid_hist]

        ax.semilogx(
            size,
            iters,
            label=k,
            linestyle="-",
            color=colors[j],
            marker=markers[j],
            markersize=ms,
            linewidth=linewidth,
            clip_on=False,
            zorder=10,
        )

        ymax = max(ymax, max(iters))
        ymin = min(ymin, min(iters))

    # y-ticks
    ax.set_yticks([20, 30, 40, 50, 60, 70, 80])
    ax.yaxis.set_minor_locator(MultipleLocator(2))

    msh_type, dim = prob_path.split("_")[0].split("/")[:2]
    msh_type = msh_type.capitalize()
    ax.set_title("%s %s" % (msh_type, dim))

for ax in axs:
    ax.set_ylim((ymin - 1, ymax + 1))
    ax.set_box_aspect(1)

########################################################################
# Time to convergence
axs = axs_all[1]

ymin = 1e4
ymax = -1
for (prob_path, data_dict), ax in zip(all_problems.items(), axs):
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

        ax.semilogx(
            dofs,
            rel_time,  # label=k,
            linestyle="-",
            color=colors[j],
            marker=markers[j],
            markersize=ms,
            linewidth=linewidth,
            clip_on=False,
            zorder=10,
        )
        ymax = max(ymax, max(rel_time))
        ymin = min(ymin, min(rel_time))

    # y-ticks
    ax.set_yticks([0.5, 1, 1.5, 2, 2.5, 3, 3.5])
    ax.yaxis.set_minor_locator(MultipleLocator(0.125))

for ax in axs:
    ax.set_ylim((ymin * 0.85, ymax * 1.05))
    ax.set_box_aspect(1)
########################################################################
# Time per iter
axs = axs_all[2]
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

        rel_time = (solve_time / iters) / ref_time
        ax.semilogx(
            dofs,
            rel_time,
            label=k,
            linestyle="-",
            color=colors[j],
            marker=markers[j],
            markersize=ms,
            linewidth=linewidth,
            clip_on=False,
            zorder=10,
        )

        ymax = max(ymax, max(rel_time))
        ymin = min(ymin, min(rel_time))

    # y-ticks
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

for ax in axs:
    ax.set_ylim((ymin * 0.85, ymax * 1.025))
    ax.set_box_aspect(1)

#######################################################
xmin = np.ones((len(axs),)) * 1e100
xmax = np.zeros((len(axs),))
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
    if r < 2:
        ax.tick_params(labelbottom=False)

    # fix x-axis alignment
    for (i, data_dict), data in zip(
        enumerate(all_problems.values()), all_problems.values()
    ):
        ndofs = np.array([int(i) for i in data_dict["timings"][names[0]].keys()])
        xmin[i] = min(xmin[i], np.min(ndofs))
        xmax[i] = max(xmax[i], np.max(ndofs))

for axs in axs_all:
    for i, ax in enumerate(axs):
        ax.set_xlim((xmin[i] * 0.9, xmax[i] * 1.05))


axs_all[0, 0].legend(
    loc="lower left",
    bbox_to_anchor=(0.8, 1.2, 2.8, 0.2),
    mode="expand",
    borderaxespad=0,
    ncol=3,
)  # ncols)

axs_all[2, 0].set_xlabel("\# DoFs")
axs_all[2, 1].set_xlabel("\# DoFs")
axs_all[2, 2].set_xlabel("\# DoFs")
axs_all[2, 3].set_xlabel("\# DoFs")

if "--savefig" in sys.argv:
    plt.savefig("fig_6.pdf")
else:
    plt.show()
