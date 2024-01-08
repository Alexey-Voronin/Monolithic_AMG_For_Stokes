# from dataloader import load_data
import sys, os

sys.path.append(os.path.abspath("../../../plot_common"))
from common import set_figure, fig_size
from palettes import get_sensitivity_plot_names, get_color, get_marker

import numpy as np
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
names = get_sensitivity_plot_names("p2p1")
colors = [get_color(n) for n in names]
markers = [get_marker(n) for n in names]
markersize = 3

########################################################################
# LOAD DATA
########################################################################
PREAM = "./"  # "../../../../data_collection/opt_params/th_stokes/heat_map/"
data_paths = {
    names[0]: {
        "2D Structured": PREAM + "eta_ho/structured/2D_bfs/",
        "2D Unstructured": PREAM + "eta_ho/unstructured/2D/",
        "3D Structured": PREAM + "eta_ho/structured/3D/",
        "3D Unstructured": PREAM + "eta_ho/unstructured/3D/",
    },
    names[1]: {
        "2D Structured": PREAM + "eta_hlo/structured/2D_bfs/",
        "2D Unstructured": PREAM + "eta_hlo/unstructured/2D/",
        "3D Structured": PREAM + "eta_hlo/structured/3D/",
        "3D Unstructured": PREAM + "eta_hlo/unstructured/3D/",
    },
}

opt_values_chosen = {
    names[0]: {
        "2D Structured": 1.00,
        "2D Unstructured": 1.00,
        "3D Structured": 0.60,
        "3D Unstructured": 1.00,
    },
    names[1]: {
        "2D Structured": 1.00,
        "2D Unstructured": 1.00,
        "3D Structured": 1.10,
        "3D Unstructured": 1.06,
    },
}


########################################################################
# Helper Function
########################################################################
def loaddata(PATH):
    cf = np.load(PATH + "cf_data_1.npy")
    iters = np.load(PATH + "iters_data_1.npy")
    param = np.load(PATH + "param_ranges.npy", allow_pickle=True).tolist()
    ETAs = param["\\eta"][0]

    return cf, iters, ETAs


def print_opt_values(rho, iters, ETAs):
    """Optimal parameters based on the smallest number of iterations is more
    accurate than based solely on the convergence factor.
    """
    iters_min = iters.min()
    idx_opt = np.where(iters == iters_min)[0]

    cf_min_idx = np.argmin(rho[idx_opt])
    eta_id = idx_opt[cf_min_idx]
    eta_opt = ETAs[eta_id]
    rho_min = rho[eta_id]
    iters_opt = iters[eta_id]

    return eta_opt, iters_opt, rho_min


########################################################################
# PLOT DATA
########################################################################

ncols = 4  # len(list(data_paths.keys())) # 2
nrows = 1
fs = fig_size.singlefull
set_figure(width=fs["width"], height=0.3 * fs["width"])
fig, AXES = plt.subplots(nrows, ncols, sharey="row", sharex="col")

cf_min = [1e4] * ncols
cf_max = [-1] * ncols
iter_min = [1e4] * ncols
iter_max = [-1] * ncols
indep_var_min = [1e4] * ncols
indep_var_max = [-1] * ncols
for pid, mg in enumerate(data_paths.keys()):
    data = data_paths[mg]
    for i, (prob_type, PATH) in enumerate(data.items()):
        axes = AXES[i]

        cf, iters, ETAs = loaddata(PATH)
        eta_opt, iters_opt, rho_min = print_opt_values(cf, iters, ETAs)

        indep_var = ETAs
        var_name = "$\eta$"  # list(param.keys())[0]
        iters0 = np.ravel(iters)
        axes.plot(
            indep_var,
            iters,
            label=mg,
            linestyle="-",
            lw=1,
            clip_on=False,
            color=colors[pid],
            marker=markers[pid],
            markersize=markersize,
            markevery=5,
        )

        eta_opt_reported = opt_values_chosen[mg][prob_type]

        idx = np.argmin(np.abs(ETAs - eta_opt_reported))
        iters_opt = iters[idx]
        axes.plot(
            eta_opt_reported,
            iters_opt,
            marker=markers[pid],
            markersize=3,
            markerfacecolor="w",
            markeredgewidth=1.5,
            linestyle="",
            color="k",
        )

        axes.set_title(prob_type)

        cf_min[i] = min(cf_min[i], min(cf))
        cf_max[i] = max(cf_max[i], max(cf))
        iter_min[i] = min(iter_min[i], min(iters0))
        iter_max[i] = max(iter_max[i], max(iters0))
        indep_var_min[i] = min(min(indep_var), indep_var_min[i])
        indep_var_max[i] = max(max(indep_var), indep_var_max[i])

        if i == 0:
            axes.set_ylabel("iterations")
        axes.set_xlabel("$\eta$")

handles, labels = axes.get_legend_handles_labels()
AXES[1].legend(
    handles=[handles[0]],
    labels=[labels[0]],
    bbox_to_anchor=(0.0, 1.2, 1.0, 0.2),
    loc="upper center",
    ncol=1,
)
AXES[2].legend(
    handles=[handles[1]],
    labels=[labels[1]],
    bbox_to_anchor=(0.0, 1.2, 1.0, 0.2),
    loc="upper center",
    ncol=1,
)

for j, ax in enumerate(AXES):
    ax.set_ylim((min(iter_min) - 1, max(iter_max) + 1))
    ax.xaxis.set_minor_locator(MultipleLocator(0.025))
    ax.grid(None)
    ax.set_xlim((indep_var_min[j], indep_var_max[j]))
    ax.set_box_aspect(1)

    ax.yaxis.set_minor_locator(MultipleLocator(2))

    # xticks = [0.3, 0.5, 0.7, 0.9, 1.1, 1.3]
    # ax.set_xticks(xticks)

if "--savefig" in sys.argv:
    plt.savefig("fig_5.pdf")
else:
    plt.show()
