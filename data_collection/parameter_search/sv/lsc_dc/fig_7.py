import sys, os

sys.path.append(os.path.abspath("../../../plot_common"))
from common import set_figure, fig_size

import numpy as np
import sys
import itertools

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, AutoMinorLocator
import matplotlib.ticker as ticker


########################################################################
# Helper Function
########################################################################
def print_opt_values(rho, iters, ETAs, OMEGAs):
    """Optimal parameters based on the smallest number of iterations is more
    accurate than based solely on the convergence factor.
    """
    #################################################################
    # based on rho
    rho_min = rho.min()
    idx_opt = np.where(rho == rho_min)
    eta_id, omega_id = idx_opt[0].item(), idx_opt[1].item()

    eta_opt = ETAs[eta_id]
    omega_opt = OMEGAs[omega_id]
    iters_min = iters[eta_id, omega_id]
    print(
        f"rho-based:  (omega_0, eta_p)=({omega_opt:2.3f}, {eta_opt:2.3f}):\titer={iters_min:1.3f}\trcf={rho_min:1.3f}"
    )
    #################################################################
    # based on iter
    # often times there are multiple parameter choices that result in the same
    # iteration count. Choose the one with the smallest convergence factor

    iters_min = iters.min()
    idx_opt = np.where(iters == iters_min)

    cf_min_idx = np.argmin(rho[idx_opt[0], idx_opt[1]])
    eta_id, omega_id = idx_opt[0][cf_min_idx], idx_opt[1][cf_min_idx]
    eta_opt = ETAs[eta_id]
    omega_opt = OMEGAs[omega_id]
    rho_min = rho[eta_id, omega_id]
    iters_opt = iters[eta_id, omega_id]
    print(
        f"iter-based: (omega_0, eta_p)=({omega_opt:2.3f}, {eta_opt:2.3f}):\titer={iters_opt:1.3f}\trcf={rho_min:1.3f}"
    )

    return omega_opt, eta_opt


def loaddata(PATH):
    cf = np.load(PATH + "cf_data_1.npy")
    iters = np.load(PATH + "iters_data_1.npy")
    param = np.load(PATH + "param_ranges.npy", allow_pickle=True).tolist()

    ETAs = param["eta_0^p"][0]
    OMEGAs = param["omega"][0]
    rho = cf

    return cf, iters, ETAs, OMEGAs


########################################################################
# LOAD DATA
########################################################################
PREAM = "eta_omega_coarse/"
data_paths = {
    "Structured": PREAM + "structured/2D/",
    "Unstructured": PREAM + "unstructured/2D/",
}

########################################################################
# PLOT DATA
########################################################################
linewidth = 2

ncols = len(list(data_paths.keys()))  # 2
nrows = 1
fs = fig_size.singlefull
set_figure(width=fs["width"] * 1, height=fs["width"] * 0.5)
fig, AXES = plt.subplots(nrows, ncols, sharey="row", sharex="col")

for ax, (msh_type, PATH) in zip(AXES, data_paths.items()):
    ax.set_title(msh_type)
    cf, iters, ETAs, OMEGAs = loaddata(PATH)
    omega_opt, eta_opt = print_opt_values(cf, iters, ETAs, OMEGAs)

    iter_min = iters.min()
    iter_max = iters.max()

    X, Y = np.meshgrid(OMEGAs, ETAs)
    im = ax.contourf(
        X,
        Y,
        iters,
        cmap="hot",
        levels=np.arange(iter_min - 1, iter_max + 1) + 0.5,
        vmin=iter_min,
        vmax=iter_max,
        origin="lower",
    )

    ax.plot(omega_opt, eta_opt, c="w", marker="s", ms=3)

cb = fig.colorbar(im, ticks=np.arange(iter_min, iter_max))

for j, ax in enumerate(AXES):
    ax.set_box_aspect(1)
    ax.xaxis.set_minor_locator(MultipleLocator(0.05))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))

AXES[0].set_xlabel("$\omega_0$")
AXES[1].set_xlabel("$\omega_0$")
AXES[0].set_ylabel("$\eta_p$")

if "--savefig" in sys.argv:
    plt.savefig("fig_7.pdf")
else:
    plt.show()
