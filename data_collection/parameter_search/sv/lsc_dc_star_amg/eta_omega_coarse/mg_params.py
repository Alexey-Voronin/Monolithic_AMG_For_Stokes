"""Generate MG parameters"""
from copy import deepcopy
import numpy as np


msh_type  = "unstructured"
disc_type = ("CG", "DG")
dim       = 2

assert dim in [2], f"dimension is wrong: {dim}"
assert msh_type in ["structured", "unstructured"], f"mesh type is wrong: {msh_type}"
assert disc_type in [("CG", "DG")], f"discretization is wrong: {str(disc_type)}"

################################################################
# Vanka relaxation type choices
vanka_outer_params = {
    "type": "geometric_dg",
    "iterations": (2, 2),
    "setup_opt": True,
    "cblas": True,
    "update": "additive",
    "patch_solver": "inv",
    "omega": 1.0,
    "debug": False,
}
################################################################
# LSC-DC inner relaxation
mat_type = "BBT"
lb = (0.5) ** dim
ub = 1.1
degree = 3 if dim == 2 else 4
step_1_iters = 1
lsc_inner_params = {
    "iterations": (2, 2),
    "momentum": {
        "solver": "chebyshev",
        "solver_params": {
            "lower_bound": lb,
            "upper_bound": ub,
            "degree": degree,
            "iterations": step_1_iters,
        },
    },
    "continuity": {
        "operator": mat_type,
        "solver": "sa-amg",
        "solver_params": {"solve_phase": {"iterations": 1}},
    },
    "transform": {
        "operator": mat_type,
        "solver": "sa-amg",
        "solver_params": {"solve_phase": {"iterations": 1}},
    },
}
################################################################
# Damping parameter choices
if dim == 2:
    if msh_type == "unstructured":
        vanka_outer_params["omega"] = 1.
    else:
        vanka_outer_params["omega"] = 1.

I = np.ones((2,))
damp_param = {
    "structured": {"2D": {"eta": ((1.0, 1.0), (1.0, 1.0))}},
    "unstructured": {"2D": {"eta": ((1.0, 1.00), (1.0, 1.0))}},
}

eta = damp_param[msh_type][f"{dim}D"].get("eta", ((1, 1), (1, 1)))
eta = (eta, (1, 1)) if isinstance(eta, np.ndarray) else eta
tau = damp_param[msh_type][f"{dim}D"].get("tau", (1, 1))
################################################################
# Putting it all together
mg_params = {
    "type": "monolithic",
    "interpolation": {
        "type": "algebraic",
        "order": "low",
        "params": {
            "u": {
                "strength": ("evolution", {"epsilon": 4.0, "k": 2}),
                "min_coarse": 75,
            },
            "p": {
                "agg_mat": ("stiffness", 0.0),
                "smooth_mat": ("stiffness", 0.0),
                "strength": ("evolution", {"epsilon": 4.0, "k": 2}),
                "min_coarse": 50,
            },
        },
    },
    "relaxation": ("lsc", lsc_inner_params),
    "wrapper_params": {
        "relax_params": ("Vanka", vanka_outer_params),
        "tau": tau,
        "eta": eta,
    },
    "levels": 10,
    "coarse_grid_solve": "splu",
}

