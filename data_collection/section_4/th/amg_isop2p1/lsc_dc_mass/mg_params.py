"""Generate MG parameters"""
from copy import deepcopy
import numpy as np


def get_mg_params(msh_type, disc_type, dim, mg_type):
    assert dim in [2, 3], f"dimension is wrong: {dim}"
    assert msh_type in ["structured", "unstructured"], f"mesh type is wrong: {msh_type}"
    assert disc_type in [("CG", "CG")], f"discretization is wrong: {str(disc_type)}"

    ################################################################
    # LSC-DC relaxation type
    mat_type = "BBT"
    lb = (0.5) ** dim
    ub = 1.1

    step_1_degree = 3 if dim == 2 else 4
    step_1_iters = 1

    step_2_degree = 3 if dim == 2 else 4
    step_2_iters = 1

    step_3_degree = 6 if dim == 2 else 16
    step_3_iters = 1

    lsc_params = {
        "iterations": (2, 2),
        "momentum": {
            "solver": "chebyshev",
            "solver_params": {
                "lower_bound": lb,
                "upper_bound": ub,
                "degree": step_1_degree,
                "iterations": step_1_iters,
            },
        },
        "continuity": {
            "operator": mat_type,
            "solver": "chebyshev",
            "solver_params": {
                "lower_bound": lb,
                "upper_bound": ub,
                "degree": step_2_degree,
                "iterations": step_2_iters,
            },
        },
        "transform": {
            "operator": mat_type,
            "solver": "chebyshev",
            "solver_params": {
                "lower_bound": lb,
                "upper_bound": ub,
                "degree": step_3_degree,
                "iterations": step_3_iters,
            },
        },
    }

    lsc_inner_params = deepcopy(lsc_params)
    lsc_outer_params = deepcopy(lsc_params)
    lsc_outer_params.update(
        {
            "mass": {
                "u": {"solver": "diag_inv", "solver_params": {}},
            }
        }
    )
    ################################################################
    # Damping parameter choices
    I = np.ones((2,))
    damp_param = {
        "structured": {
            "ho": {"2D": {"eta": I * 1.0}, "3D": {"eta": I * 0.60}},
            "hlo": {"2D": {"eta": I * 1.0}, "3D": {"eta": I * 1.10}},
        },
        "unstructured": {
            "ho": {"2D": {"eta": I * 1.00}, "3D": {"eta": I * 1.00}},
            "hlo": {"2D": {"eta": I * 1.00}, "3D": {"eta": I * 1.06}},
        },
    }

    eta = damp_param[msh_type][mg_type][f"{dim}D"].get("eta", ((1, 1), (1, 1)))
    eta = (eta, (1, 1)) if isinstance(eta, np.ndarray) else eta
    tau = damp_param[msh_type][mg_type][f"{dim}D"].get("tau", (1, 1))
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
            "relax_params": ("lsc", lsc_outer_params),
            "tau": tau,
            "eta": eta,
        },
        "levels": 10,
        "coarse_grid_solve": "splu",
    }

    if mg_type == "ho":
        # skips relaxation on on the finest iso level
        mg_params["wrapper_params"]["modify_mg_rlx"] = [(0, (0, 0))]
    elif mg_type == "lo":
        mg_params["wrapper_params"].pop("relax_params")

    return mg_params
