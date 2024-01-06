##########################################################
# Generate MG parameters
def get_mg_params(dim):
    # LSC-DGS relaxation type
    mat_type = "BBT"
    lb = (0.5) ** dim
    ub = 1.1
    degree = 3 if dim == 2 else 4
    steps_12_iters = 1
    step_3_iters = 2 if dim == 2 else 4
    lsc_dc_params = {
        "iterations": (2, 2),
        "momentum": {
            "solver": "chebyshev",
            "solver_params": {
                "lower_bound": lb,
                "upper_bound": ub,
                "degree": degree,
                "iterations": steps_12_iters,
            },
        },
        "continuity": {
            "operator": mat_type,
            "solver": "chebyshev",
            "solver_params": {
                "lower_bound": lb,
                "upper_bound": ub,
                "degree": degree,
                "iterations": steps_12_iters,
            },
        },
        "transform": {
            "operator": mat_type,
            "solver": "chebyshev",
            "solver_params": {
                "lower_bound": lb,
                "upper_bound": ub,
                "degree": degree,
                "iterations": step_3_iters,
            },
        },
    }

    mg_params = {
        "type": "monolithic",
        "interpolation": {
            "type": "algebraic",
            "order": "high",
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
        "relaxation": ("lsc", lsc_dc_params),
        "levels": 10,
        "coarse_grid_solve": "splu",
    }

    return mg_params
