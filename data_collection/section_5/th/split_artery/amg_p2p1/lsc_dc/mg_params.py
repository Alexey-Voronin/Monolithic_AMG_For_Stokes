##########################################################
# Generate MG parameters
def get_mg_params(dim):
    # LSC-DGS relaxation type
    mat_type = "BBT"
    lb = (0.5) ** (dim)
    ub = 1.1
    degree = 4
    step_1_iters = 1
    lsc_params = {
        "iterations": (3, 3),
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
        },
        "transform": {
            "operator": mat_type,
            "solver": "sa-amg",
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
        "relaxation": ("lsc", lsc_params),
        "levels": 10,
        "coarse_grid_solve": "splu",
    }

    return mg_params
