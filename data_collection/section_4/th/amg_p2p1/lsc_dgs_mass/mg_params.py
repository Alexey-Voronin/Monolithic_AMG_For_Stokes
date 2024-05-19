##########################################################
# Generate MG parameters
def get_mg_params(dim):
    # LSC-DGS relaxation type
    mat_type = "BBT"
    step_1_iters = 1
    step_2_iters = 1
    step_3_iters = 2
    lsc_params = {
        "iterations": (2, 2),
        "momentum": {
            "solver": "gauss-seidel",
            "solver_params": {"iterations": step_1_iters, "sweep": "symmetric"},
        },
        "continuity": {
            "operator": mat_type,
            "solver": "gauss-seidel",
            "solver_params": {"iterations": step_2_iters, "sweep": "symmetric"},
        },
        "transform": {
            "operator": mat_type,
            "solver": "gauss-seidel",
            "solver_params": {"iterations": step_3_iters, "sweep": "symmetric"},
        },
        "mass": {
            "u": {"solver": "diag_inv", "solver_params": {}},
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
