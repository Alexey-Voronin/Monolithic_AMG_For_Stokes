##########################################################
# Generate MG parameters
def get_mg_params(dim):
    # LSC-DGS relaxation type
    mat_type       = 'BBT' 
    steps_12_iters = 1
    step_3_iters   = 2 #if dim == 2 else 4
    lsc_dgs_params = {'iterations': (2, 2),
                      'momentum'   : {'solver' : 'gauss-seidel',
                                      'solver_params' :{'iterations'  : steps_12_iters,
                                                        'sweep'       : 'forward'},
                                      },

                      'continuity'   : {
                                      'operator' : mat_type,
                                      'solver' : 'gauss-seidel',
                                      'solver_params' :{'iterations'  : steps_12_iters,
                                                        'sweep'       : 'forward'},
                                      },
                      'transform'  : {'operator' : mat_type,
                                      'solver'   : 'gauss-seidel',
                                      'solver_params' :{'iterations'  : step_3_iters,
                                                        'sweep'       : 'symmetric'},
                                     }
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
        "relaxation": ("lsc", lsc_dgs_params),
        "levels": 10,
        "coarse_grid_solve": "splu",
    }

    return mg_params
