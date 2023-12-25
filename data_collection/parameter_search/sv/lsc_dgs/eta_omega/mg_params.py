"""Generate MG parameters"""
from copy import deepcopy

################################################################
# Vanka outer relaxation 
vanka_params = {  'type'        : 'geometric_dg',
                'iterations'    : (2,2),
                  'setup_opt'   : True,
                  'cblas'       : True,
                  'update'      : 'additive',
                  'patch_solver': 'inv',
                  'omega'       : 1.,
                  'debug'       : False}
vanka_outer_params = deepcopy(vanka_params)
################################################################
# LSC-DGS inner relaxation
mat_type       = 'BBT' 
steps_12_iters = 1
step_3_iters   = 2 
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
lsc_dgs_inner_params = deepcopy(lsc_dgs_params)

eta = ((1,1), (1,1))
tau = (1,1)
################################################################
# Putting it all together
mg_params = {
                 'type' : 'monolithic',
                 'interpolation':{  'type' : 'algebraic',
                                    'order': 'low',
                 'params'       :{  'u'    : {
                                              'strength'   : ('evolution', {'epsilon' : 4.0, 'k' : 2}),
                                              'min_coarse' : 75},
                                    'p'    : {'agg_mat'    : ('stiffness', 0.0),
                                              'smooth_mat' : ('stiffness', 0.0),
                                              'strength'   : ('evolution', {'epsilon' : 4., 'k' : 2}),
                                              'min_coarse' : 50
                                             }
                                     }
                                 },
                 'relaxation'        :  ('lsc', lsc_dgs_inner_params),
                 'wrapper_params'    : {'relax_params' : ('Vanka', vanka_outer_params),
                                        'tau' : tau,
                                        'eta' : eta,
                                        },
                 'levels'            : 10,
                 'coarse_grid_solve' : 'splu',
             }

