"""Generate MG parameters"""
from copy import deepcopy

dim = 2
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
# LSC-DC inner relaxation
mat_type       = 'BBT' 
lb             = (0.5)**dim
ub             = 1.1
degree         = 3 if dim == 2 else 4
steps_12_iters = 1
step_3_iters   = 2 if dim == 2 else 4
lsc_dc_params = {'iterations': (2, 2),
                  'momentum'   : {'solver' : 'chebyshev',
                                  'solver_params' :{'lower_bound' : lb,
                                                    'upper_bound' : ub,
                                                    'degree'      : degree,
                                                    'iterations'  : steps_12_iters,
                                                    },
                                  },

                  'continuity'   : {
                                  'operator' : mat_type,
                                  'solver' : 'chebyshev',
                                  'solver_params' :{'lower_bound' : lb,
                                                    'upper_bound' : ub,
                                                    'degree'      : degree,
                                                    'iterations'  : steps_12_iters,
                                                    },
                                  },
                  'transform'  : {'operator' : mat_type,
                                  'solver'   : 'chebyshev',
                                  'solver_params' :{'lower_bound' : lb,
                                                    'upper_bound' : ub,
                                                    'degree'      : degree,
                                                    'iterations'  : step_3_iters,
                                                    },
                                 }
                 }

lsc_dc_inner_params = deepcopy(lsc_dc_params)

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
                 'relaxation'        :  ('lsc', lsc_dc_inner_params),
                 'wrapper_params'    : {'relax_params' : ('Vanka', vanka_outer_params),
                                        'tau' : tau,
                                        'eta' : eta,
                                        },
                 'levels'            : 10,
                 'coarse_grid_solve' : 'splu',
             }

