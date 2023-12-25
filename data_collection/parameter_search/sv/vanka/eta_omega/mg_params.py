"""Generate MG parameters"""
from copy import deepcopy

################################################################
# Vanka relaxation type choices
vanka_options = { 'CG/DG' : {'2D' :
                                ({
                                   'type' : 'geometric_dg',
                                   'iterations'  : (2,2),
                                  },
                                 {
                                   'type' : 'algebraic_factorized',
                                   'iterations' : (1,1),
                                   'accel'      : {'iterations' : (2,2)},
                                  }),
                             }
                 }
vanka_outer, vanka_inner = vanka_options['CG/DG']['2D']

# shared paramters
vanka_params = {  'type'        : None,
                  'setup_opt'   : True,
                  'cblas'       : True,
                  'update'      : 'additive',
                  'patch_solver': 'inv',
                  'omega'       : 1.,
                  'debug'       : False}

vanka_inner_params = deepcopy(vanka_params)
vanka_inner_params.update(vanka_inner)
vanka_outer_params = deepcopy(vanka_params)
vanka_outer_params.update(vanka_outer)


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
                 'relaxation'        : ('Vanka', vanka_inner_params),
                 'wrapper_params'    : {'relax_params' : ('Vanka', vanka_outer_params),
                                        'tau' : tau,
                                        'eta' : eta,
                                        },
                 'levels'            : 10,
                 'coarse_grid_solve' : 'splu',
             }

