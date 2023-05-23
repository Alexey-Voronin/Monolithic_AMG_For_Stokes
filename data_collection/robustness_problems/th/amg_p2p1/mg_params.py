##########################################################
# Generate MG parameters
def get_mg_params(dim):

    vanka_params = {  
                      'type'         : 'algebraic_factorized',
                      'setup_opt'    : True,
                      'cblas'        : True,
                      'iterations'   : (1,1),
                      'accel'        : {'iterations' : (2,2)},
                      'update'       : 'additive',
                      'patch_solver' : 'inv',
                      'omega'        : 1.,
                      'debug'        : False}

    mg_params = {
                     'type' : 'monolithic',
                     'interpolation':{  'type' : 'algebraic',
                                        'order': 'high',
                     'params'       :{  'u'    : {
                                                  'strength'   : ('evolution', {'epsilon' : 4., 'k' : 2}),
                                                  'min_coarse' : 75},
                                        'p'    : {'agg_mat'    : ('stiffness', 0.0),
                                                  'smooth_mat' : ('stiffness', 0.0),
                                                  'strength'   : ('evolution', {'epsilon' : 4., 'k' : 2}),
                                                  'min_coarse' : 50
                                                 }
                                         }
                                     },
                     'relaxation'        : ('Vanka', vanka_params),
                     'levels'            : 10,
                     'coarse_grid_solve' : 'splu',
                 }

    return mg_params
