##########################################################
# Generate MG parameters
def get_mg_params(dim):

    relaxation   = ('fgmres_smoother',
                     {'iterations' : 2, 
                      'preconditioner' : ('jacobi', {'iterations' : 1})})

    u_amg_params = {  'strength'    : 'evolution',
                         'smooth'      : ('jacobi', {'omega': 4.0/3.0, 'degree' : 1}),
                         'presmoother' : relaxation,
                         'postsmoother': relaxation,
                  }

    p_amg_params= {'operator'     : 'pressure mass',
                  'structured'   : True,
                   'blocksize'    : 1e100, # kluge -> pressure Mass MatSize
                   'iterations'   : (1,1),
                   'block_solver' : 'sa_amg',
                   'sa_amg_setup_params' : {'strength'      : 'evolution', 
                                             'smooth'      : ('jacobi', {'omega': 4.0/3.0, 'degree' : 1}),
                                            'presmoother' : relaxation,
                                            'postsmoother': relaxation,
                                            #'coarse_solver' : 'splu',
                                            #'min_coarse'    : 50,
                                            #'max_coarse'    :900,
                                           },
                   'sa_amg_solve_params' : {'tol'     : 1e-16,
                                            'maxiter' : 1}
                 }

    params = {    'type' : 'Uzawa',
                  'interpolation' : {'order' : 'high'},
                  'iterations' : (1,1),
                      'u' : { 'solver' : ('sa_amg', u_amg_params)},
                      'p' : { 'solver' : ('sa_amg', p_amg_params)},
                      'debug' : False
                     }



    return params
