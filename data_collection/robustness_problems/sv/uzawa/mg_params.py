##########################################################
# Generate MG parameters
def get_mg_params(msh_type, dim):

    assert dim == 2, f'dim={dim} not supported atm.'

    relaxation   = ('fgmres_smoother',
                     {'iterations' : 2, 
                      'preconditioner' : ('jacobi', {'iterations' : 1})})
    amg_params = {  'strength'    : 'evolution',
                    'smooth'      : ('jacobi', {'omega': 4.0/3.0, 'degree' : 1}),
                    'presmoother' : relaxation,
                    'postsmoother': relaxation,
             }

    diag_params= {'operator'     : 'pressure mass',
                  'structured'   : True if msh_type == 'structured' else False,
                  'blocksize'    : 3,
                  'iterations'   : (1,1),
                  'block_solver' : 'inv'}


    mg_params = {    'type' : 'Uzawa',
                     'interpolation' : {'order' : 'high'},
                     'u' : { 'solver' : ('sa_amg', amg_params)},
                     'p' : { 'solver' : ('diag', diag_params)},
                     'debug' : False
                    }


    return mg_params
