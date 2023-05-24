"""Generate MG parameters"""
from copy import deepcopy
import numpy as np

def get_mg_params(msh_type, disc_type, dim):
    assert dim in [2], f'dimension is wrong: {dim}'
    assert msh_type in ['structured', 'unstructured'], f'mesh type is wrong: {msh_type}'
    assert disc_type in [('CG', 'DG')], f'discretization is wrong: {str(disc_type)}'

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
    vanka_outer, vanka_inner = vanka_options[f'{disc_type[0]}/{disc_type[1]}'][f'{dim}D']

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
    ################################################################
    # Damping paramter choices
    if dim == 2:
        if msh_type == 'unstructured':
            vanka_outer_params['omega'] = 0.58
        else:
            vanka_outer_params['omega'] = 0.78


    I = np.ones((2,))
    damp_param = { 'structured'   : {'2D' : {'eta' : ((1.0, 2.68), (1,1))}},
                   'unstructured' : {'2D' : {'eta' : ((1.,4.0), (1.,1.))}},
                 }

    eta = damp_param[msh_type][f'{dim}D'].get('eta', ((1,1), (1,1)))
    eta = (eta, (1,1)) if isinstance(eta, np.ndarray) else eta
    tau = damp_param[msh_type][f'{dim}D'].get('tau', (1,1))
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

    return mg_params
