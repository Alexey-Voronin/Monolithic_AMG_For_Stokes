import numpy as np
from sysmg import StokesMG
from sysmg.util.param_search import parameter_scan

import sys,os

from mg_params import mg_params
from problem import get_problem_iterator

import sys,os
path = sys.argv[1]
msh_type, dim_str = path.split('/')
dim = int(dim_str[0])
##########################################################
iterator = get_problem_iterator(msh_type, dim)
##########################################################
# Switch to directoty where the files will be wrriten to.
##########################################################
os.chdir(os.path.join(os.getcwd(), path))
##########################################################


def setter(amg, params):
    eta0_p, omega0 = params
    amg.wrapper.set_eta(((1,eta0_p), (1,1)))
    amg.wrapper.outer_relaxation.set_omega(omega0)

dp            = 3 #digits past decimal place
eta_range     = np.round(np.linspace(0.01, 8, 61), dp)
omega_range   = np.round(np.linspace(0.01, 1.1, 44), dp)
eta_range     = np.round(np.linspace(0.01, 10, 200), dp)
omega_range   = np.round(np.linspace(0.01, 1.1, 200), dp)
#eta_range     = np.round(np.linspace(2, 5, 15), dp)
#omega_range   = np.round(np.linspace(0.4, 1.1, 15), dp)
#eta_range     = np.round(np.linspace(2, 10, 55), dp)
#omega_range   = np.round(np.linspace(0.4, 1.1, 5), dp)
param_search  = {'eta_0^p': (eta_range,),
                 'omega'  : (omega_range,),
               }

parameter_scan(iterator, StokesMG, mg_params, setter, param_search)
