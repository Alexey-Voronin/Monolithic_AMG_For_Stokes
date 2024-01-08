##########################################################
# Data collection script
##########################################################
# Infer Problem type from the path
# e.g. `python  collect.py structured/3D` command
# results in msh_type='structured' and dim=3
##########################################################
import sys, os

path = sys.argv[1]
msh_type, dim_str = path.split("/")
dim = int(dim_str[0])
##########################################################
# Generate Problem Iterator
##########################################################
from problem_iterator import get_problem_iterator

iterator = get_problem_iterator(msh_type, dim)
disc_type = iterator.system_params["discretization"]["elem_type"]
##########################################################
# Generate MG parameters
# 'hlo' : $M_{ph}^{\omega_i>0}$
# 'ho'  : $M_{ph}^{\omega_1=0}$
# 'lo'  : $M_{ph}^{\omega_0=0}$
##########################################################
from mg_params import get_mg_params

mg_type = "hlo"
mg_params = get_mg_params(msh_type, disc_type, dim, mg_type)
##########################################################
# Switch to directory where the files will be wrriten to.
##########################################################
os.chdir(os.path.join(os.getcwd(), path))
##########################################################
# Collect convergence data
##########################################################
from sysmg.util.param_search import parameter_scan
import numpy as np


def setter(amg, params):
    amg.wrapper.set_eta(((params[0], params[0]), (1, 1)))


dp = 3  # digits past decimal place
tau_range = np.round(np.linspace(0.3, 1.3, 70), dp)
param_search = {
    "\eta": (tau_range,),
}

from sysmg import StokesMG

parameter_scan(iterator, StokesMG, mg_params, setter, param_search)
##########################################################
