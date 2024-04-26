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
sys.path.append(os.path.abspath("../../../"))
from problem_iterator import get_problem_iterator

iterator = get_problem_iterator(msh_type, dim)
disc_type = iterator.system_params["discretization"]["elem_type"]
##########################################################
# Generate MG parameters
# 'hlo' : $M_{ph}^{\omega_i>0}$
# 'ho'  : $M_{ph}^{\omega_1=0}$
# 'lo'  : $M_{ph}^{\omega_0=0}$
##########################################################
sys.path.append(os.path.abspath("../"))
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
from sysmg.util.collect_data import collect_conv_data

conv_data = collect_conv_data(
    iterator,
    mg_params,
    TOL=1e-10,
    MAX_ITER=60,
    rerun=5,
)
##########################################################
