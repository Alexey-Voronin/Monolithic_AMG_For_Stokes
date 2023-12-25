from sysmg.util.collect_data import collect_conv_data
import sys, os

##########################################################
# Generate MG parameters
##########################################################
from mg_params import get_mg_params

path = sys.argv[1]
msh_type, dim_str = path.split("/")
dim = int(dim_str[0])
mg_params = get_mg_params(dim)

##########################################################
# Generate Problem Iterator
##########################################################
sys.path.append(os.path.abspath("../"))
from problem_iterator import get_problem_iterator

iterator = get_problem_iterator(msh_type, dim)

##########################################################
# Switch to directory where the files will be wrriten to.
##########################################################
os.chdir(os.path.join(os.getcwd(), path))
##########################################################
# Collect convergence data
##########################################################
conv_data = collect_conv_data(
    iterator,
    mg_params,
    TOL=1e-10,
    MAX_ITER=60,
    rerun=5,
)
##########################################################
