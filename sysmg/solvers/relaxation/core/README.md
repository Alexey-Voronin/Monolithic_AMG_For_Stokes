# Find BLAS
`python -c 'import numpy as np; np.show_config()'`

# Compile the patch\_mult module
`python setup.py build_ext --inplace`

You will need to update cblas paths in the setup.py for this to work.

# Show linked libraries
`ldd *.so`
