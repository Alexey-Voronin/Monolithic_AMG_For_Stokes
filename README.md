# Monolithic Algebraic Multigrid Preconditioners for the Stokes Equations

## Abstract
TBU (To Be Updated)

## Authors
- Alexey Voronin (voronin2@illinois.edu)
- Scott MacLachlan
- Luke N. Olson
- Raymond Tuminaro

The published paper can be found at:
- [arxiv (TBU)]()
- [JCP (TBU)]()

This git repository contains the code described in the above-mentioned publications.

# Running the Code

To use this code, you will need the following dependencies:
- [PyAMG](https://github.com/Alexey-Voronin/pyamg-1/tree/e96af2b77a3baaf91ffb7ab4be43892c67ef39c0)
- [Firedrake](https://www.firedrakeproject.org/)
- Specify the OpenBLAS path in `sysmg/solvers/relaxation/core/setup.py`

The code can be found in the `sysmg` directory, and all the relevant data-collection scripts are located in the `data_collection` folder.