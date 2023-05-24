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

This GitHub repository houses the code referenced in the aforementioned publications.

# How to Run the Example Problems

The primary code is located in the `sysmg` directory. The scripts required for data collection are stored in the `data_collection` directory.

## Dependencies

To effectively utilize this code, the following dependencies are needed:
- [Custom fork of PyAMG](https://github.com/Alexey-Voronin/pyamg-1/tree/e96af2b77a3baaf91ffb7ab4be43892c67ef39c0)
- [Firedrake](https://www.firedrakeproject.org/) (Compatibility tested with version 0.13.0)
- An indication of the OpenBLAS path in `sysmg/solvers/relaxation/core/setup.py`

## Data Collection and Results

The `data_collection` directory contains detailed information regarding the methods and scripts used for data collection. Each sub-directory also includes scripts for data visualization.





