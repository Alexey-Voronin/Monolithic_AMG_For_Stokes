# Monolithic Algebraic Multigrid Preconditioners for the Stokes Equations

## Abstract
In this paper, we investigate a novel monolithic algebraic multigrid solver for
the discrete Stokes problem discretized with stable mixed finite elements. The
algorithm is based on the use of the low-order $\pmb{\mathbb{P}}_1 \text{iso}\kern1pt\pmb{ \mathbb{P}}_2/ \mathbb{P}_1$
discretization as
a preconditioner for a higher-order discretization, such as \ptwopone{}.
Smoothed aggregation algebraic multigrid is used to construct independent
coarsenings of the velocity and pressure fields for the low-order
    discretization, resulting in a purely algebraic
preconditioner for the high-order discretization (i.e., using no geometric information).
Furthermore, we incorporate a novel block LU factorization technique for Vanka patches,
which balances computational efficiency with lower storage requirements.
The effectiveness of the
new method is verified for the $\pmb{\mathbb{P}}_2/\mathbb{P}_1$ (Taylor-Hood) discretization in two
and three dimensions on both structured and unstructured meshes.
Similarly, the approach is shown to be effective when applied to
the $\pmb{\mathbb{P}}_2/\mathbb{P}_1^{disc}$ (Scott-Vogelius) discretization on 2D
barycentrically refined meshes.
This novel monolithic algebraic multigrid solver not only meets but frequently surpasses the performance 
of inexact Uzawa preconditioners, demonstrating the versatility and robust performance across a diverse 
spectrum of problem sets, even where inexact Uzawa preconditioners struggle to converge.

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

The primary code is located in the [sysmg](./sysmg/) directory. The scripts required for data collection are stored in the [data\_collection](./data_collection/) directory.

## Dependencies

To effectively utilize this code, the following dependencies are needed:
- [Custom fork of PyAMG](https://github.com/Alexey-Voronin/pyamg-1/tree/e96af2b77a3baaf91ffb7ab4be43892c67ef39c0)
- [Firedrake](https://www.firedrakeproject.org/) (Compatibility tested with version 0.13.0)
- An indication of the OpenBLAS path in `sysmg/solvers/relaxation/core/setup.py`

## Data Collection and Results

The [data\_collection](./data_collection/) directory contains detailed information regarding the methods and scripts used for data collection. Each sub-directory also includes scripts for data visualization.





