# Monolithic Algebraic Multigrid Preconditioners for the Stokes Equations

## Abstract
We investigate a novel monolithic algebraic multigrid (AMG) preconditioner for the Taylor-Hood ($\pmb{\mathbb{P}}_2/\mathbb{P}_1$) and Scott-Vogelius ($\pmb{\mathbb{P}}_2/\mathbb{P}_1^{disc}$) discretizations of the Stokes equations. The algorithm is based on the use of the lower-order $\pmb{\mathbb{P}}_1 \text{iso}\kern1pt\pmb{ \mathbb{P}}_2/ \mathbb{P}_1$ operator within a defect-correction setting, in combination with AMG construction of interpolation operators for velocities and pressures. The preconditioning framework is primarily algebraic, though the $\pmb{\mathbb{P}}_1 \text{iso}\kern1pt\pmb{ \mathbb{P}}_2/ \mathbb{P}_1$ operator must be provided. We investigate two relaxation strategies in this setting. Specifically, a novel block factorization approach is devised for Vanka patch systems, which significantly reduces storage requirements and computational overhead, and a Chebyshev adaptation of the LSC-DGS relaxation from [[Wang and Chen, 2013]](https://link.springer.com/article/10.1007/s10915-013-9684-1) is developed to improve parallelism. The preconditioner demonstrates robust performance across a variety of 2D and 3D Stokes problems, often matching or exceeding the effectiveness of an inexact block-triangular (or Uzawa) preconditioner, especially in challenging scenarios such as elongated-domain problems. 

## Authors
- Alexey Voronin (axvsim [at] proton.me)
- Scott MacLachlan
- Luke N. Olson
- Raymond Tuminaro

The published paper can be found at:
- [arxiv](https://arxiv.org/abs/2306.06795)
- [SISC (TBU)]()

This GitHub repository houses the code referenced in the aforementioned publications.

# How to Run the Example Problems

The primary code is located in the [sysmg](./sysmg/) directory. The scripts required for data collection are stored in the [data\_collection](./data_collection/) directory.

## Dependencies

To effectively utilize this code, the following dependencies are needed:
- [Custom fork of PyAMG](https://github.com/Alexey-Voronin/pyamg-1/tree/sysmg_krylov_accel)
- [Firedrake](https://www.firedrakeproject.org/) (Compatibility tested with version 0.13.0)
- An indication of the OpenBLAS path in `sysmg/solvers/relaxation/core/setup.py`

## Data Collection and Results

The [data\_collection](./data_collection/) directory contains detailed information regarding the methods and scripts used for data collection. Each sub-directory also includes scripts for data visualization.
