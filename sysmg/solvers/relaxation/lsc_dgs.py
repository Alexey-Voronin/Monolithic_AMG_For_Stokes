from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp
import pyamg

from sysmg.util.decorators import timeit
from .relaxation import System_Relaxation


class LSCDGS(System_Relaxation):
    """
    LSC-DGS smoother setup for Q2/Q1 and Q1isoQ2/Q1 Stokes Discretizations.

    """

    omega = None
    relax_iters = None

    @timeit("setup:solver:lscdgs:")
    def __init__(self, stokes,
                 params={'iterations': (2, 2),
                         'momentum'   : {'solver' : 'gauss-seidel',
                                         'solver_params' :{'iterations' : 2,
                                                           'sweep' : 'symmetric'},
                                         },
                         'continuity' : {'operator' : 'stiffness',
                                         'solver'   : 'sa-amg',
                                         'solver_params' :{
                                            'setup_phase' : {},
                                            'solve_phase' : {'iterations' : 1,
                                                             'cycle' : 'V'},
                                                         }
                                         },
                         'transform'  : {'operator' : 'stiffness',
                                         'solver'   : 'gauss-seidel',
                                         'solver_params' :{'iterations' : 2,
                                                           'sweep' : 'symmetric'},
                                        },
                         },
                 omega = 1,
                 debug = True,
                 level=0):
        super().__init__(params, level)

        # timer setup
        self.omega                = omega
        self.keep                 = params["keep"] if "keep" in params.keys() else False

        if self.empty_shell:
            # empty shell for when relaxation is skipped on a level
            return

        # Operator Setup
        self.system = stokes
        self.nullspace = getattr(stokes, 'nullspace', None)
        self._vdofs = stokes.velocity_nodes()
        self._pdofs = stokes.pressure_nodes()

        self._step_solvers = {}
        self._setup_relaxation_method()


    def _get_solver(self, A, solver, params, name):
        solver = solver.lower()

        if solver == 'direct':
            self._step_solvers[name] = sp.linalg.factorized(A.tocsc())
            def custom_solver(b):
                return self._step_solvers[name](b)
        elif solver == 'gauss-seidel':
            self._step_solvers[name] = solver
            def custom_solver(b):
                x = np.zeros_like(b)
                pyamg.relaxation.relaxation.gauss_seidel(A, x, b, **params)
                return x
        elif solver == 'sa-amg':
            solve_params = params.pop('solve_phase')
            setup_params = params.pop('setup_phase')
            mg = pyamg.smoothed_aggregation_solver(A, **setup_params)
            self._step_solvers[name] = mg

            def custom_solver(b):
                resid = []
                max_iter = solve_params.get('iterations', 1)
                x =  mg.solve(b,
                         maxiter=max_iter,
                         cycle=solve_params.get('cycle', 'V'),
                         residuals=resid
                        )
                assert len(resid)-1 == max_iter
                return x
        else:
            raise Exception(f"solver {solver} is not defined.")

        return custom_solver

    def _setup_relaxation_method(self):

        Abmat = self.system.A_bmat
        Au    = Abmat[0,0]
        B     = Abmat[1,0]
        BT    = Abmat[0,1]
        Ap    = self.system.stiffness_bmat[1,1]

        name = "momentum"
        iparams = self.params[name]
        self._momentum_solver = self._get_solver(Abmat[0,0],
                                                iparams['solver'],
                                                iparams['solver_params'],
                                                name)
        name = 'continuity'
        iparams = self.params[name]
        mat  = Ap if iparams['operator'].lower() == 'stiffness' else B*BT
        self._continuity_solver = self._get_solver(mat,
                                                  iparams['solver'],
                                                  iparams['solver_params'],
                                                  name)
        name = 'transform'
        iparams = self.params[name]
        mat  = Ap if iparams['operator'].lower() == 'stiffness' else B*BT
        self._transform_solver = self._get_solver(mat,
                                                  iparams['solver'],
                                                  iparams['solver_params'],
                                                  name)

        self._BABT = B*(Au*BT)

    def relax(self, K, up, rhs):

        Nu    = self._vdofs
        u,p   = up[:Nu], up[Nu:]
        f,g   = rhs[:Nu], rhs[Nu:]

        Abmat = self.system.A_bmat
        Au    = Abmat[0,0]
        B     = Abmat[1,0]
        BT    = Abmat[0,1]
        B_Au_BT = self._BABT

        for _ in range(self.relax_iters):
            # Step 1: Relax momentum equations
            u[:] = u + self._momentum_solver(f - Au * u - BT * p)
            # Step 2: Relax transformed continuity equations
            dq   = self._continuity_solver(g - 1.0 * (B * u))
            # Step 3: Transform the correction back to the original variables
            u[:] = u + BT * dq
            p[:] = p - self._transform_solver(B_Au_BT * dq)

        return up


