from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp
import pyamg

from sysmg.util.decorators import timeit
from .relaxation import System_Relaxation


class LSC(System_Relaxation):
    """
    Least Squares Commutor (LSC)  smoother setup for Q2/Q1 and Q1isoQ2/Q1 Stokes Discretizations.

    """

    omega = None
    relax_iters = None

    @timeit("setup:solver:lsc:")
    def __init__(
        self,
        stokes,
        params={
            "iterations": (2, 2),
            "momentum": {
                "solver": "gauss-seidel",
                "solver_params": {"iterations": 2, "sweep": "symmetric"},
            },
            "continuity": {
                "operator": "stiffness",
                "solver": "sa-amg",
                "solver_params": {
                    "setup_phase": {},
                    "solve_phase": {"iterations": 1, "cycle": "V"},
                },
            },
            "transform": {
                "operator": "stiffness",
                "solver": "gauss-seidel",
                "solver_params": {"iterations": 2, "sweep": "symmetric"},
            },
            "mass": {
                "u": {"solver": "direct", "solver_params": {}},
            },
        },
        omega=1,
        debug=True,
        level=0,
    ):
        super().__init__(params, level)

        # timer setup
        self.omega = omega
        self.keep = params["keep"] if "keep" in params.keys() else False

        if self.empty_shell:
            # empty shell for when relaxation is skipped on a level
            return

        # Operator Setup
        self.system = stokes
        Abmat = self.system.A_bmat
        self._Au = Abmat[0, 0].copy().tocsr()
        self._B = Abmat[1, 0].copy().tocsr()
        self._BT = Abmat[0, 1].copy().tocsr()

        if "mass" in self.params.keys():
            self._Mu = stokes.mass_bmat[0, 0]
            # self._Mp = stokes.mass_bmat[1, 1]
            # self._Mp_inv = self._Mp.diagonal()

        self.nullspace = getattr(stokes, "nullspace", None)
        self._vdofs = stokes.velocity_nodes()
        self._pdofs = stokes.pressure_nodes()

        self._step_solvers = {}
        self._setup_relaxation_method()

        if self.accel:
            self._xt = np.zeros((self.system.ndofs(),))

    def _get_solver(self, A, solver, params, name):
        solver = solver.lower()

        if solver == "lumped_diag_inv":
            self._step_solvers[name] = sp.diags(1.0 / np.asarray(A.sum(axis=1)).ravel())

            def custom_solver(b):
                return self._step_solvers[name] * b

        elif solver == "diag_inv":
            self._step_solvers[name] = sp.diags(1.0 / A.diagonal()).tocsr()

            def custom_solver(b):
                return self._step_solvers[name] * b

        elif solver == "direct_assembled":
            self._step_solvers[name] = sp.csr_matrix(np.linalg.inv(A.toarray()))

            def custom_solver(b):
                return self._step_solvers[name] * b

        elif solver == "direct":
            self._step_solvers[name] = sp.linalg.factorized(A.tocsc())

            def custom_solver(b):
                return self._step_solvers[name](b)

        elif solver == "gauss-seidel":
            self._step_solvers[name] = solver

            def custom_solver(b):
                x = np.zeros_like(b)
                pyamg.relaxation.relaxation.gauss_seidel(A, x, b, **params)
                return x

        elif solver == "chebyshev":

            class Level(object):
                self.A = None

                def __init__(self, A):
                    self.A = A

            self._step_solvers[name] = pyamg.relaxation.smoothing.setup_chebyshev(
                Level(A), **params
            )

            def custom_solver(b):
                x = np.zeros_like(b)
                self._step_solvers[name](A, x, b)
                return x

        elif solver == "sa-amg":
            try:
                solve_params = params.pop("solve_phase")
            except:
                solve_params = {}
            try:
                setup_params = params.pop("setup_phase")
            except:
                setup_params = {}
            mg = pyamg.smoothed_aggregation_solver(A, **setup_params)
            self._step_solvers[name] = mg

            def custom_solver(b):
                # resid = []
                max_iter = solve_params.get("iterations", 1)
                x = mg.solve(
                    b,
                    maxiter=max_iter,
                    cycle=solve_params.get("cycle", "V"),
                    # residuals=resid
                )
                # assert len(resid)-1 == max_iter
                return x

        else:
            raise Exception(f"solver {solver} is not defined.")

        return custom_solver

    def _setup_relaxation_method(self):
        Au = self._Au
        B = self._B
        BT = self._BT

        if "mass" in self.params.keys():
            iparams = self.params["mass"]["u"]
            Mu_inv = self._get_solver(
                self._Mu, iparams["solver"], iparams.get("solver_params", {}), "Mu_inv0"
            )

            BBT = (B * (Mu_inv(BT))).tocsr()
            self._mass_u_inv = Mu_inv
            self._BABT = (B * Mu_inv(Au * Mu_inv(BT))).tocsr()

            """
            Mu_solve = self._get_solver(
                self._Mu, "direct", iparams.get("solver_params", {}),
                "Mu_inv_direct"
            )
            self._mass_u_inv = Mu_solve

            from scipy.sparse.linalg import LinearOperator
            def mv(v):
                return B * Mu_solve(Au * Mu_solve(BT*v))

            self._BABT = LinearOperator((B.shape[0],B.shape[0]), matvec=mv)
            """

            self.relax = self.relax_mass
        else:
            self._BABT = (B * (Au * BT)).tocsr()
            BBT = (B * BT).tocsr()
            self.relax = self.relax_no_mass

        name = "momentum"
        iparams = self.params[name]
        self._momentum_solver = self._get_solver(
            Au, iparams["solver"], iparams.get("solver_params", {}), name
        )
        name = "continuity"
        iparams = self.params[name]

        mat = BBT
        self._continuity_solver = self._get_solver(
            mat, iparams["solver"], iparams.get("solver_params", {}), name
        )
        name = "transform"
        iparams = self.params[name]
        mat = BBT
        self._transform_solver = self._get_solver(
            mat, iparams["solver"], iparams.get("solver_params", {}), name
        )

    def relax_mass(self, K, up, rhs):
        Nu = self._vdofs
        u, p = up[:Nu], up[Nu:]
        f, g = rhs[:Nu], rhs[Nu:]

        Au = self._Au
        B = self._B
        BT = self._BT

        for _ in range(self.relax_iters):
            vstar = self._momentum_solver(f - Au * u - BT * p)
            q = self._continuity_solver(g - B * (u + vstar))

            dp = -1 * self._transform_solver(self._BABT * q)
            u[:] = u + vstar + self._mass_u_inv(BT * q)
            p[:] = p + dp

        return up

    def relax_no_mass(self, K, up, rhs):
        Nu = self._vdofs
        u, p = up[:Nu], up[Nu:]
        f, g = rhs[:Nu], rhs[Nu:]

        Au = self._Au
        B = self._B
        BT = self._BT
        B_Au_BT = self._BABT

        for _ in range(self.relax_iters):
            # Step 1: Relax momentum equations
            u[:] = u + self._momentum_solver(f - Au * u - BT * p)
            # Step 2: Relax transformed continuity equations
            dq = self._continuity_solver(g - B * u)
            # Step 3: Transform the correction back to the original variables
            u[:] = u + BT * dq
            p[:] = p - self._transform_solver(B_Au_BT * dq)

        return up
