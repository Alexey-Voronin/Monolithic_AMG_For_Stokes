import gc

import numpy as np

# analysis
import pandas as pd
import scipy.sparse as sp
from pyamg import smoothed_aggregation_solver

from sysmg.util.decorators import timeit

# sysmg imports
from .mg_common import MG
from .relaxation.block_solve import block_solve


class BlockDiagMG(MG):
    """BlockDiagMG Object.

    # Parameters
    relaxation = (  'jacobi', {'omega' : 1.7, 'iterations' : 2})
    amg_params = {  'strength'    : 'evolution',
                      'smooth'      : ('jacobi', {'omega': 4.0/3.0, 'degree' : 1}),
                      'presmoother' : relaxation,
                      'postsmoother': relaxation,
                }

    diag_params= {  'operator' : 'pressure mass', 'blocksize' : 3,
                    'iterations' : (1,1), 'block_solver' : 'pinv'}

    solve_param =  {'type' : "physics" or "uzawa",
                    'u'    : { 'solver' : ('sa_amg', amg_params)},
                    'p'    : { 'solver' : ('diag', diag_params)},
                   }
    """

    @timeit("setup:solver:mg:")
    def __init__(self, stokes_system, solver_param, keep=False):
        """Initialize BlockDiagMG Object.

        keep: bool
            - option to keep/discard multigrid setup objects not necessary for
            the actual solve.
        """
        import copy

        mg_param_c = copy.deepcopy(solver_param)  # gets modified later

        sys_order = mg_param_c["interpolation"].get("order", None)
        if sys_order == "low":
            super().__init__(stokes_system.lo_fe_sys, mg_param_c)
            self.ho_system = stokes_system
        elif sys_order == "high":
            super().__init__(stokes_system, mg_param_c)
            self.ho_system = stokes_system
        else:
            raise ValueError(
                'Please provide system order for Stokes_AMG: options are "low" or "high".'
            )

        self.dim = self.system.dim
        self.velocity_solve = mg_param_c["u"]
        self.pressure_solve = mg_param_c["p"]
        self.keep = keep
        self.cycle = mg_param_c.get("cycle", "V")
        self.type = mg_param_c.get("type", "uzawa").lower()
        self.alpha = mg_param_c.get("alpha", 1.0)
        self.beta = mg_param_c.get("beta", 1.0)
        # these options are only used when Uzawa is used as
        # a relaxation method
        self.iterations = mg_param_c.get("iterations", 1)
        self.omega = mg_param_c.get("omega", 1)
        ################################################
        # compute grid hierarchy
        self._setup_solver()
        ################################################
        ################################################
        # force construction
        self.aspreconditioner(self.cycle)

        # don't leave reference here
        del self.system

        def _del_attr(obj, attr):
            if hasattr(obj, attr):
                delattr(obj, attr)

        if not keep:
            _del_attr(self, "ho_system")
            if hasattr(self, "wrapper"):
                _del_attr(self, "stokes")

        gc.collect()  # force garbage collection

    @timeit("setup:solver:mg:")
    def _setup_solver(self):
        stokes = self.system
        Au = stokes.A_bmat[0, 0]

        if self.velocity_solve["solver"][0] == "sa_amg":
            # see kluge below for explanation of this mess
            vel_amg_params = self.velocity_solve["solver"][1]
            self.gamma = vel_amg_params.pop("gamma", 1)
            name, smoother_param = vel_amg_params["presmoother"]
            # self.relax_u_accel = smoother_param.pop('accel', None)
            vel_amg_params["presmoother"] = (name, smoother_param)
            vel_amg_params["postsmoother"] = (name, smoother_param)
            self.ml = smoothed_aggregation_solver(Au, **vel_amg_params)
            self._process_ml(self.ml, field="u")
        else:
            raise ValueError("Only sa_amg solver has been implemented for velocity.")

        if self.pressure_solve["solver"][0] in ["diag", "sa_amg"]:
            self.bs = block_solve(stokes, self.pressure_solve["solver"][1])
        else:
            raise ValueError(
                "Only block-diag solver has been implemented for pressure."
            )

        # if self.relax_u_accel is None:
        #     return

        import pyamg

        # KLUGE: krylov acceleration
        # TODO: make this more general
        # Approach: use relaxation/scalar.py
        # for lvl, level in enumerate(self.ml.levels[:-1]):
        #     self.ml.levels[lvl].presmoother_old = self.ml.levels[lvl].presmoother
        #
        #     def get_rlx(lvl, n_gmres):
        #         def mv(r):
        #             x = np.zeros_like(r)
        #             level.presmoother_old(self.ml.levels[lvl].A, x, r)
        #             return x
        #
        #         self.ml.levels[lvl].Mu = sp.linalg.LinearOperator(self.ml.levels[lvl].A.shape, matvec=mv)
        #
        #         @timeit("solution:solver:bd_mg:")
        #         def rlx(A, x, b):
        #             x[:] = pyamg.krylov.fgmres(self.ml.levels[lvl].A, b, x0=x,
        #                                        tol=1e-40,
        #                                        maxiter=1,  # = max_outer
        #                                        restrt=n_gmres,  # = max_inner
        #                                        M=self.ml.levels[lvl].Mu)[0]
        #             return lvl
        #
        #         return rlx
        #
        #     self.ml.levels[lvl].presmoother = get_rlx(lvl, self.relax_u_accel)
        #     self.ml.levels[lvl].postsmoother = self.ml.levels[lvl].presmoother

    # Needed for when Uzawa is used as relaxation
    def presmoother(self, A, x, b):
        for _ in range(self.iterations):
            x[:] += self.omega * self.M * (b - A * x)

    def postsmoother(self, A, x, b):
        for _ in range(self.iterations):
            x[:] += self.omega * self.M * (b - A * x)

    def aspreconditioner(self, cycle="V"):
        if hasattr(self, "M"):
            return self.M

        Au = self.system.A_bmat[0, 0]
        B = self.system.A_bmat[1, 0]
        null = self.null[Au.shape[0] :] if self.null is not None else None
        Mp = self.bs.aslinearoperator()

        def mv(r):
            nv = Au.shape[0]
            x = np.zeros_like(r)

            x[:nv] = self.alpha * self.ml.solve(
                r[:nv], x0=None, tol=1e-50, maxiter=self.gamma, cycle=cycle
            )
            if self.type == "uzawa":
                x[nv:] -= self.beta * (Mp @ (r[nv:] - B * x[:nv]))
            elif self.type == "physics":
                x[nv:] = Mp @ r[nv:]
            else:
                raise ValueError(
                    "Block diagonal solver can be of two types physics or Uzawa."
                )

            if null is not None:
                x[nv:] -= np.dot(x[nv:], null) * null / np.dot(null, null)

            return x

        self.M = sp.linalg.LinearOperator(self.system.A_bmat.shape, matvec=mv)

        return self.M

    def __repr__(self):
        """Print basic statistics about the multigrid hierarchy.

        Inf-sup is a bit hackey - clean it up later.
        """
        # (pre)compute this call once
        if hasattr(self, "image"):
            return self.image

        ml = self.ml
        output = "Stokes MultilevelSolver\n"
        output += f"Number of Levels:     {len(ml.levels)}\n"

        O_ml = ""
        for tmp in self.mls_info["complexities"]:
            tmp = "%4.2f" % tmp
            O_ml += f"{tmp:>5}"

        output += f"Operator Complexity: {ml.operator_complexity():6.3f}:" + O_ml + "\n"

        output += f"Grid Complexity:     {ml.grid_complexity():6.3f}\n"
        output += f"Coarse Solver:        {ml.coarse_solver.name()}\n"

        total_nnz = sum(level.A.nnz for level in ml.levels)
        output += "  level   unknowns     nonzeros           Ratio[v/p]"

        if self.dim == 2:
            output += "   CR[vx,vy,p]"
        else:
            output += "   CR[vx,vy,vz,p]"

        output += "\n"
        for n, lvl in enumerate(ml.levels):
            A = lvl.A
            stk_ratio = "%2.2f" % (100 * A.nnz / total_nnz)
            # v/p ratio
            dofs = self.mls_info["dofs"][n]
            vp_ratio = np.sum(dofs[:-1]) / dofs[-1]
            output += (
                f"{n:>6} {A.shape[1]:>11} {A.nnz:>12} [{stk_ratio:>5}%]"
                + f"{vp_ratio:>8.1f}"
            )
            # coarsening rate
            crs = self.mls_info["coarsening"][n]
            crs = ["%2.2f" % cr if cr > 0 else "" for cr in crs]
            # if self.dim == 2:
            #     output += f'       [{crs[0]:>5},{crs[1]:>5},{crs[2]:>5}]'
            # else:
            #     output += f'       [{crs[0]:>5},{crs[1]:>5},{crs[2]:>5},{crs[3]:>5}]'

            output += "\n"

        self.image = output
        return output

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_beta(self, beta):
        self.beta = beta

    def to_pandas(self):
        mg_info = self.to_dict()
        data = {k: list(v.values()) for k, v in mg_info.items()}
        return pd.DataFrame.from_dict(data)

    def to_dict(self):
        # Default PyAMG hierarchy info
        mg_info = self.ml.to_dict()
        # Ratio of velocity to pressure DoFs
        mg_info.update({"Ratio[v/p]": {}})

        # Coarsening Rates, DoFs on each grid, and Complexities
        add_fields = {
            "CR(u)": {},
            "dofs(u)": {},
            "O(ml)": {},
            "O(ml(A))": {},
            "O(u)": {},
        }

        mg_info.update(add_fields)

        for lvl, level in enumerate(self.ml.levels):
            ######################################################
            # v/p ratio
            dofs = self.mls_info["dofs"][lvl]
            vp_ratio = np.sum(dofs[:-1]) / dofs[-1]
            mg_info["Ratio[v/p]"][lvl] = round(vp_ratio, 3)

            ######################################################
            # coarsening rate
            crs = self.mls_info["coarsening"][lvl]
            crs = [round(cr, 3) if cr > 0 else None for cr in crs]

            mg_info["CR(u)"][lvl] = crs[0]
            # dofs on each grid
            mg_info["dofs(u)"][lvl] = dofs[0]

            ######################################################
            # Coomplexities
            if lvl == 0:
                complexities = self.mls_info["complexities"]
                mg_info["O(ml)"][lvl] = np.round(self.ml.operator_complexity(), 3)
                mg_info["O(ml(A))"][lvl] = np.round(self.ml.grid_complexity(), 3)
            else:
                complexities = [None] * (1 + self.dim)
                mg_info["O(ml)"][lvl] = None
                mg_info["O(ml(A))"][lvl] = None

            mg_info["O(u)"][lvl] = complexities[0]

        return mg_info
