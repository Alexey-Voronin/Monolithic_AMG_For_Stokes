import numpy as np
from pyamg import smoothed_aggregation_solver
from pyamg.krylov import fgmres as pyamg_fgmres
from pyamg.krylov import gmres as pyamg_gmres
from scipy.sparse.linalg import gmres as scipy_gmres

from sysmg.util.callback import CallBack
from sysmg.util.decorators import timeit


class MG(object):
    """Multigrid Solver Interface.

    MG Interface extended by specific MG solvers.
    """

    timings = None

    def __init__(self, system, mg_param, keep=False):
        """Initialize MG object.

        Function takes in system object and multigrid parameters dictionary.
        """
        self.system = system
        self.mg_param = mg_param
        self.keep = keep
        self.null = getattr(self.system, "nullspace", None)

    def _init_data_structs(self):
        # initialize data structures if first call..
        if not hasattr(self, "mls_info"):
            self.mls_info = {
                "complexities": [],
                "grid_complexities": [],
                "coarsening": {},
                "dofs": {},
            }
        if not hasattr(self, "Ps"):
            self.Ps = {"u": {}, "p": {}}
        if self.keep and not hasattr(self, "mls"):
            self.mls = []  # store hierarchies

    def _process_ml(self, ml, field=None):
        """Process PyAMG's multilevel object.

        Why jump through all this processing hoops? We need to be more mindful
        of memory usage when it comes to bigger problems.
        """
        lvls = ml.levels
        self._init_data_structs()

        # save hierarchy related data for analysis
        self.mls_info["complexities"].append(ml.operator_complexity())
        self.mls_info["grid_complexities"].append(ml.grid_complexity())
        for i, lvl in enumerate(lvls):
            v = 0 if i == len(lvls) - 1 else lvl.A.shape[0] / lvls[i + 1].A.shape[0]
            self.mls_info["coarsening"].setdefault(i, []).append(v)
            self.mls_info["dofs"].setdefault(i, []).append(lvl.A.shape[0])
        # save interpolation operators
        Ps_c = self.Ps[field]
        for i, lvl in enumerate(lvls[:-1]):
            Ps_c.setdefault(i, []).append(lvl.P)
        # clean-up
        if self.keep:
            self.mls.append(ml)
        else:
            del ml

    def _get_gmg_interp(self, field="u", ignore_bcs=False):
        """Get GMG interpolation operator."""
        raise Exception("Not implemented here")

    def _get_amg_interp(self, A, params):
        """Get AMG interpolation operator."""
        ml = smoothed_aggregation_solver(A, **params)
        """
        Ps = []
        for lvl in range(len(ml.levels)-1):
            P = ml.levels[lvl].P

            Ps.append(P)
            if not isinstance(Ps[-1], sp.csr.csr_matrix):
                Ps[-1] = Ps[-1].tocsr()
        return Ps, ml
        """
        return ml

    def __repr__(self):
        return self.ml.__repr__()

    def _compute_agg_size(self, ml, lvl):
        """Compute average aggregate size for."""
        ave = -1.0
        if hasattr(ml.levels[lvl], "AggOp"):
            AggOp = ml.levels[lvl].AggOp.T.tocsr()
            indptr = AggOp.indptr
            ave = np.mean(indptr[1:] - indptr[:-1])
        return ave

    @timeit("solution:solver:mg:")
    def solve(
        self,
        b,
        x0=None,
        A=None,
        maxiter=40,
        tol=1e-8,
        cycle_type="V",
        accel={"module": ("pyamg", "gmres"), "resid": "rel", "restart": 20},
    ):
        """
        Linear system iterative solver.

        Linear solver that uses (preconditioned) GMRES or stationary
        iteration provided as linear operator to compute solution to the input
        linear system.

        Args:
            b: array
                right hand side, shape is (n,) or (n,1)
            x0 : array
                initial guess, default is a vector of zeros.
            A : sparse matrix
                If the system being solved is different from the one that mg
                is based on. For example using a mg based on a lower-order
                discretization (isoP2/P1) to precondition a problem
                discretized with higher-order bases.
            maxiter : int
                - default to 40
                - Maximum number of stationary iterations if Krylov method is
                not used.
                - If Krylov method used, it is passed to the Krylov solver
                as maxiter.
            tol: float
                relative convergence tolerance.
            accel: dict
                Dictionary containing the information about what type of
                solver to use and what type of residual to report
                (rel. or abs.).

                Options:
                    - 'module' :  'cycle', 'scipy', ('pyamg', 'gmres'),
                                    ('pyamg', 'fgmres')
                    - "resid"  : 'rel', 'abs'

                Examples:
                    - {'module': "cycle",      "resid": 'abs'}
                    - {'module': ("pyamg", 'fgmres'),
                        "resid": 'abs', 'restart' : 20}

        Returns:
            solution vector, and array with residual history.

        """
        solver_type = None
        if accel is not None:
            resid_type = accel.get("resid", "abs")
            solver_type = accel.get("module", ("pyamg", "fgmres"))
            restart = accel.get("restart", 20)
        else:
            resid_type = "abs"

        A_mg = self.ml.levels[0].A
        A = A if A != None else A_mg
        M = self.aspreconditioner(cycle=cycle_type)

        bnorm = np.linalg.norm(b)
        tol = tol / bnorm if bnorm > 0 else tol
        x0 = np.zeros_like(b) if x0 is None else x0
        cb = CallBack(A, b, M) if resid_type == "rel" else CallBack(A, b)

        if solver_type == "cycle":
            x1 = x0
            for i in range(maxiter):
                cb(x1)
                if cb.resid[-1] < tol or (cb.resid[-1] > 1e6 and i > 5):
                    # if converged or begins to diverge stop iterating
                    break
                r = b - A * x1
                x1 += M * r
        elif solver_type == ("scipy", "gmres"):
            cb = CallBack()
            x1, info = scipy_gmres(
                A, b, x0=x0, tol=tol, maxiter=maxiter, M=M, callback=cb, restart=restart
            )
        elif solver_type == ("pyamg", "gmres"):
            x1, info = pyamg_gmres(
                A,
                b,
                x0=x0,
                tol=tol,
                # inner iteration
                restrt=restart,
                # outer iter
                maxiter=int(maxiter / restart),
                M=M,
                callback=cb,
                orthog="mgs",
            )
        elif solver_type == ("pyamg", "fgmres"):
            out = pyamg_fgmres(
                A,
                b,
                x0=x0,
                tol=tol,
                # inner iteration
                restrt=restart,
                # outer iter
                maxiter=int(maxiter / restart),
                M=M,
                callback=cb,
                return_time=True,
            )
            x1, info = out[:2]
        else:
            raise ValueError("module type is wrong: " + str(solver_type))

        resid = cb.get_residuals()
        del cb

        return x1, np.array(resid)

    def reset_timers(self, reset=["setup:solver", "solution:solver"]):
        from sysmg.util.decorators import timings_data

        to_pop = []
        for k, v in timings_data.items():
            bools = [r in k for r in reset]
            if all(bools):
                to_pop.append(k)
        for k in to_pop:
            timings_data.pop(k)

    def get_setup_timings(self):
        """Return timings data for the Stokes system setup phase."""
        from sysmg.util.decorators import timings_data
        import pandas as pd

        # construct frame with all data: skip first 2 key identifiers ("setup:system:")
        setup_data = {}
        for k, v in timings_data.items():
            a, b, fun = k.split(":", 2)
            if f"{a}:{b}" == "setup:solver":
                setup_data[fun] = v
        frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in setup_data.items()]))
        # sum up each column and sort decreasing order
        idx = np.argsort(np.nansum(frame.values, axis=0))[::-1]
        # reorder the frame
        sorted_frame = frame[frame.columns[idx]]

        return sorted_frame

    def get_solution_timings(self):
        """Return timings data for the Stokes system setup phase."""
        from sysmg.util.decorators import timings_data
        import pandas as pd

        # construct frame with all data: skip first 2 key identifiers ("setup:system:")
        setup_data = {}
        for k, v in timings_data.items():
            a, b, fun = k.split(":", 2)
            if f"{a}:{b}" in ["solution:solver", "solution:mg"]:
                setup_data[fun] = v
        frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in setup_data.items()]))
        frame = frame.fillna(0)  # fill NaN with 0
        frame = frame.loc[
            (frame != 0).any(axis=1)
        ]  # Remove rows with all 0s in a Dataframe
        # sum up each column and sort decreasing order
        idx = np.argsort(np.nansum(frame.values, axis=0))[::-1]
        # reorder the frame
        sorted_frame = frame[frame.columns[idx]]

        # move 'rlx(c)' to the end
        # cols = list(sorted_frame.columns)
        # cols.remove('rlx(c)')
        # sorted_frame = sorted_frame[cols+[ 'rlx(c)']]

        rlx_timings = {}
        try:
            presmoother = self.wrapper.outer_relaxation
            # print(f'lvl={0}:\n\t', presmoother.timings)
            rlx_timings[0] = presmoother.timings
        except:
            pass
        for lvl, level in enumerate(self.ml.levels):
            presmoother = getattr(level, "relax_obj", None)
            try:
                # print(f'lvl={lvl+1}:\n\t', presmoother.timings)
                rlx_timings[lvl + 1] = presmoother.timings
            except:
                pass
        # print(rlx_timings)

        return sorted_frame

    def reset_solution_timings(self):
        """Return timings data for the Stokes system setup phase."""
        from sysmg.util.decorators import timings_data

        # construct frame with all data: skip first 2 key identifiers ("setup:system:")
        to_pop = []
        for k, v in timings_data.items():
            a, b, fun = k.split(":", 2)
            if f"{a}:{b}" in ["solution:solver", "solution:mg"]:
                to_pop.append(k)

        for k in to_pop:
            timings_data.pop(k)
