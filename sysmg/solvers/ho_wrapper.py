import numpy as np
import scipy.sparse as sp

from sysmg.util.decorators import timeit
from .relaxation.vanka import Vanka


class HighOrderWrapper(object):
    """
    Wrapper that "glues" together higher-order and lower-order systems and solvers.

    This class is used to solve the higher-order system using a lower-order solver.
    This usually includes an additional set of relaxation steps defined on the
    higher-order system.
    """

    @timeit("setup:solver:relaxation:")
    def __init__(self, stokes, params=None, keep=False):
        """
        Initialize wrapper object.

        Parameters
        ----------
        stokes : StokesSystem
            Stokes system object.
        params : dict
            Dictionary with parameters for the wrapper.
        keep : bool
            If True, the Stokes system is kept in memory.
        """

        self.stokes = stokes  # higher-order system
        self.keep = keep
        self.relax_setup_time = 0
        self.A_nnz = stokes.A_bmat.tocsr().nnz
        self.elem_order = stokes.elem_order
        self.elem_type = stokes.elem_type

        self.params = params
        self.tau = self.params.get('tau', (1, 1))
        self.eta = self.params.get('eta', [(1, 1), (1, 1)])
        self.gamma = self.params.get('gamma', 1)

        if params.get("relax_params"):
            self.outer_relaxation = self._setup_relaxation(params["relax_params"])
        else:
            self.outer_relaxation = None

    @timeit("setup:solver:relaxation:")
    def _setup_relaxation(self, params):
        """
        Setup relaxation object for outer (higher-order) relaxation.
        """
        relax_name, relax_params = params
        if relax_name.lower() == "vanka":
            rlx = Vanka(self.stokes, params=relax_params)
        elif relax_name.lower() == "uzawa":
            from sysmg import BlockDiagMG
            rlx = BlockDiagMG(self.stokes, relax_params)
        else:
            raise ValueError("%s relaxation has not been integrated yet." % relax_name)

        # remove operators that will not be used anymore
        if not self.keep:
            system = self.stokes
            for attr in ['B', 'BT', 'C', 'M', 'dof_cells']:
                try:
                    delattr(system, attr)
                except:
                    pass

        return rlx

    def construct_preconditioner(self, ml, cycle='V'):
        """
        Construct preconditioner for higher-order system.
        This function figures out whether higher-order system is TH or SV.
        """

        self.cycle = cycle
        assert self.elem_order == (2, 1), \
            "Only element orders (2,1) have been tested."
        disc = self.elem_type
        if disc == ("CG", "CG"):
            self.M = self._th_preconditioner(ml, cycle)
        elif disc == ("CG", "DG"):
            self.timings = {'P_01(t)': 0.0, 'P_01(c)': 0,
                            'P_10(t)': 0.0, 'P_10(c)': 0}
            self.M = self._sv_preconditioner(ml, cycle)
        else:
            raise ValueError("%s discretization has not been integrated yet." % disc)

        if not self.keep:
            delattr(self, "stokes")

        return self.M

    @timeit("setup:solver:relaxation:")
    def _th_preconditioner(self, ml, cycle):
        """
        Construct preconditioner for TH discretization.

        Parameters
        ----------
        ml : pyamg multilevel object
            Lower-order multilevel object.
        cycle : str
            Multigrid cycle to be used. Either 'V' or 'W'.
        """

        stokes = self.stokes
        A0 = stokes.A_bmat.tocsr()
        self.x = np.zeros(A0.shape[0])
        nv = stokes.velocity_nodes()
        outer_rlx = self.outer_relaxation

        if self.outer_relaxation != None:
            self.dx = np.zeros(A0.shape[0])

            @timeit("solution:solver:wrapper:")
            def mv(b):
                tau = self.tau
                eta_0, eta_ell = self.eta
                gamma = self.gamma
                x = self.x;
                x *= 0.
                dx = self.dx;
                dx *= 0.

                #  MG correction
                # using __solve because avoids unnecessary residual comp
                outer_rlx.presmoother(A0, x, b)
                r = b - A0 * x
                # residual overcorrection
                r[:nv] = eta_0[0] * r[:nv]
                r[nv:] = eta_0[1] * r[nv:]
                for _ in range(gamma):
                    ml._MultilevelSolver__solve(0, dx, r, cycle, eta_corr=(nv, eta_ell))

                x[:nv] += tau[0] * dx[:nv]
                x[nv:] += tau[1] * dx[nv:]
                outer_rlx.postsmoother(A0, x, b)
                return x
        else:
            @timeit("solution:solver:wrapper:")
            def mv(b):
                x = self.x;
                x *= 0.
                gamma = self.gamma
                eta_0, eta_ell = self.eta
                tau = self.tau
                # tau does not do anything here when used as a preconditioner for FGMRES
                # dx = tau*(M*b)
                for _ in range(gamma):
                    ml._MultilevelSolver__solve(0, x, b, cycle, eta_corr=(nv, eta_ell))
                x[:nv] *= tau[0]
                x[nv:] *= tau[1]

                return x

        M_full = sp.linalg.LinearOperator(A0.shape, matvec=mv)
        return M_full

    @timeit("setup:solver:relaxation:")
    def _sv_preconditioner(self, ml, cycle):
        """
        Construct preconditioner for SV discretization.
        This function differs from _th_preconditioner in that it
        uses non-identity operators to map between DG to CG
        pressure space.

        Parameters
        ----------
        ml : pyamg multilevel object
            Lower-order multilevel object.
        cycle : str
            Multigrid cycle to be used. Either 'V' or 'W'.
        """

        stokes = self.stokes
        A0 = stokes.A_bmat.tocsr()
        assert A0.shape[0] == stokes.ndofs(), 'A0 dimensions are wrong.'
        self.x = np.zeros(A0.shape[0])
        self.dx = np.zeros(stokes.lo_fe_sys.ndofs())
        # Grid Transfer Operators
        P_1to0 = stokes.P_cg_to_dg_1to1
        P_0to1 = stokes.P_dg_to_cg
        # null1 might be left out by accident in some cases (no relax on l=0)
        null1 = None
        if stokes.bcs_type == 'washer':
            null1 = np.zeros_like(self.dx)
            null1[stokes.lo_fe_sys.velocity_nodes():] = 1

        outer_rlx = self.outer_relaxation

        @timeit("solution:solver:wrapper:")
        def mv(b):
            nv = stokes.velocity_nodes()
            tau = self.tau
            eta_0, eta_ell = self.eta
            gamma = self.gamma
            dx = self.dx;
            dx *= 0.
            x = self.x;
            x *= 0.

            if self.outer_relaxation != None:
                outer_rlx.presmoother(A0, x, b)

            r0 = b - A0 * x
            # residual overcorrection
            r0[:nv] = eta_0[0] * r0[:nv]
            r0[nv:] = eta_0[1] * r0[nv:]
            # Restrict to K_1
            r = P_0to1 * r0

            # K_1 MG Cycle
            for _ in range(gamma):
                ml._MultilevelSolver__solve(0, dx, r, cycle, eta_corr=(nv, eta_ell))
            """
            if gamma == 1:
                ml._MultilevelSolver__solve(0, dx, r, cycle, eta_corr=(nv, eta_ell))
            else:
                dx = ml.solve(r, tol=1e-25, maxiter=gamma, cycle=cycle)
            """
            if hasattr(null1, '__len__'):
                dx -= np.dot(null1, dx) * null1 / np.dot(null1, null1)

            # interpolate correction from K_1 to K_0
            tmp = P_1to0 * dx
            x[:nv] += tau[0] * tmp[:nv]
            x[nv:] += tau[1] * tmp[nv:]

            if self.outer_relaxation != None:
                outer_rlx.postsmoother(A0, x, b)

            return x

        M_full = sp.linalg.LinearOperator(A0.shape, matvec=mv)
        return M_full

    def set_tau(self, tau):
        self.tau = tau

    def set_eta(self, eta):
        self.eta = eta
