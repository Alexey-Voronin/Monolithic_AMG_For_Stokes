import scipy.sparse as sp
import numpy as np
from pyamg.multilevel import coarse_grid_solver, multilevel_solver
import gc

# sysmg imports
from .mg_common import MG
from .relaxation.vanka import Vanka
from .relaxation.lsc import LSC
from .ho_wrapper import HighOrderWrapper
from sysmg.util.bmat import BlockMatrix
from sysmg.util.alg_util import dropSmallEntries
from sysmg.util.decorators import timeit

# analysis
import pandas as pd


class StokesMG(MG):
    """StokesMG Object.

    Attributes
    ----------
    ml : pyamg multilevel_solver object
        - multigrid hierarchy
    ho_system : HO_FE_System object
        - high order finite element system. Differ from system if system is
        low order. this object is used to construct the wrapper.
    wrapper : HighOrderWrapper object
        - wrapper for high order finite element system
    dim : int
        - dimension of the system
    interpolation : dict
        - interpolation parameters
    levels : int
        - number of levels in the hierarchy
    relaxation : dict
        - relaxation parameters
    cg_solve : str
        - coarse grid solver
    keep : bool
        - option to keep/discard multigrid setup objects not necessary for
        the actual solve.
    cycle : str
        - multigrid cycle
    """

    @timeit("setup:solver:mg:")
    def __init__(
        self,
        stokes_system,
        mg_param,
        keep=True,
    ):
        """Initialize StokesMG Object.


        Parameters
        ----------
        stokes_system : Stokes system object
            - finite element system used to construct multigrid hierarchy
        mg_param : dict
            - multigrid parameters
            - Example:
                vanka_outer = ('Vanka', { 'iterations': (1,1), 'accel' : {'iterations' : (2,2)},
                                          'type' : 'algebraic','update': 'additive', 'omega' : 1.,
                                          'patch_solver' : 'inv'
                                        })
                vanka_inner = vanka_outer
                mg_param   = {   'interpolation' :
                                      {  'type'  : 'algebraic',
                                         'order' : 'low',
                                         'params':{ 'u'    : {'strength' : 'evolution', 'max_coarse' : 50},
                                                    'p'    : {'agg_mat': ('stiffness', 0.0),
                                                              'smooth_mat': ('stiffness', 0.0),
                                                              'strength' : 'evolution',
                                                              'max_coarse' : 50}
                                                    }
                                       },
                                 'relaxation'        : vanka_inner,
                                 'wrapper_params' : {'modify_mg_rlx' : [(0,(0,0))],
                                                     'relax_params' : vanka_outer,
                                                     'tau' : (1,1),
                                                     'eta' : ((0.92, 0.92), (1,1))
                                                     },
                                 'levels'            : 10,
                                 'coarse_grid_solve' : 'splu',
                             }
        keep: bool
            - option to keep/discard multigrid setup objects not necessary for
            the actual solve.
        """
        import copy

        mg_param_c = copy.deepcopy(mg_param)  # gets modified later

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
        self.interpolation = mg_param_c["interpolation"]
        self.levels = mg_param_c["levels"]
        self.relaxation = mg_param_c.get("relaxation", None)
        self.cg_solve = mg_param_c.get("coarse_grid_solve", "pinv2")
        self.keep = keep
        self.cycle = mg_param_c.get("cycle", "V")
        ################################################
        # compute grid hierarchy
        levels = self._setup_hierarchy()
        try:
            del self.mg_param["interpolation"]["params"]["p"]["agg_mat"]
        except:
            pass
        ################################################
        # relaxation of level's bmats to csr
        rlx_levels = self._setup_relaxation(levels)
        ################################################
        # convert matrices from bmat to csr
        ml = multilevel_solver(levels)
        # map relaxation to pyamg hierarchy
        for lvl, rlx in zip(range(len(ml.levels) - 1), rlx_levels):
            ml.levels[lvl].presmoother = rlx.presmoother
            ml.levels[lvl].postsmoother = rlx.postsmoother
            ml.levels[lvl].relax_obj = rlx

        # coarse-grid solve
        if ml.levels[-1].A.shape[0] > 1200:
            self.cg_solve = "splu"
        ml.coarse_solver = coarse_grid_solver(self.cg_solve)
        # coarse grid solver gets built during first call
        # force construction here for timing purposes
        ml.coarse_solver(ml.levels[-1].A, np.ones((ml.levels[-1].A.shape[0],)))

        @timeit("solution:solver:coarse:")
        def coarse_solve(A, b):
            return ml.coarse_solver_obj(A, b)

        ml.coarse_solver_obj = ml.coarse_solver
        ml.coarse_solver = coarse_solve

        self.ml = ml
        # TODO: make this work for geometric
        if self.interpolation["type"] == "algebraic":
            self.__repr__()

        # don't leave reference here
        if self.interpolation["type"] == "algebraic":
            del self.system

        # force construction
        self.aspreconditioner(self.cycle)

        def _del_attr(obj, attr):
            if hasattr(obj, attr):
                delattr(obj, attr)

        if not keep:
            _del_attr(self, "ho_system")
            if hasattr(self, "wrapper"):
                _del_attr(self, "stokes")

        gc.collect()  # force garbage collection

    @timeit("setup:solver:mg:")
    def _setup_hierarchy(self):
        """setup the multilevel hierarchy"""
        system = self.system

        if self.interpolation["type"] == "algebraic":
            Ps_bmat = self._get_amg_interp()
        elif self.interpolation["type"] == "geometric":
            Ps_bmat = self._get_gmg_interp()
        else:
            raise ValueError(
                "Interpolation type %s not defined." % self.interpolation["type"]
            )

        #################################
        # Construct Multigrid Hierarchy #
        #################################
        levels = []
        levels.append(multilevel_solver.level())

        # temporary stores in ml object for algebraic coarsening
        for mat in ["mass_bmat", "stiffness_bmat"]:
            if hasattr(system, mat):
                setattr(levels[-1], mat, getattr(system, mat))

        levels[-1].A_bmat = system.A_bmat
        # there may not be as many grids as requested
        self.levels = min(len(Ps_bmat) + 1, self.levels)
        for lvl in range(self.levels - 1):
            # Interpolation
            levels[-1].P_bmat = P = Ps_bmat[lvl]
            levels[-1].R_bmat = R = P.T
            #  Ritz-Galerkin Coarsening
            levels.append(multilevel_solver.level())
            levels[-1].A_bmat = R * levels[-2].A_bmat * P

            for mat in ["mass_bmat", "stiffness_bmat"]:
                if hasattr(levels[-2], mat):
                    setattr(levels[-1], mat, R * getattr(levels[-2], mat) * P)

        return levels

    def aspreconditioner(self, cycle="V"):
        """return a linear operator that can be used as a preconditioner"""
        wrapper_params = self.mg_param.get("wrapper_params", None)

        if wrapper_params is not None:
            # Fewer iterations on lvl=0 of the lo system?
            lo_iters_fix = wrapper_params.get("modify_mg_rlx", [])
            if lo_iters_fix:
                for lvl, (pre, post) in lo_iters_fix:
                    self.ml.levels[lvl].relax_obj.relax_iters = (pre, post)

            # wrap the low-order preconditioner
            if hasattr(self.wrapper, "M") and self.wrapper.cycle == cycle:
                M = self.wrapper.M
            else:
                M = self.wrapper.construct_preconditioner(self.ml, cycle)
        else:
            M = self.ml.aspreconditioner(cycle=cycle)

        return M

    @timeit("setup:solver:mg:")
    def _setup_relaxation(self, levels):
        """setup the relaxation objects"""
        system = getattr(self, "system", None)

        rlx_name, rlx_param = self.relaxation
        # used to pass submatrices and grids to relaxation
        from sysmg.systems import Stokes

        class Stokes_tmp(Stokes):
            def __init__(self, A_bmat, dim, structured):
                super().__init__(None)
                self.A_bmat = A_bmat
                self.dim = dim
                self.structured = structured

        # when mg_wrapper is present, relaxation on lvl=0
        # is not always necessary. Check which levels can be skipped.
        wrapper_params = self.mg_param.get("wrapper_params", None)
        skip_rlx_lvls = []
        if wrapper_params:
            for lvl, pre_post in wrapper_params.get("modify_mg_rlx", []):
                if pre_post == (0, 0):
                    skip_rlx_lvls.append(lvl)

            # Set-Up outer relaxation too
            # Add outer relaxation if needed
            self.wrapper = HighOrderWrapper(
                self.ho_system, wrapper_params, keep=self.keep
            )

        stks_levels = []
        rlx_levels = []
        for i, level in enumerate(levels):  # exact solve on level[-1]
            # Stokes object that will be passed to relaxation

            level.A_bmat[0, 0].eliminate_zeros()
            level.A_bmat[1, 0].eliminate_zeros()
            level.A_bmat[0, 1].eliminate_zeros()
            stks = Stokes_tmp(level.A_bmat, system.dim, system.structured)
            stks.stiffness_bmat = system.stiffness_bmat if i == 0 else None
            stks.mass_bmat = system.mass_bmat if i == 0 else None

            stks.dim = self.dim
            stks_levels.append(stks)
            level.A = level.A_bmat.tocsr()
            level.A.eliminate_zeros()
            if i == len(levels) - 1:  # exact-solve here, no relax needed
                continue

            # for mat in ['mass_bmat', 'stiffness_bmat']:
            #    if hasattr(levels[-2], mat):
            #        setattr(levels[-1], mat, R * getattr(levels[-2], mat) * P)

            if rlx_name.lower() == "vanka" and rlx_param["type"] == "geometric":
                stks.grid_hier = [system.grid_hier[-1 - i]]
                stks.ext_grid_hier = [system.ext_grid_hier[-1 - i]]
                stks.periodic = system.periodic
            elif rlx_name.lower() == "lsc":
                pass
            elif rlx_param["type"] == "geometric_dg" and i == 0:
                stks.dof_cells = system.dof_cells

            empty_shell = True if i in skip_rlx_lvls else False
            rlx_param_copy = rlx_param
            if empty_shell:
                from copy import deepcopy

                rlx_param_copy = deepcopy(rlx_param)
                rlx_param_copy["iterations"] = (0, 0)

            if rlx_name.lower() == "vanka":
                # On coarser AMG levels the components of the vector Lapalcian are not always
                # of the same size. This is due to the fact that the Laplacian is constructed
                # one component at a time.
                stks._udofs_component_wise = self.mls_info["dofs"][i][: self.dim]
                relax = Vanka(stks, params=rlx_param_copy, level=(i + 1))
            elif rlx_name.lower() == "lsc":
                relax = LSC(stks, params=rlx_param_copy, level=(i + 1))
            else:
                raise ValueError("%s Relaxation not defined" % rlx_name.lower())

            rlx_levels.append(relax)
            level.P = level.P_bmat.tocsr()
            level.R = level.P.T.tocsr()

            del stks
            if not self.keep:
                del level.A_bmat
                for mat in [
                    "mass_bmat",
                    "stiffness_bmat",
                    "P_bmat",
                    "R_bmat",
                    "A_bmat",
                ]:
                    if hasattr(level, mat):
                        delattr(level, mat)

                if i == 0:  # delete stokes object data as well
                    for attr in ["dof_cells"]:
                        try:
                            delattr(self.system, attr)
                        except Exception as e:
                            pass
            gc.collect()  # force garbage collection

        if self.keep:
            self.stks_levels = stks_levels

        return rlx_levels

    def _get_pressure_matrix(self, mat_name, component):
        """Get Pressure Matrix.

        Allows user to either input a csr_matrix directly
        into the argument list or provide a shorthand reference
        to what the desired pressure matrix should be.

        The current options are:
        - csr_matrix
        - string:
            - 'BBT': divergence times the gradient operator
            - 'BDBT': same as abovr but scaled by diagonal of vector-laplacian
            - 'Stiffness': pressure stiffness operator
            - 'Mass': pressure mass matrix
        - component: string ('u' or 'p')
        """
        if isinstance(mat_name, sp.csr_matrix):
            M = mat_name
            return M

        system = self.system
        mat_name = mat_name.lower()
        idx = 0 if component == "u" else 1
        if mat_name == "bdbt":
            assert idx == 1, "BDBT only defined for pressure component"
            B = system.A_bmat[1, 0]  # B
            BT = system.A_bmat[0, 1]  # B.T
            C = system.A_bmat[0, 0].diagonal()
            nnz = C.nonzero()
            C_inv = np.ones(C.shape)
            C_inv[nnz] = 1.0 / C[nnz]
            D_inv = sp.spdiags(C_inv, 0, C.size, C.size)
            M = B * (D_inv * BT)
        elif mat_name == "bbt":
            assert idx == 1, "BBT only defined for pressure component"
            B = system.A_bmat[1, 0]
            BT = system.A_bmat[0, 1]
            M = B * BT
        elif mat_name == "stiffness":
            M = system.stiffness_bmat[idx, idx]
        elif mat_name == "mass":
            M = system.mass_bmats[idx, idx]
        else:
            raise ValueError("%s Pressure matrix not defined" % mat_name)

        if not sp.isspmatrix_csr(M):
            M = M.tocsr()

        return M

    @timeit("setup:solver:mg:")
    def _get_amg_interp(self, **kwargs):
        """Get AMG Interpolation Operator.

        This function interfaces with the pyamg package to construct an
        interpolation operator for each field. The user can specify the
        smoothed aggregation parameters to be used for each field.
        """
        params = self.interpolation["params"]
        system = self.system
        ################################################################
        # Velocity Interpolation
        M = system.A_bmat[0, 0]
        dim = self.dim
        nvx = M.shape[0] // dim
        # No need to create relaxtion
        params["u"]["presmoother"] = None
        params["u"]["postsmoother"] = None

        for d, name in zip(range(dim), ["u_x", "u_y", "u_z"]):
            Mi = sp.csr_matrix(M[nvx * d : nvx * (d + 1), nvx * d : nvx * (d + 1)])
            ml_vi = super()._get_amg_interp(Mi, params["u"])
            self._process_ml(ml_vi, field="u")
        ################################################################
        # Pressure Interpolation
        p_params = params["p"]
        params["p"]["presmoother"] = None
        params["p"]["postsmoother"] = None

        agg_drop_tol = 0.0
        smooth_drop_tol = 0.0
        if type(p_params["agg_mat"]) is tuple:
            agg_mat, agg_drop_tol = p_params["agg_mat"]
        else:
            agg_mat = p_params["agg_mat"]
        if type(p_params["smooth_mat"]) is tuple:
            smooth_mat, smooth_drop_tol = p_params["smooth_mat"]
        else:
            smooth_mat = p_params["smooth_mat"]
        p_params.pop("agg_mat")
        p_params.pop("smooth_mat")

        agg_mat = self._get_pressure_matrix(agg_mat, "p")
        smooth_mat = self._get_pressure_matrix(smooth_mat, "p")
        # trim stencil if needed
        if hasattr(p_params, "drop_tol"):
            raise ValueError(
                """Parameter deprecated. Pass in as tuple to
                            Agg/Smooth-Mat."""
            )

        if agg_drop_tol > 0:
            agg_mat = dropSmallEntries(agg_mat, agg_drop_tol)
        if agg_drop_tol > 0:
            smooth_mat = dropSmallEntries(smooth_mat, smooth_drop_tol)

        # Currently assuming that the smoothing matrix is also the matrix
        # being coarsened.
        p_params["agg_mat"] = agg_mat
        ml_p = super()._get_amg_interp(smooth_mat, p_params)
        self._process_ml(ml_p, field="p")
        ################################################################
        # combine the above field matrices in block-diagonal matrix
        # of type bmat.
        nlvls = np.min([len(self.Ps["u"].values()), len(self.Ps["p"])])
        Ps_bmat = []
        for lvl in range(nlvls):
            Pv_lvl = self.Ps["u"][lvl]
            Pp_lvl = self.Ps["p"][lvl]
            # Velocity (u_x, u_y, ..) interpolation
            Pv_lvl_bmat = [[None for _ in range(dim)] for _ in range(dim)]
            for i in range(dim):
                Pv_lvl_bmat[i][i] = Pv_lvl[i]
            Pv_lvl_bmat = sp.bmat(Pv_lvl_bmat, format="csr")
            # Add pressure interp
            Pi = BlockMatrix([[Pv_lvl_bmat, None], [None, Pp_lvl[0]]])
            Ps_bmat.append(Pi)

        del self.Ps
        return Ps_bmat

    @timeit("setup:solver:mg:")
    def _get_gmg_interp(self, **kwargs):
        self._init_data_structs()

        P_vx = super()._get_gmg_interp(field="u", ignore_bcs=False)
        P_p = super()._get_gmg_interp(field="p", ignore_bcs=False)

        # combine the above field matrices in block-diagonal matrix
        # of type bmat.
        Ps_bmat = []
        nlvls = min(len(P_vx), len(P_p), self.levels - 1)
        dim = self.dim
        for lvl in range(nlvls):
            P_vx_i = P_vx[lvl]
            # (u_x, u_y, ..) interpolation
            P_v_i = []
            for i in range(dim):
                P_v_tmp = [None] * dim
                P_v_tmp[i] = P_vx_i
                P_v_i.append(P_v_tmp)

            P_v_i = sp.bmat(P_v_i).tocsr()
            # (u,p) interp
            Pi = BlockMatrix([[P_v_i, None], [None, P_p[lvl]]])
            Ps_bmat.append(Pi)

        return Ps_bmat

    def __repr__(self):
        """Print basic statistics about the multigrid hierarchy.

        Inf-sup is a bit hackey - clean it up later.
        """
        # (pre)compute this call once
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
        output += f"Coarse Solver:        {ml.coarse_solver_obj.name()}\n"

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
            if self.dim == 2:
                output += f"       [{crs[0]:>5},{crs[1]:>5},{crs[2]:>5}]"
            else:
                output += f"       [{crs[0]:>5},{crs[1]:>5},{crs[2]:>5},{crs[3]:>5}]"

            output += "\n"

        self.image = output
        return output

    def to_dict(self):
        """Return a dictionary containing the multigrid hierarchy
        setup information.

        This is useful for saving the hierarchy to disk.
        """
        # Default PyAMG hierarchy info
        mg_info = self.ml.to_dict()
        # Ratio of velocity to pressure DoFs
        mg_info.update({"Ratio[v/p]": {}})

        # Coarsening Rates, DoFs on each grid, and Complexities
        if self.dim == 2:
            add_fields = {
                "CR(u_x)": {},
                "CR(u_y)": {},
                "CR(p)": {},
                "dofs(u_x)": {},
                "dofs(u_y)": {},
                "dofs(p)": {},
                "O(ml)": {},
                "O(ml(A))": {},
                "O(u_x)": {},
                "O(u_y)": {},
                "O(p)": {},
                "eigs(MA)": {},
            }
        else:
            add_fields = {
                "CR(u_x)": {},
                "CR(u_y)": {},
                "CR(u_z)": {},
                "CR(p)": {},
                "dofs(u_x)": {},
                "dofs(u_y)": {},
                "dofs(u_z)": {},
                "dofs(p)": {},
                "O(ml)": {},
                "O(ml(A))": {},
                "O(u_x)": {},
                "O(u_y)": {},
                "O(u_z)": {},
                "O(p)": {},
                "eigs(MA)": {},
            }
        mg_info.update(add_fields)
        mg_info.update(
            {
                "Aloc(mean)": {},
                "Aloc(std)": {},
                "Aloc(min)": {},
                "Aloc(max)": {},
                "Aloc(sum)": {},
                "Aloc(sq_sum)": {},
                "Aloc(num)": {},
            }
        )

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

            mg_info["CR(u_x)"][lvl] = crs[0]
            mg_info["CR(u_y)"][lvl] = crs[1]
            if self.dim == 3:
                mg_info["CR(u_z)"][lvl] = crs[2]
            mg_info["CR(p)"][lvl] = crs[-1]

            # dofs on each grid
            mg_info["dofs(u_x)"][lvl] = dofs[0]
            mg_info["dofs(u_y)"][lvl] = dofs[1]
            if self.dim == 3:
                mg_info["dofs(u_z)"][lvl] = dofs[2]
            mg_info["dofs(p)"][lvl] = dofs[-1]
            ######################################################
            # eigenvalues from Krylov acceleration
            presmoother = getattr(level, "relax_obj", None)
            if presmoother and hasattr(presmoother, "patch_stats"):
                mg_info["eigs(MA)"][lvl] = presmoother.eigs.round(4)
                for k, v in presmoother.patch_stats.items():
                    mg_info[f"Aloc({k})"][lvl] = v
            elif (
                presmoother == None
                and lvl == len(self.ml.levels) - 1
                and hasattr(self, "wrapper")
                and self.wrapper.outer_relaxation != None
                and hasattr(self.wrapper.outer_relaxation, "patch_stats")
            ):
                # save outer relaxation info on the coarse-level
                presmoother = self.wrapper.outer_relaxation
                mg_info["eigs(MA)"][lvl] = presmoother.eigs.round(4)
                for k, v in presmoother.patch_stats.items():
                    mg_info[f"Aloc({k})"][lvl] = v
            else:  # coarse level doe not have a smoother
                mg_info["eigs(MA)"][lvl] = 0.0
                for k in ["mean", "std", "min", "max", "sum", "sq_sum", "num"]:
                    mg_info[f"Aloc({k})"][lvl] = None
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

            mg_info["O(u_x)"][lvl] = complexities[0]
            mg_info["O(u_y)"][lvl] = complexities[1]
            if self.dim == 3:
                mg_info["O(u_z)"][lvl] = complexities[2]
            mg_info["O(p)"][lvl] = complexities[-1]

            # if lvl == 0
            # if lvl < len(self.ml.levels)-1:
            #    presmoother = self.wrapper.outer_relaxation
            #    for k,v in presmoother._patch_stats.items():
            #        mg_info[f"Aloc({k})"]

        return mg_info

    def to_pandas(self):
        """
        Convert the multigrid information dictionary to a pandas dataframe
        """
        mg_info = self.to_dict()
        data = {k: list(v.values()) for k, v in mg_info.items()}
        return pd.DataFrame.from_dict(data)
