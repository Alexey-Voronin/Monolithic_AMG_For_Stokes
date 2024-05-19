import numpy as np
from firedrake import *
from scipy.sparse import csr_matrix, bmat, identity

from sysmg.solvers.gmg_interp.p1_unstruct_interp import contruct_unstruct_p1_interp
from sysmg.util.bmat import BlockMatrix
from sysmg.util.decorators import timeit
from .fe_factory.stokes import StokesProblem
from .system import System


class Stokes(System):
    """Stokes system object."""

    @timeit("setup:system:stokes:")
    def __init__(self, system_param, lo_fe_sys=False):
        """Initialize Stokes system Object.

        lo_fe_sys argument is only to be used in a recursive call in order to construct
        the low order finite element system.
        """
        if system_param is None:
            super().__init__(system_param)
            return

        self.disc = system_param["discretization"]
        self.elem_type = self.disc["elem_type"]
        self.elem_order = self.disc["order"]
        self.grad_div_stab = self.disc.get("grad_div_stab", None)
        self.bcs_type = self.disc.get("bcs", None)
        self.ordering = system_param["dof_ordering"]
        self.additional = system_param.get("additional", None)
        self.lo_fe_precond = lo_fe_sys
        self.debug = system_param.get("debug", False)
        self.keep = system_param.get("keep", False)

        super().__init__(system_param)

        self.form_params = system_param.get("additional", {})
        # Assemble FE system
        self._form_system(**self.form_params)
        # Split velocity field into components (x,y,z)
        # see the function for more details
        if self.ordering.get("split_by_component", False):
            self._split_by_component()
        # Sort each fields dofs by their coordinates.
        # This is needed for low-order preconditioning, at least until
        # macroelement are implemented.
        if self.ordering.get("lexicographic", False):
            assert self.ordering.get(
                "split_by_component", False
            ), """lexicographic ordering currently also requires
                split_by_component=True"""
            self._renumber_dofs()
        # GMG relic
        # TODO: revise when GMG comes back
        if self.form_params.get("grid_hierarchy", False):
            # mostly needed for GMG
            self._grid_hierarchy()
        # Construct low order finite element system
        if self.additional.get("lo_fe_precond", False):
            self._form_low_order_system()
        # verify that (self.A, self.b) gives the same solution as problem.
        if self.debug:
            self._debug()

        self.cleanup()  # frees memory and deletes unnecessary attributes

    @timeit("setup:system:stokes:")
    def _form_system(self, plot=False, solve=False, **kwargs):
        """Form the Stokes system.

        Delegates most of the work to the fe_factory.stokes.StokesProblem.
        The assembled system is stored blockwise as a BlockMatrix object.

        """
        self.problems = []
        for mesh in self.meshes:
            problem = StokesProblem(
                mesh, self.elem_order, self.elem_type, self.grad_div_stab
            )
            self.problems.append(problem)

        if self.elem_type[1] == "DG" or self.additional.get("cells", False):
            """only needed for geometric_dg Vanka"""
            self.dof_cells = []
            for space in [problem.Z for problem in self.problems[-1:]]:
                U, P = space.subfunctions
                P_cells = P.cell_node_list.copy()
                U_cells = U.cell_node_list.copy()
                self.dof_cells.append({"u": U_cells, "p": P_cells})
        ######################################################################
        # Weak form
        Z = self.problems[-1].Z
        a = problem.residual()
        L = problem.rhs()
        ######################################################################
        # BCs
        bcs_hier = [
            problem.bcs(self.bcs_type, self.boundary_marker)
            for problem in self.problems
        ]
        self.bcs_nodes_hier = self._get_bc_nodes(bcs_hier)
        self.bcs_hier = bcs_hier
        ######################################################################
        # Assemble the system
        bcs = bcs_hier[-1]
        A = assemble(a, bcs=bcs, mat_type="nest", sub_mat_type="aij")
        b = assemble(L, bcs=bcs)

        ######################################################################
        # Extract Operators and Save Submatrices in csr format
        # | M   B^T |
        # | B    C  |
        def _extract_op(A, idx):
            if idx is None:
                return csr_matrix((A.petscmat.getValuesCSR())[::-1])
            else:
                tmp = A.petscmat.getNestSubMatrix(*idx)
                return csr_matrix((tmp.getValuesCSR())[::-1])

        self.A_bmat = BlockMatrix(
            [
                [_extract_op(A, (0, 0)), _extract_op(A, (0, 1))],
                [_extract_op(A, (1, 0)), _extract_op(A, (1, 1))],
            ]
        )
        # the wrong way to get the right hand-side
        # stokes.b =  np.copy(b.vector().array())
        # the right way to form the right handside
        # read - https://www.firedrakeproject.org/boundary_conditions.html
        #  A x = b - action(A, zero_function_with_bcs_applied)
        # the code below was adopted from linear_solver.py:_lifted.py
        # from firedrake.assemble import create_assembly_callable
        utmp = Function(Z)
        utmp.dat.zero()
        for bc in A.bcs:
            bc.apply(utmp)

        expr = -ufl_expr.action(A.a, utmp)
        assemble(expr, tensor=b)

        blift = Function(Z)
        blift += b.riesz_representation(riesz_map="l2")
        for bc in A.bcs:
            bc.apply(blift)
        self.b = np.copy(blift.vector().array())
        ######################################################################
        # Auxiliary operators
        add_tmp = {} if not self.additional else self.additional
        bc_p = [bc if bc.function_space().name == "p" else None for bc in bcs_hier[-1]]
        bc_p = [bc for bc in bc_p if bc is not None]
        bc_u = [bc if bc.function_space().name == "u" else None for bc in bcs_hier[-1]]
        bc_u = [bc for bc in bc_u if bc is not None]

        for mat_type in ["mass", "stiffness"]:
            attr_name = f"ho_{mat_type}" if not self.lo_fe_precond else f"lo_{mat_type}"
            bmat = [[None, None], [None, None]]
            components = add_tmp.get(attr_name, ())
            for c in components:
                bc_i = bc_p if c == "p" else bc_u
                fxn = getattr(problem, mat_type + "_matrix")  # fxn return petsc matrix

                i = 0 if c == "u" else 1
                bmat[i][i] = _extract_op(
                    fxn(c, bc_i), (i, i)
                )  # None -> not block-ioperator

            if len(components) > 0:
                setattr(self, f"{mat_type}_bmat", BlockMatrix(bmat))
        ######################################################################
        # eliminate zeros, C can come out fairly dense with zeros.
        # firedrake spits out zero-matrix w/ pressure mass matrix sparsity
        # in case of C
        for i in range(self.A_bmat._nrows):
            for j in range(self.A_bmat._ncols):
                self.A_bmat[i, j].eliminate_zeros()
        ######################################################################
        # null-space vector depends on bc_type
        fnullspace = self.problems[-1].nullspace(self.bcs_type)
        if fnullspace is not None:
            fnullspace._build_monolithic_basis()
            nullspace_new = fnullspace._nullspace
            petscVec = nullspace_new.getVecs()[0]
            petscVec.assemble()
            self.nullspace = petscVec.getArray()
        ######################################################################
        # keep a random initial guess, x0, around for testing.
        # zero out dirichlet BCs, since they end up being ignored
        # anyway.
        np.random.seed(7)
        x0 = np.random.rand(self.A_bmat.shape[0])
        bcs_x = np.array(self.bcs_nodes_hier[-1]["u"])
        if len(bcs_x) > 0:
            nvx = self.velocity_nodes() // self.dim
            for i in range(self.dim):
                x0[bcs_x + nvx * i] = 0
        self.x0 = x0 / np.linalg.norm(x0)
        if isinstance(getattr(self, "nullspace", None), np.ndarray):
            null = self.nullspace
            x0 -= null * np.dot(null, x0)
        ######################################################################
        # rm dirichlet-boundary contribution from rhs. when it is left behind
        # it will mess with Vanka relaxation.
        # This is equivalent to removing dirichlet BCs from the matrix.
        bc_vx = np.array(self.bcs_nodes_hier[-1]["u"])
        # see node numbering notes in util.dof_handler for details..
        bc_v = np.hstack([bc_vx * self.dim + i for i in range(self.dim)])
        x_bc = np.zeros_like(self.b)
        self.b_dirichlet = self.b.copy()
        if len(bc_v):
            x_bc[bc_v] = self.b_dirichlet[bc_v]
        self.b -= self.A_bmat.tocsr() * x_bc
        ######################################################################
        self._construct_dof_coord()
        ######################################################################
        # solve the system
        if solve:
            _, self.up_sol = self._solve_sys(a, L, Z, bcs, nullspace=fnullspace)

    @timeit("setup:system:stokes:")
    def _form_low_order_system(self):
        """Form low-order system for preconditioning.
        This class assembles an equal order system (P1/P1, or Q1/Q1) on a once (uniformly) refined.
        The pressure space is then restricted to the original mesh. The restriction operators currently only work
        for the first order basis functions.
        """
        # TODO cleanup this function
        # 1. use the same meshing approach for both structured and unsturctured grids (might break GMG)
        assert self.ordering.get("lexicographic", False) and self.ordering.get(
            "split_by_component", False
        ), """Construction of Low-order preconditioner currently
                requires the system DoFs to be sorted."""
        from firedrake.mesh import MeshGeometry

        assert isinstance(self.mesh, MeshGeometry) or (
            isinstance(system_param["mesh"], dict)
            and system_param["mesh"]["mesh_hierarchy"]
        ), """Need to construct mesh hierarchy for lo_fe_precond"""

        # 1) Get Meshes - need both original and refined
        lo_disc = self.disc.copy()
        mesh = None
        if getattr(self, "bary", False):
            assert lo_disc["elem_type"] in [
                ("CG", "DG"),
                ("CG", "CG"),
            ], """lo_order_sys based on barycentric meshes  have
                only been tested with CG/DG element-pair."""
            mc, mf = MeshHierarchy(self.mesh, 1)
            lo_disc["elem_type"] = ("CG", "CG")
        else:
            mc, mf = MeshHierarchy(self.mesh, 1)  # .meshes

        factor = 4 if self.dim == 2 else 8
        assert (
            self.mesh.num_cells() * factor == mf.num_cells()
        ), "Low-order system mesh was not refined via quadsection-ref."

        # 2) Construct the P1/P1 system on the refined mesh
        lo_disc["order"] = (1, 1)
        form_params = self.form_params.copy()
        form_params["lo_fe_precond"] = False
        form_params["solve"] = False
        form_params["plot"] = False
        lo_sys_params = {
            "mesh": mf,
            "discretization": lo_disc,
            "dof_ordering": self.ordering,
            "additional": form_params,
        }
        lo_fe_sys = Stokes(lo_sys_params, lo_fe_sys=True)
        self.lo_fe_sys = lo_fe_sys

        assert lo_disc["elem_type"] in [("CG", "CG")], (
            "%s-element might not work here" % lo_disc["elem_type"]
        )
        assert lo_disc["order"] == (1, 1), "%s-element-order might not work here" % str(
            lo_disc["order"]
        )

        # 3) Construct P1 interpolation from coarse to fine
        # pressure field. It will be used to coarsen the pressure
        # field of the lo_fe_sys system.
        P_2n_to_n = contruct_unstruct_p1_interp(mc, mf)
        # 4) Reorder the DoFs in P_2n_to_n to match lexicographic
        # ordering of operators in self and lo_fe_sys.
        Pp_sort = None
        if getattr(self, "bary", False) and self.elem_type == ("CG", "DG"):
            # in case when the main pressure field is DG we need to
            # construct a sorting operator for unique pressure dofs
            mesh = self.mesh
            P1 = VectorFunctionSpace(mesh, "CG", 1)
            coord = interpolate(SpatialCoordinate(mesh), P1).dat.data_ro.copy()
            Pp_sort, _ = self._get_field_sorting_operator(coord)
            lo_fe_sys.dof_coord[-1]["p"] = Pp_sort * coord
        else:
            Pp_sort = self.Pbmat_sort[1, 1]

        try:
            P = lo_fe_sys.Pbmat_sort[1, 1] * P_2n_to_n * Pp_sort.T
        except:
            print("B_ho.shape=", self.A_bmat[1, 0].shape)
            print("B_lo.shape=", lo_fe_sys.A_bmat[1, 0].shape)
            print("lo_fe_sys.Pbmat_sort[1, 1].shape=", lo_fe_sys.Pbmat_sort[1, 1].shape)
            print("P_2n_to_n.shape=", P_2n_to_n.shape)
            print("Pp_sort.T.shape", Pp_sort.T.shape)
            raise Exception(
                """Low order iso discretization not currently
                working for discretizations P_{k}/P_{k-1} k > 2."""
            )
        lo_fe_sys.P_2n_to_n = P.tocsr()

        # 5) Restrict Pressure space to a coarser grid to get
        # Q1isoQ2-Q1 system
        Iu = identity(self.velocity_nodes())
        P = BlockMatrix([[Iu, None], [None, P]])
        R = P.T
        lo_fe_sys.A_bmat = R * (lo_fe_sys.A_bmat * P)

        for mat in ["mass_bmat", "stiffness_bmat"]:
            if hasattr(lo_fe_sys, mat):
                setattr(lo_fe_sys, mat, R * getattr(lo_fe_sys, mat) * P)
        # TODO: some of these are only used in GMG
        for dat in ["NE_hier", "NE", "grid_hier", "ext_grid_hier", "bcs_nodes_hier"]:
            if hasattr(self, dat):
                setattr(lo_fe_sys, dat, getattr(self, dat))

        lo_fe_sys.b = np.zeros_like(self.b)  # dummy
        if not getattr(self, "bary", False):
            lo_fe_sys.dof_coord = self.dof_coord

        if getattr(self, "bary", False):
            # Construct mapping between DG and CG discretizations
            assert (
                lo_fe_sys.velocity_nodes() == self.velocity_nodes()
            ), "M dim is wrong."
            assert (
                lo_fe_sys.velocity_nodes() == self.velocity_nodes()
            ), "M dim is wrong."

            Pv_map = identity(self.velocity_nodes())
            if self.elem_type[1] == "DG":
                dg = self.problems[-1].Z.sub(1).collapse()
                cg = FunctionSpace(dg.mesh(), "CG", 1)
                self.P_dg_to_cg = self._get_sv_to_th_operator(
                    dg, cg, self.Pbmat_sort[1, 1], self.velocity_nodes()
                )
                assert (
                    lo_fe_sys.ndofs() == self.P_dg_to_cg.shape[0]
                    and self.ndofs() == self.P_dg_to_cg.shape[1]
                ), "P_cg_to_dg_1to1 dimension is wrong."

                cg = lo_fe_sys.dof_coord[-1]["p"]
                dg = self.dof_coord[-1]["p"]
                # TODO: make this match _get_sv_to_th_operator
                Pp_map = self._get_cg_to_dg_operator(cg, dg)
            else:
                # can have Taylor-Hood rediscretization on barycentric
                # meshes too..
                pdofs = self.dof_coord[-1]["p"]
                Pp_map = identity(pdofs.shape[0])

            # 1-to-1 mapping from lower-order disc to higher-order

            assert (
                lo_fe_sys.pressure_nodes() == Pp_map.shape[1]
                and self.pressure_nodes() == Pp_map.shape[0]
            ), "P_cg_to_dg_1to1 dimension is wrong."
            self.P_cg_to_dg_1to1 = bmat(
                [[Pv_map, None], [None, Pp_map]], format="csr", dtype=bool
            )  # int)
            if self.keep:
                self.P_cg_to_dg_1to1_p_only = Pp_map

    @timeit("setup:system:stokes:")
    def _split_by_component(self):
        """Split the velocity field into Cartisian components.
        Original -> New ordering
        [u_x^0, u_y^0, u_z^0, ... u_x^n, u_y^n, u_z^n]
        -> [u_x^0, u_x^1, ..., u_y^0, u_y^1, ..., u_z^0, u_z^1, ...]
        """
        Pv = self._get_component_splitter(self.velocity_nodes())
        Ip = identity(self.pressure_nodes(), dtype=float, format="dia")
        self.Pbmat_split = BlockMatrix([[Pv, None], [None, Ip]])

        for mat in ["A_bmat", "mass_bmat", "stiffness_bmat"]:
            if hasattr(self, mat):
                setattr(
                    self,
                    mat,
                    self.Pbmat_split * (getattr(self, mat) * self.Pbmat_split.T),
                )

        for vector in ["b", "b_dirichlet", "up_sol", "nullspace", "x0"]:
            Pcsr = self.Pbmat_split.tocsr()
            if hasattr(self, vector):
                try:
                    setattr(self, vector, Pcsr * getattr(self, vector))
                except:
                    print("WARNING _split_by_component: %s not renumbered" % vector)

    @timeit("setup:system:stokes:")
    def _renumber_dofs(self):
        """
        Renumber dofs by sorting dofs by their coordinates.
        """
        # TODO: gmg data structures ...
        self.pdof_map = []
        self.vdof_map = []

        for coord_lvl, bcs_nodes in zip(
            self.dof_coord,
            self.bcs_nodes_hier,
        ):
            Ps = []
            for (cmpnt, coord), dmap in zip(
                coord_lvl.items(), [self.vdof_map, self.pdof_map]
            ):
                P, dof_map = self._get_field_sorting_operator(coord)
                Ps.append(P)
                coord_lvl[cmpnt] = P * coord
                if cmpnt in bcs_nodes:
                    bcs_nodes[cmpnt] = [dof_map[v] for v in bcs_nodes[cmpnt]]

                dmap.append(dof_map)

        if self.elem_type[1] == "DG":
            """only needed for geometric_dg Vanka"""
            dof_cells = self.dof_cells[-1]
            for cells, dof_map in zip(
                [dof_cells["u"], dof_cells["p"]], [self.vdof_map[-1], self.pdof_map[-1]]
            ):
                for i in range(cells.shape[0]):
                    for j in range(cells.shape[1]):
                        cells[i, j] = dof_map[cells[i, j]]

        Pvx, Pp = Ps
        Pv_sort = bmat(
            [
                [Pvx if i == j else None for i in range(self.dim)]
                for j in range(self.dim)
            ]
        ).tocsr()

        self.Pbmat_sort = BlockMatrix([[Pv_sort, None], [None, Pp]])
        for mat in ["A_bmat", "mass_bmat", "stiffness_bmat"]:
            if hasattr(self, mat):
                setattr(
                    self,
                    mat,
                    self.Pbmat_sort * (getattr(self, mat) * self.Pbmat_sort.T),
                )

        for vector in ["b", "b_dirichlet", "up_sol", "nullspace", "x0"]:
            Pcsr = self.Pbmat_sort.tocsr()
            if hasattr(self, vector):
                try:
                    setattr(self, vector, Pcsr * getattr(self, vector))
                except:
                    print("WARNING _renumber_dofs: %s not renumbered" % vector)

    def velocity_nodes(self):
        """Return the number of velocity nodes.
        During the data collection induvidual blocks may be deleted, so store
        the number of velocity nodes in a separate variable.
        """
        try:
            nv = self.A_bmat[0, 0].shape[0]
            self.nvelocities = nv
        except:
            pass

        return self.nvelocities

    def pressure_nodes(self):
        """Return the number of pressure nodes."""
        try:
            npressure = self.A_bmat[1, 1].shape[0]
            self.npressure = npressure
        except:
            pass

        return self.npressure

    def ndofs(self):
        """Return the number of dofs."""
        return self.velocity_nodes() + self.pressure_nodes()

    def _get_bc_nodes(self, bcs_hier):
        bcs_nodes_hier = []
        for bcs in bcs_hier:
            nodes = {"u": [], "p": []}
            for bc in bcs:
                component = bc.function_space().name
                nodes[component] += list(bc.nodes)

            bcs_nodes_hier.append({c: np.array(list(set(n))) for c, n in nodes.items()})

        return bcs_nodes_hier

    def plot(self, up, items=["u", "div u", "p", "grad p"], save_name=None):
        """Plot Solution."""
        problem = self.problems[-1]

        up0 = up.copy()
        # Unsort/split the vector if needed
        for P in ["Pbmat_sort", "Pbmat_split"]:
            if hasattr(self, P):
                PT_csr = getattr(self, P).tocsr().T
                up0 = PT_csr * up0

        # Transfer the coefficients to a firedrake Function
        f_up = Function(problem.Z)
        f_u, f_p = f_up.subfunctions
        nv = self.velocity_nodes()
        dim = problem.dim

        f_u.vector().dat.data[:, :] = up0[:nv].reshape((nv // dim, dim))
        f_p.vector().dat.data[:] = up0[nv:]

        problem.plot_solution(f_up, items=items, save_name=save_name)

    def save_solution(self, up, file_name="stokes_sol"):
        """Save solution to disk."""
        assert self.keep, """Need to keep splitting/sorting operators to use
                            this function."""
        # Unsort/split the vector if needed
        for P in ["Pbmat_sort", "Pbmat_split"]:
            if hasattr(self, P):
                PT_csr = getattr(self, P).tocsr().T
                up = PT_csr * up

        nv = self.velocity_nodes()
        u_sol = up[:nv]
        p_sol = up[nv:]

        up_empty = Function(self.problems[-1].Z)
        uguess, pguess = up_empty.subfunctions
        ##################################################
        # Velocity
        # move the result data into plotting object
        usol_renumbered = u_sol.reshape((nv // self.dim, self.dim))
        uguess.vector().dat.data[:, :] = usol_renumbered
        ##################################################
        # Pressure
        pguess.vector().dat.data[:] = p_sol
        self.problems[-1].save_function(
            up_empty, file_name=f"{file_name}_{nv+self.pressure_nodes()}"
        )

    def _debug(self):
        A0 = self.A_bmat.tocsr()
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2)
        ax1.spy(A0)
        if self.lo_fe_sys is not None:
            A1 = self.lo_fe_sys.A_bmat.tocsr()
            ax2.spy(A1)
        plt.show()

        # verify solution correctness
        if hasattr(self, "sol"):
            sol = spsolve(A0, self.b)
            resid_norm = np.linalg.norm(self.b - A0 * sol)
            P_split = self.Pbmat_split.tocsr()
            P_sort = self.Pbmat_sort.tocsr()

            nv = self.velocity_nodes()
            v_err_norm = np.linalg.norm(sol[:nv] - self.up_sol[:nv])
            sol = P_split.T * (P_sort.T * sol)

            # compute mean pressure
            tmp = (sol + np.inner(self.nullspace, sol))[nv:]
            tmp0 = tmp - np.mean(tmp)
            tmp1 = self.up_sol[nv:]
            p_err_norm = np.linalg.norm(tmp0 - tmp1)

            assert resid_norm < 1e-9, "residual is wrong: error=%.2e" % resid_norm
            assert p_err_norm < 1e-9, (
                "pressure solution is wrong: error=%.2e" % p_err_norm
            )
            assert v_err_norm < 1e-9, (
                "velocity solution is wrong: error=%.2e" % v_err_norm
            )

    def cleanup(self):
        def free_up(stks):
            def _del_attr(obj, attr):
                if hasattr(obj, attr):
                    delattr(obj, attr)

            del stks.meshes
            del stks.mesh
            _del_attr(stks, "grid_hierarchy")
            _del_attr(stks, "grid_hierarchy_ext")
            _del_attr(stks, "dof_coord")
            # _del_attr(stks, 'Pvp_sort')
            if hasattr(self, "lo_fe_sys"):
                # maybe needed for block-digonal solver when lo_fe_sys is not present
                _del_attr(stks, "Pp_sort")
            # _del_attr(stks, 'Pv_sort')
            # _del_attr(stks, 'P_split')
            _del_attr(stks, "pdof_map")
            _del_attr(stks, "vdof_map")
            _del_attr(stks, "mesh_orig")
            _del_attr(stks, "b_dirichlet")
            _del_attr(stks, "x0")

            if not hasattr(self, "grid_hier"):
                _del_attr(stks, "bcs_nodes_hier")
            from firedrake import mesh

            if isinstance(stks.params["mesh"], mesh.MeshGeometry):
                del stks.params["mesh"]

        # clean-up
        if not self.keep and not self.lo_fe_precond:
            free_up(self)
            if hasattr(self, "lo_fe_sys"):
                lo_stks = self.lo_fe_sys
                free_up(lo_stks)
                del lo_stks.b
                del lo_stks.P_2n_to_n
                # use the same vector for both hierarchies
                if hasattr(lo_stks, "nullspace"):
                    del lo_stks.nullspace
                    lo_stks.nullspace = self.nullspace
