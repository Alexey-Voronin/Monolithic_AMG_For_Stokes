import numpy as np
import scipy.sparse as sp
from firedrake import *

from sysmg.util.decorators import timeit


class DoFHandler(object):
    """Class that deals with geometric aspect of discretized PDE systems."""

    def __init__(self):
        """Init DoFHandler.

        DoFHandler class is responsible for dealing with degree of
        freedom orderings.
        """
        pass

    #####################################################
    # DoF renumbering
    @timeit("setup:system:DoFHandler:")
    def _construct_dof_coord(self):
        """Get DoF coordinates for each field.

        Construct dictionaries containing DoFs coordinates for each field
        for each level of the hierarchy
        """
        elems = self.elem_type
        self.dof_coord = []
        for Vm in [p.Z for p in self.problems]:
            coords = {}
            for V, elem in zip(
                Vm.subfunctions if type(elems) in [tuple, list] else Vm,
                elems if type(elems) in [tuple, list] else [elems],
            ):
                mesh = V.mesh()
                order = V.ufl_element().degree()
                if V.ufl_element().family() == "Lagrange": 
                    Vspace = VectorFunctionSpace(mesh, elem, order)
                else: #DG
                    # The default varient "spectral" breaks _get_cg_to_dg_operator
                    # because DG and CG DoFs are no longer collocated.
                    elem0 = FiniteElement("DG", mesh.ufl_cell(), degree=order, variant='equispaced')
                    Vspace = VectorFunctionSpace(mesh, elem0)
                coord_field = interpolate(
                    SpatialCoordinate(mesh), Vspace
                ).dat.data_ro.copy()
                coords[V.name] = coord_field

            self.dof_coord.append(coords)

    @timeit("setup:system:DoFHandler:")
    def _get_component_splitter(self, nDoFs):
        """Create a matrix that splits vector-valued-matrices by component.

        Create a mapping operator, P, that splits the DoFs
        component wise. The default DoF ordering in firdrake is
        nodal - [u_x^1, u_y^1, .., u_x^n, u_y^n].

        Operator P reorders the system variable such that they
        are listed component-wise -
        [u_x^1, u_x^2, ..,  u_x^n, u_y1n, ..,  u_y^n].
        """
        nv = nDoFs
        dim = self.dim
        nvx = nv // dim
        cols = np.linspace(0, nDoFs - 1, nDoFs, dtype=int)
        rows = np.zeros_like(cols)

        for d in range(dim):
            for j in range(0, nvx):
                rows[j * dim + d] = j + nvx * d

        Pv = sp.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), shape=(nv, nv), dtype=int
        )

        return Pv

    @timeit("setup:system:DoFHandler:")
    def _get_field_sorting_operator(self, coord):
        """Create a linear operator that sorts DoFs in lexicographic order.

        Creates mapping (csr_matrix) that reorders the dofs
        from firedrake ordering to the lexicographic.
        """
        coord = coord.T.round(12)
        ind = np.lexsort(coord) if self.dim > 1 else np.argsort(coord)

        dof_map = dict()
        rows = []
        cols = []
        for j in range(len(ind)):
            rows.append(j)
            cols.append(ind[j])
            dof_map[ind[j]] = j

        Nx = coord.T.shape[0]
        P = sp.coo_matrix(
            (np.ones(len(rows)), (np.array(rows), np.array(cols))),
            shape=(Nx, Nx),
            dtype=int,
        ).tocsr()

        return P, dof_map

    @timeit("setup:system:DoFHandler:")
    def _get_sv_to_th_operator(self, V_dg, V_cg, Pp_dg_sort, nv):
        # Extract trial/test functions from VectorFunction Spaces
        p_dg = TrialFunction(V_dg)
        q_dg = TestFunction(V_dg)

        p_cg = TrialFunction(V_cg)
        q_cg = TestFunction(V_cg)

        # Need to construct V_cg field DoF sorting operator
        # it has not been constructed anywhere prior to this.
        mesh = V_cg.mesh()
        order = V_cg.ufl_element().degree()
        elem_type = V_cg.ufl_element().family()
        Vspace = VectorFunctionSpace(mesh, elem_type, order)
        cg_coord = interpolate(SpatialCoordinate(mesh), Vspace).dat.data_ro.copy()
        # TODO: Can I just pass lo_fe_sys sorting operator?
        Pp_cg_sort, _ = self._get_field_sorting_operator(cg_coord)

        # Assemble Operators and Extract CSR matrices
        def _extract_op(A):
            tmp = A.petscmat
            return sp.csr_matrix((tmp.getValuesCSR())[::-1])

        a = inner(p_dg, q_cg) * dx
        T = assemble(a, bcs=[], mat_type="aij")
        T_dg_to_cg = _extract_op(T)
        T_dg_to_cg = Pp_cg_sort * T_dg_to_cg * Pp_dg_sort.T

        a = inner(p_cg, q_cg) * dx
        M = assemble(a, bcs=[], mat_type="aij")
        M_cg = _extract_op(M)
        M_cg = (Pp_cg_sort * M_cg * Pp_cg_sort.T).tocsr()

        if self.keep:
            self.T_dg_to_cg = T_dg_to_cg
            self.M_cg = M_cg

        # Assemble a solver for Mass Matrix solve
        ml_solve = None
        exact_solve = None
        try:
            from pyamg import smoothed_aggregation_solver

            smoother = ("jacobi", {"omega": 3.0 / 2.0})  # omega < 1 doesn't work well
            ml_solve = smoothed_aggregation_solver(
                M_cg,
                presmoother=smoother,
                postsmoother=smoother,
            )
        except:
            # if M_cg is too small for SA-AMG
            exact_solve = sp.linalg.splu(M_cg.tocsc())

        # Put everything into a linear operator for seamless use
        from scipy.sparse.linalg import LinearOperator

        @timeit("solution:solver:dg_to_cg:")
        def mv(up_dg):
            resid = []
            up_cg = np.zeros((T_dg_to_cg.shape[0] + nv,))
            up_cg[:nv] = up_dg[:nv]
            if ml_solve is not None:
                up_cg[nv:] = ml_solve.solve(
                    T_dg_to_cg * up_dg[nv:],
                    accel="fgmres",
                    maxiter=100,
                    tol=1e-12,
                    residuals=resid,
                )
            else:
                tmp = T_dg_to_cg * up_dg[nv:]
                up_cg[nv:] = exact_solve.solve(tmp)
                resid = [np.linalg.norm(tmp - M_cg * up_cg[nv:])]

            assert resid[-1] < 1e-10, (
                """Grid-transfer solution is not as
                                        accurate as you think it is
                                        (resid=%2.2e)."""
                % resid[-1]
            )

            return up_cg

        # need to include velocity dofs too
        shape = tuple(np.array(T_dg_to_cg.shape) + nv)
        return LinearOperator(shape, matvec=mv)

    """ functions as an identity operator
    def _get_cg_to_dg_operator2(self, V_dg, V_cg, Pp_dg_sort, nv):
        # Extract trial/test functions from VectorFunction Spaces
        p_dg = TrialFunction(V_dg)
        q_dg = TestFunction(V_dg)

        p_cg = TrialFunction(V_cg)
        q_cg = TestFunction(V_cg)

        # Need to construct V_cg field DoF sorting operator
        # it has not been constructed anywhere prior to this.
        mesh        = V_cg.mesh()
        order       = V_cg.ufl_element().degree()
        elem_type   = V_cg.ufl_element().family()
        Vspace      = VectorFunctionSpace(mesh, elem_type, order)
        cg_coord    = interpolate(SpatialCoordinate(mesh), Vspace).dat.data_ro.copy()
        Pp_cg_sort, _ = self._get_field_sorting_operator(cg_coord)

        # Assemble Operators and Extract CSR matrices
        def _extract_op(A):
            tmp  = A.petscmat
            return sp.csr_matrix(( tmp.getValuesCSR())[::-1])

        a          = inner(p_cg, q_dg)*dx
        T          = assemble(a, bcs=[],  mat_type='aij')

        T_cg_to_dg = _extract_op(T)
        T_cg_to_dg = Pp_dg_sort*T_cg_to_dg*Pp_cg_sort.T

        a          = inner(p_dg, q_dg)*dx
        M          = assemble(a, bcs=[],  mat_type='aij')
        M_dg       = _extract_op(M)
        M_dg       = (Pp_dg_sort*M_dg*Pp_dg_sort.T).tocsr()

        # Assemble a solver for Mass Matrix solve
        ml_solve = None; exact_solve = None
        try:
            from pyamg import smoothed_aggregation_solver
            smoother = ('jacobi' , {'omega' : 3./2.}) # omega < 1 doesn't work well
            ml_solve = smoothed_aggregation_solver(M_dg, presmoother=smoother,
                                                    postsmoother=smoother,
                                            )
        except:
            # if M_cg is too small for SA-AMG
            exact_solve = sp.linalg.splu(M_dg.tocsc())

        # Put everything into a linear operator for seamless use
        from scipy.sparse.linalg import LinearOperator
        def mv0(up_cg):
            resid=[]
            up_dg       = np.zeros((T_cg_to_dg.shape[0]+nv,))
            up_dg[:nv]  = up_cg[:nv]
            if ml_solve is not None:
                up_dg[nv:] = ml_solve.solve(T_cg_to_dg*up_cg[nv:],
                                        accel='fgmres',
                                        maxiter=M_dg.shape[0],
                                        tol=1e-18,
                                        residuals=resid)
            else:
                tmp        = T_cg_to_dg*up_cg[nv:]
                up_dg[nv:] = exact_solve.solve(tmp)
                resid      = [np.linalg.norm(tmp-M_dg*up_dg[nv:])]

            assert resid[-1] < 1e-10, '''Grid-transfer solution is not as
                                        accurate as you think it is
                                        (resid=%2.2e).''' % resid[-1]

            return up_dg

        # need to include velocity dofs too
        shape = tuple(np.array(T_cg_to_dg.shape)+nv)
        return LinearOperator(shape, matvec=mv0)
    """

    @timeit("setup:system:DoFHandler:")
    def _get_cg_to_dg_operator(self, cg, dg):
        """Construct Mapping from CG to DG DoFs.

        TODO: Replace with firedrake's function calls.
        """
        n_dg, n_cg = dg.shape[0], cg.shape[0]
        rows = np.zeros((n_dg,))
        cols = np.zeros((n_dg,))
        dg_start = 0
        for i, cg_dof in enumerate(cg):
            for j, dg_dof in enumerate(dg[dg_start:, :]):
                if np.array_equal(cg_dof, dg_dof):
                    rows[dg_start + j] = i
                    cols[dg_start + j] = dg_start + j
                else:
                    dg_start += j
                    break

        P_1to1 = sp.csr_matrix((np.ones(n_dg), (cols, rows)), shape=(n_dg, n_cg))
        assert np.allclose(P_1to1 * cg, dg), "CG->DG mapping is broken."

        return P_1to1

    def _renumber_dofs(self):
        """Renumbers the system's DoFs.

        Will need to be implemented by each extending class individually.
        Logic for sorting Stokes DoFs is a lot more complicated than
        a simple scalar field like in Poisson.
        """
        pass

    @timeit("setup:system:DoFHandler:")
    def _grid_hierarchy(self):
        """Construct 2D DoF array.

        Construct 2D arrays summarizing topological location of DoFs.

        Construct 2D arrays containing DoF number for each field (velocity,
                pressure). These grids only make sense for structured grids.
        These grid are later used to construct interpolation operators and
        Vanka relaxation patches.

        This method currently only works for structured grids in 2D.
        """
        assert self.dim == 2, "Only 2D problems are supported at this time."

        NEx_hier = self.NE_hier[0]
        NEy_hier = self.NE_hier[1]
        """
        if self.diagonal == 'crossed':
            for i in range(len(NEx_hier)):
                NEx_hier[i] = NEx_hier[i]*2
            for i in range(len(NEy_hier)):
                NEy_hier[i] = NEy_hier[i]*2
        """

        self.grid_hier = []
        self.ext_grid_hier = []
        for dof_coord, Z, NEx, NEy in zip(
            self.dof_coord, [p.Z for p in self.problems], NEx_hier, NEy_hier
        ):
            grids = {}
            ext_grids = {}  # try eliminating this variable in the future
            for name, dof_coord in dof_coord.items():
                bc_type = "periodic" if self.periodic else "dirichlet"

                # figure out the order of space
                for Vt in Z.subfunctions:
                    if Vt.name == name:
                        break
                Velem = Vt.ufl_element()
                dim = Velem.cell().geometric_dimension()
                order = dim if isinstance(Velem, VectorElement) else 1
                # dirty hack to fix diffusion
                order = order if isinstance(self.elem_order, tuple) else self.elem_order
                assert order in [1, 2], (
                    "Grids for element of order=%d are not supported yet." % order
                )
                ###############################################################
                # Diffusion/velocity DoFs map
                ###############################################################
                if order == 2 and not self.lo_fe_precond:
                    DOFs_per_side_x = NEx * 2 + 1
                    DOFs_per_side_y = NEy * 2 + 1
                else:
                    DOFs_per_side_x = NEx + 1
                    DOFs_per_side_y = NEy + 1

                if self.periodic:
                    DOFs_per_side_x -= 1
                    DOFs_per_side_y -= 1

                dofs = dof_coord.T
                loc_to_dof_map = np.zeros_like(dofs, dtype=int)

                min_val = np.min(dofs, axis=1)
                dof_shift = (dofs.T - min_val).round(12).T
                dx = min((dof_shift[0][np.nonzero(dof_shift[0])]))
                dy = min((dof_shift[1])[np.nonzero(dof_shift[1])])
                # compute index of each element in each direction
                loc_to_dof_map[1] = np.round((dof_shift[1]) / dy).astype(int)
                loc_to_dof_map[0] = np.round((dof_shift[0]) / dx).astype(int)

                grid_shape = (DOFs_per_side_x, DOFs_per_side_y)
                fine_grid = np.zeros(grid_shape, dtype=int) - 1

                # store the DOF at each respective node
                for p in range(len(loc_to_dof_map[0])):
                    fine_grid[loc_to_dof_map[0][p], loc_to_dof_map[1][p]] = p

                offset = 3
                grids[name] = fine_grid
                ext_grid = self._extend_dof_grid(
                    fine_grid, bc_type=bc_type, padding=offset
                )
                ext_grids[name] = ext_grid
            self.grid_hier.append(grids)
            self.ext_grid_hier.append(ext_grids)

    def _extend_dof_grid(self, grid, bc_type=None, padding=0):
        """pad the grid with zeros around the perimeter"""
        assert self.dim == 2, "Only 2D problems are supported at this time."
        ext_grid = None
        if padding > 0:
            pad = padding
            if bc_type == "periodic":
                ext_grid = (
                    np.ones(
                        (grid.shape[0] + pad * 2, grid.shape[1] + pad * 2), dtype=int
                    )
                    * -1
                )
                ext_grid[pad : -1 * pad, pad : -1 * pad] = grid

                lx = grid.shape[1]
                ly = grid.shape[0]
                # left and right border
                ext_grid[pad : -1 * pad, :pad] = grid[:, (lx - pad) : (lx - 0)]
                ext_grid[pad : -1 * pad, (lx + pad) :] = grid[:, :pad]
                # top and bottom
                ext_grid[:pad, pad : -1 * pad] = grid[(ly - pad) : (ly - 0), :]
                ext_grid[(ly + pad) :, pad : -1 * pad] = grid[:pad, :]
                # corners
                ext_grid[:pad, :pad] = grid[
                    (ly - pad) : (ly - 0), (lx - pad) : (lx - 0)
                ]
                ext_grid[(ly + pad) :, (lx + pad) :] = grid[:pad, :pad]
                ext_grid[:pad, (lx + pad) : (lx + pad * 2)] = grid[
                    (ly - pad) : (ly - 0), :pad
                ]
                ext_grid[(lx + pad) : (lx + pad * 2), :pad] = grid[
                    :pad, (ly - pad) : (ly - 0)
                ]
            else:
                ext_grid = (
                    np.ones(
                        (grid.shape[0] + pad * 2, grid.shape[1] + pad * 2), dtype=int
                    )
                    * -1
                )
                ext_grid[pad : -1 * pad, pad : -1 * pad] = grid

        return ext_grid
