from collections.abc import Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy
import scipy.sparse as sp

from sysmg.util.decorators import timeit
from .relaxation import System_Relaxation


class Vanka(System_Relaxation):
    """
    Vanka relaxation for 2x2 block matrices.

    Vanka relaxation designed for Q2/Q1 and Q1isoQ2/Q1 Stokes Discretizations.

    """

    omega = None
    relax_iters = None
    vanka_type = None
    update_type = None
    # addtive optimizations
    _GS = None
    _r_gs = None
    _Aloc = None

    @timeit("setup:solver:vanka:")
    def __init__(self, stokes,
                 params={'iterations': (2, 2), 'type': 'geometric',
                         'update': 'additive', 'omega': 1,
                         'patch_solver': 'pinv', 'debug': True},
                 level=0):
        """
        Vanka initialization function.

        Args:
            stokes (Stokes): stokes object from sysmg.systems.stokes.

            params (dict): dictionary object specifying all the parameters
                needed for BS relaxation.

                Options:
                'vcycle'          : tuple
                    (# of pre-, #  of post-) relaxation sweeps.
                'type'            : {'geometric', 'algebraic'}
                    Vanka relaxation type.
                'update': {'addtive', 'addtive-opt', 'multiplicative'}
                    Update type.
                'omega'         : float
                    global update damping.

                Example:
                    {'type'  : 'geometric',  'update' : 'additive',
                     'vcycle': (nu_1, nu_2), 'omega': omega}

        Returns:
            Nothing

        """
        super().__init__(params, level)

        # timer setup
        self.omega                = params["omega"]
        self.update_type          = params.get("update", 'additive')
        self.vanka_type           = params["type"]
        self.vanka_opt            = params.get("opt", False)
        self.vanka_setup_opt      = params.get("setup_opt", False)
        assert type(self.vanka_opt) == bool, 'vanka_opt must be a boolean'
        self.patch_solver         = params.get('patch_solver', 'inv')
        self.vanka_ein_mem_factor = params.get("memory_factor", 1.2)
        self.vanka_ein_hi_cutoff  = params.get("hi_cutoff", 0.5)
        self.pdof_order           = params.get("patch_order", "natural")
        assert self.pdof_order in ["natural", "sorted"], 'patch_order must be "natural" or "sorted"'
        self.keep                 = params["keep"] if "keep" in params.keys() else False
        self.debug                = params['debug'] if 'debug' in params else False
        self.cblas               = params.get('cblas', False)
        if 'blas' in params.keys():
            raise Exception('blas is no longer supported. Use cblas instead.')

        if self.empty_shell:
            # empty shell for when relaxation is skipped on a level
            return

        # Operator Setup
        self.system = stokes
        B = stokes.A_bmat[1, 0]
        B.eliminate_zeros()
        self.dim = stokes.dim
        self.B = B if type(B) == sp.csr_matrix else B.tocsr()
        self.M = stokes.A_bmat[0, 0]
        self.nullspace = getattr(stokes, 'nullspace', None)
        # needed for certain optimizations
        if not self.B.has_sorted_indices:
            self.B.sort_indices()
        self._vdofs = stokes.velocity_nodes()
        self._pdofs = stokes.pressure_nodes()
        self._udofs_component_wise = getattr(stokes, '_udofs_component_wise',
                                             [self.M.shape[0]//self.dim] * self.dim)

        if self.keep:
            self._Aloc = []
            self._V_neigh_gIDs = []
        #########################################################
        # for each pressure DoF (in Taylor-Hood disc) or element
        # (in Scott-Vogelius), find algebraically connected velocity
        # DoFs.
        # if Vanka has already been setup use it instead of setting
        # it up anew.
        ##########################################################\
        # Patch setup and data collection
        self._compute_block_sizes()
        self._setup_patches(self.vanka_type)
        ##########################################################
        # initialize self.relax to the correct version
        self._setup_relaxation_method()
        ##########################################################
        # Patch size stats
        self._compute_patch_stats()
        ##########################################################
        if self.debug:
            print('vanka-init-end-------------------------------------------')
            dim = stokes.dim
            pdofs = stokes.dof_coord[-1]['p']
            vdofs = stokes.dof_coord[-1]['u']
            dofs = np.vstack((vdofs, vdofs, pdofs)) if dim == 2 else np.vstack((vdofs, vdofs, vdofs, pdofs))
            GS = self._GS

            dofs_remapped = GS.T * np.einsum('ij,i->ij', GS * dofs, self._partition_of_unity)
            non_zero_dofs = dofs[np.where(np.abs(dofs_remapped - dofs) > 1e-12)[0]]
            hasZeroCoord = np.min(non_zero_dofs, axis=-1) == 0.0
            hasOneCoord = np.max(non_zero_dofs, axis=-1) == 1.0
            assert np.logical_or(hasZeroCoord, hasOneCoord).all(), \
                'GS operator does not work..'
        else:
            # not needed anymore
            delattr(self, 'B')
            delattr(self, 'M')
            delattr(self, 'system')

    @timeit("setup:solver:vanka:")
    def _compute_block_sizes(self):
        """ Compute the block sizes for the algebraic Vanka relaxation.
        """
        if self.vanka_type in ['algebraic_n_part', 'algebraic_factorized']:
            self.block_sizes = (self.B.indptr[1:]-self.B.indptr[:-1]+1).astype(np.int64)
        elif self.vanka_type in ['geometric_dg']:
            P_cells = self.system.dof_cells[-1]['p']
            pdofs_dim = P_cells.shape[1]
            self.block_sizes = np.zeros((P_cells.shape[0],), dtype=np.int64)
            self.v_ids_tmp = np.zeros((self.B.shape[1],))
            for i, pcell in enumerate(P_cells):
                l_slice = slice(0, 0)
                for indptr0, indptr1 in [self.B.indptr[pid:pid + 2] for pid in pcell]:
                    l_slice = slice(l_slice.stop, l_slice.stop + indptr1 - indptr0)

                    self.v_ids_tmp[l_slice] = self.B.indices[indptr0:indptr1]

                v_ids = np.unique(self.v_ids_tmp[:l_slice.stop])
                self.block_sizes[i] = v_ids.size + pdofs_dim
        else:
            raise NotImplementedError(f'Vanka type {self.vanka_type} _compute_block_sizes not implemented.')

    @timeit("setup:solver:vanka:")
    def _setup_patches(self, vanka_type):
        setup_fxn = getattr(self, '_patch_setup_' + vanka_type)
        setup_fxn()

    @timeit("setup:solver:vanka:")
    def _patch_solver(self, Aloc, wrap=True):
        """ Assumes that Aloc is already in sparse matrix format.
        """
        if self.patch_solver in ['pinv', 'inv']:
            if self.patch_solver == 'pinv':
                invert = scipy.linalg.pinv
            elif self.patch_solver == 'inv':
                invert = np.linalg.inv
            Aloc_inv = invert(Aloc.toarray() if sp.isspmatrix(Aloc) else Aloc)
        else:
            Aloc_inv = sp.linalg.splu(Aloc, permc_spec='NATURAL')

        if wrap and self.patch_solver == 'splu':
            return lambda r: Aloc_inv.solve(r)
        elif wrap:
            return lambda r: Aloc_inv @ r
        else:
            return Aloc_inv

    @timeit("setup:solver:vanka:")
    def _get_patch(self, p_idx):
        """ Returns the patch corresponding to the pressure DoF(s) p_idx.
        """

        if not isinstance(p_idx, Iterable):
            v_ids = self.B.indices[self.B.indptr[p_idx]:self.B.indptr[p_idx + 1]]
            p_idx = np.array([p_idx])
            v_ids = self.B.indices[self.B.indptr[p_idx[0]]:self.B.indptr[p_idx[-1] + 1]]
        else:
            #get non-zero divergence values (Bdata)
            #and the associated column indecies (v_ids)
            l_slice = slice(0, 0)
            for indptr0, indptr1 in [self.B.indptr[pid:pid+2] for pid in p_idx]:
                l_slice = slice(l_slice.stop, l_slice.stop + indptr1 - indptr0)
                self.v_ids_tmp[l_slice] = self.B.indices[indptr0:indptr1]
                break

            v_ids = np.unique(self.v_ids_tmp[:l_slice.stop])

        Mloc = self.M[v_ids, :]
        Mloc = Mloc[:, v_ids]

        if len(p_idx) == 1:
            Bloc = self.B.getrow(p_idx)
            Bloc = Bloc.data
        else:
            Bloc = self.B[p_idx, :]
            Bloc = Bloc[:, v_ids]

        if len(p_idx) == 1:
            # np.block approach is slower
            _Aloc = np.zeros((Mloc.shape[0]+1, Mloc.shape[0]+1))
            _Aloc[:Mloc.shape[0], :Mloc.shape[0]] = Mloc.toarray()
            _Aloc[:Mloc.shape[0], -1] = Bloc
            _Aloc[-1, :Mloc.shape[0]] = Bloc
        else:
            Bloc = Bloc.toarray()
            _Aloc = np.zeros((Mloc.shape[0]+3, Mloc.shape[0]+3))
            _Aloc[:Mloc.shape[0], :Mloc.shape[0]] = Mloc.toarray()
            _Aloc[:Mloc.shape[0], Mloc.shape[0]:] = Bloc.T
            _Aloc[Mloc.shape[0]:, :Mloc.shape[0]] = Bloc

        if self.patch_solver == 'splu':
            _Aloc = sp.csc_matrix(_Aloc)
            _Aloc.eliminate_zeros()

        if self.keep:
            self._Aloc.append(_Aloc)
            self._V_neigh_gIDs.append(v_ids)

        return _Aloc

    def _compute_patch_stats(self):
        """ Patch statistics. """
        self.patch_stats = {'mean': np.mean(self.block_sizes),
                            'std': np.std(self.block_sizes),
                            'min': np.min(self.block_sizes),
                            'max': np.max(self.block_sizes),
                            'sum': np.sum(self.block_sizes),
                            'sq_sum': np.sum(self.block_sizes ** 2),
                            'num': len(self.block_sizes)}

    @timeit("setup:solver:vanka:")
    def _assemble_scatter(self, pressure_dofs_order=None):
        """ Create a gather and scatter operator for the patch
        dof mapping. """
        B        = self.B  # divergence operator (csr)
        B_ptr   = B.indptr  # csr row ptr
        row_idx = 0

        if self.vanka_type in ['algebraic', 'algebraic_n_part']:
            pdofs_order = pressure_dofs_order if pressure_dofs_order is not None \
                else np.linspace(0, self._pdofs - 1, self._pdofs, dtype=int)

            rows = np.linspace(0, B.nnz+B.shape[0]-1,B.nnz+B.shape[0], dtype=int)
            cols = np.zeros_like(rows)

            for p_idx in pdofs_order:
                v_ids = B.indices[B_ptr[p_idx]:B_ptr[p_idx + 1]]
                cols[row_idx:row_idx + len(v_ids)] = v_ids
                cols[row_idx + len(v_ids)] = self._vdofs + p_idx
                row_idx += v_ids.shape[0] + 1

            data = np.ones(len(rows))
            nrows = np.max(rows) + 1
            ncols = np.max(cols) + 1
        elif self.vanka_type in ['geometric_dg']:

            P_cells = self.system.dof_cells[-1]['p']
            nv_g = self._vdofs
            pdofs_order = pressure_dofs_order if pressure_dofs_order is not None \
                else np.linspace(0, P_cells.shape[0] - 1, P_cells.shape[0], dtype=int)

            patch_nnz = np.sum(self.block_sizes)
            rows = np.linspace(0, patch_nnz - 1, patch_nnz, dtype=int)
            cols = np.zeros_like(rows)

            npi = P_cells.shape[1]
            for p_idx in pdofs_order:
                pcell = P_cells[p_idx]
                v_ids = []
                for pid in pcell:
                    v_ids += list(B.indices[B_ptr[pid]:B_ptr[pid + 1]])

                v_ids = np.unique(v_ids)
                nv = v_ids.shape[0]

                cols[row_idx:row_idx + nv] = v_ids
                cols[row_idx + nv:row_idx + nv + npi] = nv_g + pcell
                row_idx += nv + npi

            data = np.ones(len(rows))
            nrows = np.max(rows) + 1
            ncols = np.max(cols) + 1

        else:
            raise NotImplementedError

        _GS = sp.csr_matrix((data, (rows, cols)),
                            shape=(nrows, ncols),
                            dtype=bool)
        self._GS = _GS

    @timeit("setup:solver:vanka:")
    def _assemble_partition_of_unity(self):
        """ Create a partition of unity for vanka patches.
            Requires the _GS operator to be already setup.
        """
        #############################################################
        # a little hack for computing the partition of unity
        if self._GS is not None:
            tmp = self._GS * (self._GS.T * np.ones((self._GS.shape[0],)))
            idx = np.where(tmp > 0)[0]
            tmp[idx] = 1.0 / tmp[idx]
            self._partition_of_unity = tmp
            return
        else:
            raise NotImplementedError("First setup _GS operator before calling _assemble_partition_of_unity")

    def _patch_setup_algebraic_n_part(self):
        """ Setup the algebraic Vanka relaxation.
            this approach iterates over each block in the solution phase.
            Also, referred to as n-part Vanka.
        """

        if self.vanka_opt:
            assert self.pdof_order == 'sorted',\
            '_patch_setup_geometric_dg: optimal option requires sorted pdofs'

        if self.pdof_order == 'natural':
            Ps_ordered = np.linspace(0, self._pdofs - 1, self._pdofs, dtype=int)
        elif self.pdof_order == 'sorted':
            Ps_ordered = np.argsort(self.block_sizes)
            self.block_sizes = self.block_sizes[Ps_ordered]
        else:
            raise NotImplementedError("patch_order = {} not implemented".format(self.patch_order))

        _A_size = np.sum(self.block_sizes ** 2)
        self._A_inv_flat = np.zeros((_A_size,), dtype=np.double)

        if self.vanka_setup_opt:
            from .core.patch_mult import th_patch_setup
            th_patch_setup(self.M.indptr, self.M.indices, self.M.data,
                               self.B.indptr, self.B.indices, self.B.data,
                               Ps_ordered,
                               self._A_inv_flat,
                               np.max(self.block_sizes)
                               )
        else:
            q_slice  = slice(0, 0)
            for p_idx in Ps_ordered:
                """get local patch matrices and their inverses"""
                _Aloc   = self._get_patch(p_idx)
                q_slice = slice(q_slice.stop, q_slice.stop + _Aloc.size)
                self._A_inv_flat[q_slice] = self._patch_solver(_Aloc, wrap=False).ravel()

        self._patch_dim_cumsum = np.cumsum(self.block_sizes)
        self._assemble_scatter(pressure_dofs_order=Ps_ordered)
        self._assemble_partition_of_unity()

        if self.vanka_opt:
            # needed for batching
            self.patch_size_and_counts = np.vstack(np.unique(self.block_sizes, return_counts=True)).T

    def _patch_setup_algebraic_factorized(self):
        B = self.B
        M = self.M
        dim = self.dim
        n_u = B.shape[1]
        n_p = B.shape[0]
        # On coarser AMG levels the components of the vector Laplacian are not always
        # of the same size. This is due to the fact that the Laplacian is constructed
        # one component at a time.
        n_ux = np.cumsum(self._udofs_component_wise)
        assert B.shape[1] == n_ux[-1], 'Div operator or dim is wrong.'

        Mloc_x_sizes = np.zeros((n_p, dim), dtype=int)  # Laplacian Block sizes

        for p in range(n_p):
            u_ids = B.indices[B.indptr[p]:B.indptr[p + 1]]
            idx_comp_wise = [np.where(u_ids < n_ux[0])[0],
                             np.where((u_ids >= n_ux[0]) & (u_ids < n_ux[1]))[0]
                             ]
            if dim == 3:
                idx_comp_wise.append(np.where(u_ids >= n_ux[1])[0])

            tmp = np.array([len(idx_comp_wise[i]) for i in range(dim)])
            Mloc_x_sizes[p] = tmp

        # storage required
        M_inv_nnz_orig = np.sum((np.ravel(Mloc_x_sizes) ** 2))
        B_nnz_orig = np.sum(np.ravel(Mloc_x_sizes))

        # allocate
        M_inv = np.zeros((M_inv_nnz_orig,))
        #Bhat = np.zeros((B_nnz_orig,))
        Uhat = np.zeros((B_nnz_orig,))
        S_inv = np.zeros((n_p))

        if self.vanka_setup_opt:
            from .core.patch_mult import th_bf_patch_setup
            th_bf_patch_setup(self.M.indptr, self.M.indices, self.M.data,
                              self.B.indptr, self.B.indices, self.B.data,
                              Mloc_x_sizes.ravel(),
                              M_inv, Uhat, S_inv,
                              self.dim,
                              np.max(self.block_sizes)
                              )
        else:
            M_inv_nnz_offset_no_padding = 0  # needed to keep track of where we are in M_inv
            B_nnz_offset = 0

            # iterate through each pressure DoF
            for p in range(n_p):
                Us_idx = slice(B.indptr[p], B.indptr[p + 1])
                Us_i = B.indices[Us_idx]  # Velocities in a patch

                # Extract the local patch
                M_i = M[Us_i, :]
                M_i = M_i[:, Us_i]
                M_i = M_i.toarray()
                B_i = B.data[Us_idx] #B.getrow(p).data

                Bloc_offset = 0
                for Ux_size in Mloc_x_sizes[p]:
                    M_inv_i = M_inv[M_inv_nnz_offset_no_padding:  # Patch vector Laplacian
                                               M_inv_nnz_offset_no_padding + Ux_size ** 2]  # will go here
                    M_inv_nnz_offset_no_padding += Ux_size ** 2

                    Uhat_i = Uhat[B_nnz_offset:B_nnz_offset + Ux_size]

                    # M^{-1}_x
                    M_x_inv = np.linalg.inv(M_i[Bloc_offset:Bloc_offset + Ux_size,
                                                Bloc_offset:Bloc_offset + Ux_size])
                    M_inv_i[:] = M_x_inv.ravel()
                    # \hat{U} = - M^{-1}_x B_x
                    Bx = B_i[Bloc_offset:Bloc_offset + Ux_size]
                    Uhat_i[:] = -1.0 * M_x_inv @ Bx

                    S_inv[p] += np.dot(Bx, Uhat_i[:])

                    B_nnz_offset += Ux_size
                    Bloc_offset += Ux_size

            # Invert Schur Complement
            S_inv = 1. / S_inv


        # Update Bhat values
        # B_nnz_offset = 0
        # for p in range(n_p):
        #     U_size = (Mloc_x_sizes[p]).sum()
        #     Bhat_i = Bhat[B_nnz_offset:B_nnz_offset + U_size]
        #     Uhat_i = Uhat[B_nnz_offset:B_nnz_offset + U_size]
        #     # \hat{B} = -S^{-1} BA^{-1} = S_inv^{-1} \hat{U} = S^{-1} (- M^{-1} B)
        #     Bhat_i[:] = S_inv[p] * Uhat_i
        #
        #     B_nnz_offset += U_size

        ############################################################
        # Gather/Scatter Operator
        rows = np.zeros((B.nnz,), dtype=int)
        cols = np.zeros((B.nnz,), dtype=int)
        offset = 0
        for p in range(n_p):  # Traversal order is important!!!
            u_ids = B.indices[B.indptr[p]:B.indptr[p + 1]]
            nv = u_ids.shape[0]
            rows[offset:offset + nv] = np.linspace(offset, offset + nv - 1, nv, dtype=int)
            cols[offset:offset + nv] = u_ids

            offset += nv

        # idx_type = np.int32 if csr_max_idx < np.iinfo(np.int32).max else np.int64
        self._GS = sp.csr_matrix((np.ones(rows.size), (rows, cols)),
                            shape=(np.sum(Mloc_x_sizes), n_u),
                            dtype=bool)

        ############################################################
        # self._assemble_partition_of_unity() #Us_count,
        #                              #total_us_in_patches,
        #                              #pdofs_per_patch=0,
        #                              #pressure_dofs_order=Ps_ordered)

        tmp = self._GS.T * (self._GS * np.ones((self._GS.shape[1],)))
        idx = np.where(tmp > 0)[0]
        tmp[idx] = 1.0 / tmp[idx]
        self._partition_of_unity = tmp

        ############################################################
        # Assemble other related info

        # Store for relaxation kernel
        self._Mloc_x_sizes = Mloc_x_sizes
        self._M_inv = M_inv
        #self._Bhat = Bhat
        self._Uhat = Uhat
        self._S_inv = S_inv

    def _patch_setup_geometric_dg(self):
        """cell-wise DG Vanka.

        """
        assert hasattr(self.system, 'dof_cells'), \
            'Vanka(geometric_dg) needs Cell Dof Map.'

        if self.vanka_opt:
            assert self.pdof_order == 'sorted',\
            '_patch_setup_geometric_dg: optimal option requires sorted pdofs'

        if self.pdof_order == 'natural':
            block_sizes = self.block_sizes
            P_cells = self.system.dof_cells[-1]['p']
            Ps_ordered = np.linspace(0, P_cells.shape[0] - 1, P_cells.shape[0], dtype=int)
        elif self.pdof_order == 'sorted':
            Ps_ordered = np.argsort(self.block_sizes)
            self.block_sizes = self.block_sizes[Ps_ordered]
            block_sizes = self.block_sizes
            P_cells = self.system.dof_cells[-1]['p'][Ps_ordered,:]
        else:
            raise NotImplementedError("patch_order = {} not implemented".format(self.patch_order))

        # Compute patches and place them into a linear array
        self._A_inv_flat = np.zeros((np.sum(self.block_sizes ** 2),), dtype=np.double)
        if self.vanka_setup_opt:
            from .core.patch_mult import sv_patch_setup
            sv_patch_setup(self.M.indptr, self.M.indices, self.M.data,
                           self.B.indptr, self.B.indices, self.B.data,
                           P_cells.reshape(-1),
                           P_cells.shape[0],
                           self._A_inv_flat,
                           np.max(self.block_sizes)
                           )
        else:
            l_idx, q_idx = 0, 0
            for pcell, bs in zip(P_cells, block_sizes):
                Aloc = self._get_patch(pcell)
                assert Aloc.shape == (bs, bs), f'Aloc.shape = {Aloc.shape}, bs = {bs}'
                self._A_inv_flat[q_idx:q_idx + Aloc.size] = self._patch_solver(Aloc, wrap=False).reshape(-1)
                l_idx += bs
                q_idx += Aloc.size


        del self.v_ids_tmp
        # Assemble the partition of unity and the gather scatter matrix
        self._patch_dim_cumsum = np.cumsum(self.block_sizes)
        self._assemble_scatter(pressure_dofs_order=Ps_ordered)
        self._assemble_partition_of_unity()

        if self.vanka_opt:
            # needed for batching
            self.patch_size_and_counts = np.vstack(np.unique(self.block_sizes, return_counts=True)).T

    def _relax_multiplicative(self, A, x, b):
        # TODO: update multiplicative Vanka
        raise ValueError("NEEDS TO BE UPDATED")

    def _setup_relaxation_method(self):
        """
        Setup the relaxation method for the Vanka smoother.
        Different Vanka patching methods and solvers use different solution methods.
        """

        if self.update_type == 'additive':
            self._set_additive_relaxation_method()
        elif self.update_type == 'multiplicative':
            def mv(A, x, b):
                return self._relax_multiplicative(A, x, b)

            self.relax = mv
        else:
            raise ValueError("Unknown relaxation type: {}".format(self.update_type))

        # If Krylov-wrapping is used pre-allocate temporary storage array.
        if self.accel:
            self._xt = np.zeros((self.system.ndofs(),))
        # run it once to trigger creation of self.eigs
        system = self.system
        np.random.seed(7)
        if self.accel:
            self.presmoother(system.A_bmat.tocsr(),
                             np.zeros(system.ndofs()),
                             np.random.rand(system.ndofs()))
            if not hasattr(self, 'eigs'):
                self.postsmoother(system.A_bmat.tocsr(),
                                  np.zeros(system.ndofs()),
                                  np.random.rand(system.ndofs()))

    def _set_additive_relaxation_method(self):
        """
        Setup additive relaxation method for the Vanka smoother.
        """

        if self.vanka_opt and self.vanka_type in ["algebraic_n_part", 'geometric_dg']:
            self._r_gs = np.zeros((self._GS.shape[0],), dtype=np.float64)

            # precompute indices
            self._l_slice = np.zeros((self.patch_size_and_counts.shape[0] + 1), dtype=np.int64)
            self._q_slice = np.zeros((self.patch_size_and_counts.shape[0] + 1), dtype=np.int64)
            for i, (patch_size, count) in enumerate(self.patch_size_and_counts):
                tmp = count * patch_size
                self._l_slice[i+1] = self._l_slice[i] + tmp
                self._q_slice[i+1] = self._q_slice[i] + tmp * patch_size

            self.patch_size = self.patch_size_and_counts[:, 0]
            self.patch_count = self.patch_size_and_counts[:, 1]
            assert self.block_sizes.shape[0] == self.patch_count.sum(), 'blocks missing'
            assert self._A_inv_flat.shape[0] == self._q_slice[-1], \
                "A_inv_flat size does not match the number of patches: {} != {}".format(self._A_inv_flat.shape[0], self._q_slice[-1])
            assert self._r_gs.shape[0] == self._l_slice[-1], \
                "r_gs size does not match the number of patches: {} != {}".format(self._r_gs.shape[0], self._l_slice[-1])

            if self.cblas:
                self._x_gs = np.zeros((self._GS.shape[0],), dtype=np.float64)
                from .core.patch_mult import n_part_batch
                def mv(A, x, b):
                    _GS           = self._GS
                    for _ in range(self.relax_iters):
                        self._r_gs[:] = _GS * (b - A * x)

                        n_part_batch(# offset
                                     self.patch_size,self.patch_count,
                                     self._l_slice, self._q_slice,
                                     # data
                                     self._r_gs, self._A_inv_flat, self._x_gs,
                                     # partion of unity and omega scaling
                                     self._partition_of_unity, self.omega,
                                     )

                        x += _GS.T * self._x_gs
                    return x
            else:
                def mv(A, x, b):
                    r_gs = self._r_gs  # reuse the same memory for temporary data
                    # optimized vanka has operators integrated into self._A_inv
                    _GS = self._GS
                    _v_scale_glob = self._partition_of_unity
                    omega = self.omega
                    l_slice = self._l_slice
                    q_slice = self._q_slice
                    for _ in range(self.relax_iters):
                        # recompute residual and split it
                        r = b - A * x
                        r_gs[:] = _GS * r

                        for i, (patch_size, count) in enumerate(self.patch_size_and_counts):
                            #print(f'i={i:5d}: {patch_size:5d} x {count:5d}')

                            # K^{-1} r
                            if count == 1:
                                r_batch    = r_gs[l_slice[i]:l_slice[i+1]]
                                Ainv_batch = self._A_inv_flat[q_slice[i]:q_slice[i+1]].reshape((patch_size, patch_size))
                                np.dot(Ainv_batch, r_batch, out=r_batch)
                            else:
                                r_batch = r_gs[l_slice[i]:l_slice[i+1]].reshape((-1, patch_size))
                                Ainv_batch = self._A_inv_flat[q_slice[i]:q_slice[i+1]].reshape((count, patch_size, patch_size))
                                np.einsum("pij,pj->pi", Ainv_batch, r_batch,
                                           #optimize=['einsum_path', (0, 1)],
                                           out=r_batch)

                        r_gs[:] = omega * (_v_scale_glob * r_gs)
                        x      += _GS.T * r_gs

                    return x

        elif self.vanka_type == 'algebraic_factorized':

            nu_gs = self._GS.shape[0]
            self._dx_scatter = np.zeros((nu_gs + self._pdofs,))
            self._rg_scatter = np.zeros((nu_gs + self._pdofs,))

            if self.cblas:
                from .core.patch_mult import n_part_bs
                def mv(A, x, b):
                    GS = self._GS

                    Mloc_x_sizes = self._Mloc_x_sizes
                    M_inv = self._M_inv
                    #Bhat = self._Bhat
                    Uhat = self._Uhat
                    S_inv = self._S_inv

                    partition_of_unity = self._partition_of_unity

                    nu_gs, nu = self._GS.shape

                    dx = self._dx_scatter
                    #dx *= 0
                    du = dx[:nu_gs]
                    dp = dx[nu_gs:]

                    rg = self._rg_scatter

                    for _ in range(self.relax_iters):
                        # compute residual
                        r = b - A * x
                        # scatter residual (and order it according to patch sizes)
                        rg[:nu_gs] = GS * r[:nu]
                        rg[nu_gs:] = r[nu:]

                        # optimization technique to minimize memory movement
                        du[:] = Uhat
                        dp[:] = S_inv

                        n_part_bs(Mloc_x_sizes.ravel(), M_inv, #Uhat, S_inv,
                                  rg, dx,
                                  self.dim, nu_gs)

                        x[:nu] += self.omega * (partition_of_unity * (GS.T * du))
                        x[nu:] += self.omega * dp

                    return x

            else:
                def mv(A, x, b):
                    GS = self._GS

                    Mloc_x_sizes = self._Mloc_x_sizes
                    M_inv = self._M_inv
                    #Bhat = self._Bhat
                    Uhat = self._Uhat
                    S_inv = self._S_inv

                    partition_of_unity = self._partition_of_unity

                    omega     = self.omega  # omega can be changed during the run
                    nu_gs, nu = self._GS.shape

                    dx = self._dx_scatter
                    dx *= 0
                    du = dx[:nu_gs]
                    dp = dx[nu_gs:]

                    rg = self._rg_scatter


                    for _ in range(self.relax_iters):
                        # compute residual
                        r = b - A * x
                        # scatter residual (and order it according to patch sizes)
                        rg[:nu_gs] = GS * r[:nu]
                        rg[nu_gs:] = r[nu:]
                        # split component-wise
                        ru = rg[:nu_gs]
                        rp = rg[nu_gs:]

                        U_offset = 0
                        M_offset = 0
                        for i, Mloc_sizes in enumerate(Mloc_x_sizes):
                            M_size   = Mloc_sizes.sum()
                            M_size2  = (Mloc_sizes**2).sum()
                            # A^{-1} r_u // Laplacian Correction
                            ru_patch = ru[U_offset:U_offset + M_size]
                            du_patch = du[U_offset:U_offset + M_size]
                            Minv_patch = M_inv[M_offset:M_offset + M_size2]
                            Uhat_patch = Uhat[U_offset:U_offset + M_size]
                            dp[i] = S_inv[i] * (np.dot(Uhat_patch, ru_patch) + rp[i])

                            Mx_offset = 0
                            Mx_offset2 = 0
                            for bs in Mloc_sizes:
                                Minv_x = Minv_patch[Mx_offset2:Mx_offset2 + bs**2].reshape((bs, bs))
                                np.dot(Minv_x,
                                       ru_patch[Mx_offset:Mx_offset + bs],
                                       out=du_patch[Mx_offset:Mx_offset + bs])

                                Mx_offset  += bs
                                Mx_offset2 += bs**2

                            # original approach requiring storage of Bhat
                            # Bhat_patch = Bhat[U_offset:U_offset + M_size]
                            # dp[i] = np.dot(Bhat_patch, ru_patch)

                            # S^{-1} r_{p} // Schur Complement correction
                            # dp[i]  = S_inv[i] * rp[i]
                            # dp[i] += S_inv[i] * np.dot(Uhat_patch, ru_patch)


                            # \hat{U} r_{p} // divergence correction
                            du_patch += (Uhat_patch * dp[i])

                            U_offset += M_size
                            M_offset += M_size2

                        x[:nu] += omega * (partition_of_unity * (GS.T * du))
                        x[nu:] += omega * dp

                    return x

        elif self.vanka_type in ["algebraic_n_part", "geometric_dg"]:
            self._r_gs           = np.zeros((self._GS.shape[0],), dtype=np.float64)
            self._A_inv_flat     = self._A_inv_flat.astype(np.float64)
            self._offset_bs      = np.zeros((len(self.block_sizes)+1,), dtype=np.int64)
            self._offset_bs[1:]  = np.cumsum(self.block_sizes)
            self._offset_nnz     = np.zeros((len(self.block_sizes)+1,), dtype=np.int64)
            self._offset_nnz[1:] = np.cumsum(self.block_sizes**2)

            if self.cblas:
                self._x_gs = np.zeros((self._GS.shape[0],), dtype=np.float64)
                from .core.patch_mult import n_part

                def mv(A, x, b):
                    _GS = self._GS
                    for _ in range(self.relax_iters):
                        self._r_gs[:] = _GS * (b - A * x)

                        n_part(self.block_sizes, self._offset_bs, self._offset_nnz,# offsets
                               self._A_inv_flat, self._r_gs, self._x_gs,           # data
                               self._partition_of_unity, self.omega                # scaling
                               )

                        x   += _GS.T * self._x_gs
                    return x
            else:
                def mv(A, x, b):
                    r_gs          = self._r_gs  # reuse the same memory for temporary data
                    omega         = self.omega  # omega can be changed during the run
                    _GS           = self._GS
                    _v_scale_glob = self._partition_of_unity

                    offset_bs     = self._offset_bs
                    offset_nnz    = self._offset_nnz
                    for _ in range(self.relax_iters):
                        r = b - A * x
                        r_gs[:] = _GS * r

                        for p, bs in enumerate(self.block_sizes):
                            r_t = r_gs[offset_bs[p]:offset_bs[p + 1]]
                            np.dot(self._A_inv_flat[offset_nnz[p]:offset_nnz[p + 1]].reshape((bs, bs)),
                                   r_t, out=r_t)

                        r_gs[:] = omega * (_v_scale_glob * r_gs)
                        x += _GS.T * r_gs
                    return x
        else:
            raise NotImplementedError('vanka_type=%s not implemented' % self.vanka_type)

        setattr(self, "relax", staticmethod(mv))

    def set_omega(self, omega):
        self.omega = omega
