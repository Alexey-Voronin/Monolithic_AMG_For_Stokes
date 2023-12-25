import numpy as np
from scipy.sparse.linalg import splu

from .relaxation import System_Relaxation


class block_solve(System_Relaxation):
    """
    Block-diagonal relaxation function.

    """

    # TODO: Generalize to any block not just pressure.
    def __init__(self, stokes, params):
        """
        stokes (Stokes): stokes object from sysmg.systems.stokes.

        Args:
            stokes: sysmg.systems.stokes.StokesSystem
                Stokes object used to pass in Stokes submatrices.
                The only required operators

            params: dict
                Dictionary object specifying all the parameters needed for BS
                relaxation.

                Keys:
                'operator': (str)
                    Operator to use for solve.  Options are:
                    "pressure mass", "pressure stiffness".
                'blocksize'          : int
                    dimension of the blocks in the diagonal matrix
                'iterations'          : tuple
                    (# of pre-, #  of post-) relaxation sweeps.
                'block_solver'      : str
                    type of block solver to use.
                    Current options are ('pinv', 'splu', 'inv')
                debug: (bool)
                    If True, then check that the block-diagonal solve does
                    what is expected.

        Returns:
            Nothing

        """
        super().__init__(params)
        self._blocksize = params.get("blocksize", 1)
        self._block_solver = params.get("block_solver", "inv")
        self._sa_amg_setup_params = params.get("sa_amg_setup_params", {})
        self._sa_amg_solve_params = params.get("sa_amg_solve_params", {})

        # input
        self._system = stokes
        self._operator = params.get("operator", "pressure mass")
        self._structured = params.get("structured", False)
        self._M = (
            stokes.mass_bmat[1, 1]
            if self._operator == "pressure mass"
            else stokes.stiffness_bmat[1, 1]
        )
        self._Asize = self._M.shape[0]
        self._debug = params.get("debug", False)
        self._dx = np.zeros(self._Asize)
        self._null = getattr(stokes, "nullspace", None)
        if self._null is not None:
            # only keep the nullspace in the pressure space
            self._null = self._null[stokes.velocity_nodes() :]

        if self._blocksize > self._Asize:
            # KLUGE:  Pass in a dummy large blocksize if you do not know
            #         the mass matrix size in advance.
            self._blocksize = self._Asize

        self._nblocks = self._Asize // self._blocksize

        if self._blocksize < self._Asize:
            Pp_sort = getattr(stokes, "Pbmat_sort")[1, 1]
            # block-diagonal solve requires the knowledge of how the DoFs are sorted.
            if (
                stokes.params["dof_ordering"].get("lexicographic", False)
                and not stokes.keep
            ):
                # if the sorting is lexicographic, then need to keep the sorting
                # operator around, especially if it is going to be deleted by
                # stokes_mg later.
                Pp_sort = getattr(stokes, "Pbmat_sort")[1, 1]
                self._Psort = Pp_sort.copy() if Pp_sort is not None else None
            else:
                # if not lexicographic or the sorting operator is not deleted,
                # then just use the sorting operator from stokes class.
                self._Psort = Pp_sort
        else:
            self._Psort = None

        # setup
        self._setup()

        if self.accel:
            self._xt = np.zeros((self._Asize,))

    def _setup(self):
        Psort = self._Psort
        M = self._M
        blocksize = self._blocksize
        # lexicographic sorting breaks up the block-diagonal nature of
        # the DG mass matrix. Hence, I do some `unsorting` here to recover it.
        # When DoFs are not sorted Psort is None.
        self.needs_sorting = Psort != None and blocksize < M.shape[0]
        M_sorted = Psort.T * M * Psort if self.needs_sorting else M

        if self._structured or self._blocksize == self._Asize:
            M_block = M_sorted[0:blocksize, 0:blocksize]
            if self._block_solver in ["pinv", "inv"]:
                M_block_inv = getattr(np.linalg, self._block_solver)(M_block.toarray())
                self.M_block_inv = lambda x: M_block_inv @ x
            elif self._block_solver == "splu":
                M_block_inv = splu(M_block.tocsc())
                self.M_block_inv = lambda x: M_block_inv.solve(x)
            elif self._block_solver == "diag":
                M_block_inv = 1.0 / M_block.diagonal()
                self.M_block_inv = lambda x: x * M_block_inv
            elif self._block_solver == "sa_amg":
                from pyamg import smoothed_aggregation_solver

                ml = smoothed_aggregation_solver(
                    M_block.tocsr(), **self._sa_amg_setup_params
                )
                self._ml = ml
                tol = self._sa_amg_solve_params.get("tol", 1e-12)
                maxiter = self._sa_amg_solve_params.get("maxiter", 100)
                self.M_block_inv = lambda r: ml.solve(r, tol=tol, maxiter=maxiter)
                M_block_inv = None
            else:
                raise ValueError("Block solver not recognized.")
            self.M_block_inv_unwrapped = M_block_inv

        else:
            assert (
                self._block_solver == "inv"
            ), "Only 'inv' solver is supported for unstructured matrices."
            M_blocks = np.zeros((blocksize, blocksize, self._nblocks))
            M_sorted.sort_indices()
            for i in range(self._nblocks):
                row_ptr = M_sorted.indptr[i * blocksize : (i + 1) * blocksize + 1]
                col_s, col_e = row_ptr[0], row_ptr[-1]
                data = M_sorted.data[col_s:col_e].reshape((blocksize, blocksize))
                M_blocks[:, :, i] = np.linalg.inv(data)
            self.M_block_inv = M_blocks

        if self._debug and self._blocksize > self._Asize:
            assert self._Asize < 1e3, "Use smaller matrix for debugging."
            M_dense = M_sorted.toarray()
            for i in range(M_dense.shape[0] // blocksize):
                M_dense[
                    i * blocksize : (i + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ] -= M_block
            assert np.sum(np.ravel(M_dense)) < 1e-12, "block-diag solve is not correct."

            print("block-diag solve is correct.")
            self.M_block = M_block
            self.Msorted = M_sorted

        return M

    def relax(self, A, x, b):
        """
        Apply Block Diagonal relaxation.

        Args:
            A: (csr_matrix)
                Systems matrix (ignored, but needed to be compatible with pyamg
                relaxation).

            x: (numpy.ndarray)
                Initial guess.

            b: (numpy.ndarray)
                Right hand side.

        Returns:
            (numpy.ndarray):  Solution vector.

        Notes:
            If you would like to change the parameters then modify the
            relaxation object via the helper functions.

        """
        blocksize = self._blocksize
        Psort = self._Psort
        null = self._null
        needs_sorting = self.needs_sorting

        for _ in range(self.relax_iters):
            r = b - self._M * x
            if needs_sorting:
                r = Psort.T * r
                x = Psort.T * x

            if self._blocksize == self._Asize:
                x += self.M_block_inv(r)
            elif self._structured:
                if self._block_solver in ["pinv", "inv"]:
                    np.einsum(
                        "ij,kj->ki",
                        self.M_block_inv_unwrapped,
                        r.reshape((-1, blocksize)),
                        out=x.reshape((-1, blocksize)),
                    )
                else:
                    for i in range(self._nblocks):
                        start, end = i * blocksize, (i + 1) * blocksize
                        x[start:end] += self.M_block_inv(r[start:end])
            else:
                if self._block_solver in ["pinv", "inv"]:
                    np.einsum(
                        "ijk,kj->ki",
                        self.M_block_inv,
                        r.reshape((-1, blocksize)),
                        out=x.reshape((-1, blocksize)),
                    )
                else:
                    for i in range(self._nblocks):
                        start, end = i * blocksize, (i + 1) * blocksize
                        x[start:end] += self.M_block_inv[:, :, i] @ r[start:end]

            x = Psort * x if needs_sorting else x
            if null is not None:
                x -= (
                    np.dot(null, x) * null / np.dot(null, null)
                )  # project out nullspace

        return x
