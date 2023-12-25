import scipy.sparse as sp
import numpy as np


class BlockMatrix(object):
    """Custom Block Matrix Class.

    Allows operations (+,-,*) between two block-structured (sparse)
    matrices and have the result retain the block structure.
    The assumption is that all the dimensions of the blocks between
    two matrices match up exactly.

    (Why not use scipy.sparse.bmat? It converts block to a single
    sparse matrix, which is not always useful.)

    Objects from this class are used in stokes.py and stokes_mg.py
    to construct fine and coarse level coarse-grid operarots
    (M, B, BT, C) by operating directly on the bmat that contains
    them.

    e.g. | P1     |.| M   B|
         |      P2| | BT  C|
    """

    def __init__(self, bmat_list, dtype="csr"):
        """Initialize BlockMatrix object."""
        assert dtype == "csr", "Only csr matrix type support for now."
        self._nrows = len(bmat_list)
        self._ncols = len(bmat_list[0])
        self._bmat = bmat_list

        self.update_shape()

    def update_shape(self):
        """
        Update the shape of the block matrix.
        When the operators are modified in place, which is the case when constructing
        low order rediscretization operators, the shape of the block matrix needs to be
        updated.
        """
        try:
            Arow = []
            Acol = []
            for i in range(self._nrows):
                Arow.append([])
                Acol.append([])
                for j in range(self._ncols):
                    if self[i, j] is not None:
                        Arow[-1].append(self[i, j].shape[0])
                        Acol[-1].append(self[i, j].shape[1])
                    else:
                        Arow[-1].append(0)
                        Acol[-1].append(0)

            self.shape = (
                np.sum(np.array(Arow), axis=0).max().astype(int),
                np.sum(np.array(Acol), axis=1).max().astype(int),
            )
        except Exception as e:
            print("exception:\n", e)
            self.shape = None

        return self.shape

    def __getitem__(self, ij):
        i, j = ij
        return self._bmat[i][j]

    def __setitem__(self, ij, a):
        i, j = ij
        self._bmat[i][j] = a

    def __delitem__(self, ij):
        i, j = ij
        tmp = self._bmat[i][j]
        self._bmat[i][j] = None
        del tmp

    def __sub__(self, other):
        assert self._nrows == other._nrows, "Number of rows does not match."
        assert self._ncols == other._ncols, "Number of columns does not match."

        bmat_new = BlockMatrix(
            [[None for _ in range(self._ncols)] for _ in range(self._nrows)]
        )
        for i in range(self._nrows):
            for j in range(self._ncols):
                if self[i, j] == None and other[i, j] == None:
                    continue
                elif self[i, j] == None:
                    bmat_new[i, j] = -1 * other[i, j]
                elif other[i, j] == None:
                    bmat_new[i, j] = self[i, j]
                else:
                    bmat_new[i, j] = self[i, j] - other[i, j]
        return bmat_new

    def __add__(self, other):
        assert self._nrows == other._nrows, "Number of rows does not match."
        assert self._ncols == other._ncols, "Number of rows does not match."

        bmat_new = BlockMatrix(
            [[None for _ in range(self._ncols)] for _ in range(self._nrows)]
        )
        for i in range(self._nrows):
            for j in range(self._ncols):
                if self[i, j] == None and other[i, j] == None:
                    continue
                elif self[i, j] == None:
                    bmat_new[i, j] = other[i, j]
                elif other[i, j] == None:
                    bmat_new[i, j] = self[i, j]
                else:
                    bmat_new[i, j] = self[i, j] + other[i, j]

        return bmat_new

    def __mul__(self, other):
        assert self._ncols == other._nrows, "A*B: ncol(A) != nrows(B)."
        bmat_new = [[None for _ in range(self._ncols)] for _ in range(self._nrows)]

        for i in range(self._nrows):
            for j in range(other._ncols):
                sum_ = None
                for k in range(self._ncols):
                    lmat = self[i, k]
                    rmat = other[k, j]

                    if lmat != None and rmat != None:
                        assert (
                            lmat.shape[1] == rmat.shape[0]
                        ), "Dimension mismatch A_{%d,%d}xB_{%d,%d}." % (i, k, k, j)

                        if sum_ is None:
                            sum_ = sp.csr_matrix(
                                (lmat.shape[0], rmat.shape[1]), dtype=np.float64
                            )
                        sum_ += lmat * rmat

                bmat_new[i][j] = sum_

        return BlockMatrix(bmat_new)

    def tocsr(self):
        """Convert to csr matrix.

        The matrix is constructed in the self.__getattr__.
        """
        return getattr(self, "_bmat_csr")

    def toarray(self):
        return self.bmat.toarray()

    def __getattr__(self, attr):
        if attr == "T":
            AT = BlockMatrix(
                [[None for _ in range(self._nrows)] for _ in range(self._ncols)]
            )

            for r in range(self._nrows):
                for c in range(self._nrows):
                    if self[r, c] != None:
                        AT[c, r] = self[r, c].T

            return AT
        elif attr == "bmat":
            return sp.bmat(self._bmat)
        elif attr == "_bmat_csr":
            self._bmat_csr = self.bmat.tocsr()
            return self._bmat_csr
        else:
            raise ValueError("attributed %s not implemented." % attr)


if __name__ == "__main__":
    import numpy as np
    from random import randint

    for _ in range(10):
        # multiplication test
        r1 = randint(10, 20)  # 5
        r2 = randint(5, 9)  # 2
        A = sp.csr_matrix(np.random.rand(r1 * r1).reshape((r1, r1)))
        B = sp.csr_matrix(np.random.rand(r2 * r1).reshape((r2, r1)))
        BT = sp.csr_matrix(np.random.rand(r2 * r1).reshape((r1, r2)))
        C = sp.csr_matrix(np.random.rand(r2 * r2).reshape((r2, r2)))

        cr1 = r1 - randint(1, 9)
        cr2 = r2 - randint(1, 4)  # 1
        P1 = sp.csr_matrix(np.random.rand(r1 * cr1).reshape((r1, cr1)))
        P2 = sp.csr_matrix(np.random.rand(r2 * cr2).reshape((r2, cr2)))

        K = sp.bmat([[A, BT], [B, C]], format="csr")
        P = sp.bmat([[P1, None], [None, P2]], format="csr")
        true_result = (P.T * K * P).toarray()
        true_K_diff = (K - K.T).toarray()

        K = BlockMatrix([[A, BT], [B, C]])
        P = BlockMatrix([[P1, None], [None, P2]])
        R = BlockMatrix([[P1.T, None], [None, P2.T]])
        bmat12 = R * K * P
        bmat_result = bmat12.toarray()

        assert np.allclose(
            true_result, bmat_result
        ), "BlockMatrix multiplication is broken"
        # subtraction test
        assert (
            np.linalg.norm((R - R).tocsr().data) < 1e-8
        ), "BlockMatrix subtraction is broken"
        # addition test
        K2 = (bmat12 + bmat12).toarray()
        assert np.allclose(true_result * 2, K2), "BlockMatrix addition is broken"
        # Transpose test
        my_K_diff = (K - K.T).toarray()
        assert np.allclose(my_K_diff, true_K_diff), "BlockMatrix transpose is broken"
