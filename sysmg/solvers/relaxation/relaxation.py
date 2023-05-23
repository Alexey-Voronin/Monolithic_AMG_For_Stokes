import numpy as np
import pyamg
import scipy.sparse as sp

from sysmg.util.decorators import timeit


class System_Relaxation(object):
    params = None
    empty_shell = False
    relax_iters = None
    accel = False
    accel_iter = None
    _xt = None  # dummy vector
    timings = None
    nullspace = None
    _Asize = None

    def __init__(self, params, level=0):
        self.params = params
        self.level = level

        if 'accel' in params:
            self.accel = True
            self.accel_iter = params["accel"]['iterations']
        self.eigs = np.zeros((1,))  # place holder
        # check if the setup is even necessary
        relax_iters = params["iterations"]
        accel_iter = self.accel_iter if self.accel else (1, 1)
        if relax_iters[0] * accel_iter[0] == 0 and relax_iters[1] * accel_iter[1] == 0:
            self.empty_shell = True

    def relax(self, A, x, b):
        """
        Apply relaxation in place.

        Args:
            A (csr_matrix):
                Systems matrix (ignored, but needed to be compatible with
                pyamg relaxation).

            x (numpy.ndarray):
                Initial guess.

            b: (numpy.ndarray)
                Right-hand side.

        Returns:
            (numpy.ndarray):  Solution vector.

        Notes:
            If you would like to change the parameters then modify the relaxation
            object via the helper functions.
        """
        pass

    @timeit("solution:solver:relaxation:")
    def relax_inplace(self, A, x, b):
        """
        In place relaxation.

        Args:
            A: (csr_matrix)
                Systems matrix
            x: (numpy.ndarray)
                Initial guess.
            b: (numpy.ndarray)
                Right hand side.

        Returns:
              level: (int)
                Level of the relaxation. If it wasn't initialized return 0.
                This information is used by the timings decorator.
        """
        if self.relax_iters == 0:
            return self.level
        elif self.accel and self.accel_iter > 0:  # wrap relaxation w/ GMRES
            # if relaxation has not been put into linear operator yet
            # do so now.
            if not hasattr(self, 'M_accel'):
                def mv(r):
                    xu = self._xt * 0.0
                    xu[:] = self.relax(A, xu, r)
                    #self.project_out_nullspace(xu)
                    return xu

                self.M_accel = sp.linalg.LinearOperator(A.shape, matvec=mv)
                np.random.seed(7)
                self.eigs = pyamg.krylov.fgmres(A, np.random.rand(A.shape[0]),
                                                x0=None,
                                                tol=1e-20,
                                                maxiter=1,  # = max_outer
                                                restrt=self.accel_iter,  # = max_inner
                                                M=self.M_accel,
                                                eig_bounds=True
                                                )[2]
            x[:] = pyamg.krylov.fgmres(A, b, x0=x,
                                       tol=1e-20,
                                       maxiter=1,  # = max_outer
                                       restrt=self.accel_iter,  # = max_inner
                                       M=self.M_accel)[0]
        else:
            x[:] = self.relax(A, x, b)

        #self.project_out_nullspace(x)

        return self.level

    def project_out_nullspace(self, x):
        if isinstance(getattr(self, 'nullspace', None), np.ndarray):
            null = self.nullspace
            x[:] -= null * np.dot(null, x) / np.dot(null, null)

    def presmoother(self, A, x, b, *args):
        self.relax_iters = self.params["iterations"][0]
        if self.accel:
            self.accel_iter = self.params["accel"]['iterations'][0]
        self.__call__(A, x, b, *args)

    def postsmoother(self, A, x, b, *args):
        self.relax_iters = self.params["iterations"][1]
        if self.accel:
            self.accel_iter = self.params["accel"]['iterations'][1]
        self.__call__(A, x, b, *args)

    def __call__(self, A, x, b, *args):
        """
        Call to relaxation. Allows us to store the entire relaxation
        object in pyamg's ml hierarchy, instead of only supplying
        an appropriate function call.

        Args:
            A: (csr_matrix)
                Systems matrix

            x: (numpy.ndarray)
                Initial guess.

            b: (numpy.ndarray)
                Right hand side.

            args: (dict)
                ignored.

        Returns:
            None, x will be modified in place.

        Notes:
            If you would like to chain parameters then modify the relaxation
            object via the helper functions.

        """
        self.relax_inplace(A, x, b)

    def aslinearoperator(self):
        """
        Return a linear operator that implements the relaxation.

        Returns:
            (scipy.sparse.linalg.LinearOperator):  Linear operator.

        """

        def mv(r):
            dx = np.zeros_like(r)
            self.presmoother(None, dx, r)
            return dx

        return sp.linalg.LinearOperator((self._Asize, self._Asize), matvec=mv)
