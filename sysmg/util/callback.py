import numpy as np


class CallBack(object):
    """Callback class used to compute and record residual history."""

    resid = None
    A = None
    b = None

    def __init__(self, A=None, b=None, M=None):
        self.resid = []
        self.A = A
        self.b = b
        self.M = M

    def __call__(self, x=None):
        if self.A is None:
            # x is already the norm of residual
            self.resid.append(x)
        else:
            r = self.b - self.A * x
            if self.M is None:
                self.resid.append(np.linalg.norm(r))
            else:
                self.resid.append(np.linalg.norm(self.M * r))

    def get_residuals(self):
        return self.resid
