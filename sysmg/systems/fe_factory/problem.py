from firedrake import *


class Problem(object):
    """Outline a class that can be used separately from sysmg code
    if needed."""

    def mesh(self, distribution_parameters):
        raise NotImplementedError

    def function_space(self, mesh, elem_type, order, name=None, scalar=True):
        if scalar:
            V = FunctionSpace(mesh, elem_type, order, name=name)
        else:
            V = VectorFunctionSpace(mesh, elem_type, order, name=name)
        return V

    def residual(self):
        raise NotImplementedError

    def bcs(self, Z):
        raise NotImplementedError

    def has_nullspace(self):
        raise NotImplementedError

    def nullspace(self, Z):
        if self.has_nullspace():
            MVSB = MixedVectorSpaceBasis
            return MVSB(Z, [Z.sub(0), VectorSpaceBasis(constant=True)])
        else:
            return None

    def mesh_size(self, u, domain_type):
        ## alfi's
        mesh = u.ufl_domain()
        if domain_type == "facet":
            dim = u.ufl_domain().topological_dimension()
            return FacetArea(mesh) if dim == 2 else FacetArea(mesh) ** 0.5
        elif domain_type == "cell":
            return CellSize(mesh)

    def rhs(self, Z):
        return None
