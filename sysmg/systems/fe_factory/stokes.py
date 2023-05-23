import numpy as np
from firedrake import *

from .problem import Problem


class StokesProblem(Problem):

    def __init__(self, msh, elem_order=(2, 1),
                 elem_type=('CG', 'CG'),
                 grad_div_stab=None,  # ('u', 1.) or ('p', 1.)
                 ):
        """
        elem_order: tuple of ints
            (velocity order, pressure order)
        elem_type: tuple of strings
            (velocity element type, pressure element type)
        grad_div_stab: tuple of strings and floats
            ('u', 1.) or ('p', 1.) or None
        """
        self.msh = msh
        self.dim = msh.geometric_dimension()
        self.elem_order = elem_order
        self.elem_type = elem_type
        self.grad_div_stab = grad_div_stab
        self.Z = self.function_space(self.msh, elem_type, elem_order)

        if elem_type[0] == 'CG' and elem_type[1] == 'CG':
            self.disc = 'Taylor-Hood'
        elif elem_type[0] == 'CG' and elem_type[1] == 'DG':
            self.disc = 'Scott-Vogelius'
        else:
            raise ValueError("Discretization with element types {str(elem_type) not recognized.")

        super().__init__()

    def function_space(self, msh, elem_type, order, name=('u', 'p'), scalar=False):
        assert not scalar, "Vector field."
        U = super().function_space(msh, elem_type[0], order[0], name[0], scalar=False)
        P = super().function_space(msh, elem_type[1], order[1], name[1], scalar=True)
        self.Z = U * P
        return self.Z

    def residual(self):
        u, p = TrialFunctions(self.Z)
        v, q = TestFunctions(self.Z)

        a = (inner(grad(u), grad(v)) - div(v) * p - q * div(u)) * dx

        if self.disc == 'Taylor-Hood' and self.grad_div_stab is not None:
            h = self.mesh_size(u, 'cell')
            if self.grad_div_stab[0] == 'u':
                a += self.grad_div_stab[1] * inner(grad(u), grad(v)) * dx
            elif self.grad_div_stab[0] == 'p':
                a += self.grad_div_stab[1] * (h ** 2) * inner(grad(p), grad(q)) * dx
            else:
                raise ValueError("Stabilization term not recognized.")

        return a

    def rhs(self, Z):
        v, _ = TestFunctions(Z)
        f = Constant(tuple([0] * self.dim))
        L = inner(f, v) * dx
        return L

    def bcs(self,
            bc_type,  # int or str
            boundary_markers,  # dictionary e.g. {'inflow' : 1,'sides' : 0, 'outflow' : 2}
            ):
        bcs = []

        Z = self.Z
        U, P = Z.subfunctions
        mesh = Z.mesh()
        assert U.name == 'u', 'check that the velocity is the first component.'

        bc_type = bc_type.lower()
        if bc_type in [None, 'periodic']:
            return bcs
        elif bc_type in ['in-out-flow']:
            # KLUGE: infer the flow direction by examining which side of the object
            # is the longest. Currently works for all meshes in the collection.
            coords = mesh.coordinates.dat.data_ro
            flow_direction = np.argmax(np.max(coords, axis=0) - np.min(coords, axis=0))
            flow_vector = np.zeros((self.dim,))
            flow_vector[flow_direction] = 1

            val0 = tuple([0] * self.dim)
            val1 = as_vector(flow_vector)
            bm = boundary_markers

            if 'inflow' in bm.keys():
                inflow = (bm['inflow'],)
                sides = (bm['sides'],
                         # bm['outflow'] # Neumann
                         )
            elif self.dim == 2:
                inflow = (bm['left'],)
                sides = (bm['top'],
                         bm['bottom'],
                         # bm['right'] # neumann
                         )
            elif self.dim == 3:
                inflow = (bm['left'],)
                sides = (bm['top'], bm['bottom'],
                         bm['front'], bm['back'],
                         # bm['right'] # neumann
                         )

            xyz = SpatialCoordinate(Z.mesh())
            y, x = xyz[0], xyz[1]
            mesh_max = np.max(mesh.coordinates.dat.data_ro, axis=0)
            mesh_min = np.min(mesh.coordinates.dat.data_ro, axis=0)
            center = float((mesh_max[0]+mesh_min[0]) / 2.)
            if 'inflow' in bm.keys() and abs(center) < 1e-6:
                # parabolic in-flow boundary condition
                ms = 1
                fx = -(ms * abs(x)) ** 2 + (mesh_max[0] * ms) ** 2

                val1 = as_vector([fx] + [0] * (self.dim - 1))
                bcs = [
                    DirichletBC(U, Constant(val0), sides),
                    DirichletBC(U, val1, inflow),
                ]
            else:
                bcs = [
                    DirichletBC(U, Constant(val0), sides),
                    DirichletBC(U, val1, inflow),
                ]
        elif bc_type in ['lid-driven-cavity']:
            val0 = tuple([0] * self.dim)

            xyz = SpatialCoordinate(Z.mesh())
            nsides = len(boundary_markers)
            if nsides == 3:
                lid_side = 'inflow'
                y, x = xyz[0], xyz[1]
            else:
                lid_side = 'top'
                x, y = xyz[0], xyz[1]

            # parabolic in-flow boundary condition
            mesh_max = np.max(mesh.coordinates.dat.data_ro, axis=0)
            mesh_min = np.min(mesh.coordinates.dat.data_ro, axis=0)
            center = float((mesh_max[0]+mesh_min[0]) / 2.)
            if abs(center) < 1e-6:
                # here to accomodate airfoil mesh, which is defined
                # over domain [-8,8]x[-8,8]
                ms = 1
                fx = -(ms * abs(x)) ** 2 + (mesh_max[0] * ms) ** 2
            else:
                # should be able to accomodate mesged defined over
                # domains [0,L]x[0,L]
                ms = 2 * mesh_max[0] * mesh_max[1]
                fx = -(ms * (x - center)) ** 2 + (center * ms) ** 2

            if nsides == 3:
                val1 = as_vector([0] * (self.dim - 1) + [fx])
            else:
                val1 = as_vector([fx] + [0] * (self.dim - 1))

                if self.dim == 3:
                    fx = -(ms * (x - center)) ** 2 + (center * ms) ** 2
                    fy = -(ms * (y - center)) ** 2 + (center * ms) ** 2
                    val1 = as_vector([fx, fy] + [0])

            bm = boundary_markers
            lid_id = (bm[lid_side],)
            other_ids = tuple([s for s in bm.values() \
                               if s != lid_id[0]])

            bcs = [
                DirichletBC(U, Constant(val0), other_ids),
                DirichletBC(U, val1, lid_id),
            ]
        else:
            raise ValueError(f"Boundary condition {bc_type} not recognized.")

        return bcs

    def has_nullspace(self, bc_type):

        if bc_type in ['in-out-flow', 'periodic']:
            return False
        elif bc_type in ['lid-driven-cavity', None]:
            return True
        else:
            raise ValueError(f"Boundary condition {bc_type} not recognized.")

    def nullspace(self, bc_type):
        if self.has_nullspace(bc_type):
            MVSB = MixedVectorSpaceBasis
            return MVSB(self.Z, [self.Z.sub(0), VectorSpaceBasis(constant=True)])
        else:
            return None

    def mass_matrix(self, component, bcs):
        if component == 'u':
            F = self.Z.sub(0)
        elif component == 'p':
            F = self.Z.sub(1)
        else:
            raise ValueError('component must be "u" or "p".')

        u = TrialFunction(F)
        v = TestFunction(F)

        a = inner(u, v) * dx
        M = assemble(a, bcs=bcs)
        return M

    def stiffness_matrix(self, component, bcs):
        if component == 'u':
            F = self.Z.sub(0)
        elif component == 'p':
            F = self.Z.sub(1)
        else:
            raise ValueError('component must be "u" or "p".')

        u, v = TrialFunction(F), TestFunction(F)
        if F.ufl_element().family() == 'Lagrange':
            a = inner(grad(u), grad(v)) * dx
        elif F.ufl_element().family() == 'Discontinuous Lagrange':
            assert len(bcs) == 0, 'bcs not supported for discontinuous elements.'

            mesh = v.extract_unique_domain()
            h = assemble(CellDiameter(mesh) * dx)
            myalpha = 20.0
            alpha = Constant(myalpha)
            n = FacetNormal(mesh)
            a = (
                    inner(grad(u), grad(v)) * dx
                    + alpha * (h ** (-1)) * inner(jump(u), jump(v)) * dS
                    - inner(jump(u, n), avg(grad(v))) * dS
                    - inner(jump(v, n), avg(grad(u))) * dS
                # Dirichlet terms were dropped
            )
        else:
            raise ValueError(f'Unknown element type: {F.ufl_element().family()}.')

        K = assemble(a, bcs=bcs)
        return K

    def plot_solution(self, up, items=['mesh', 'u', 'div u', 'p', 'grad p'], save_name=None):
        import matplotlib.pyplot as plt
        u, p = up.subfunctions

        nplots = len(items)
        f, axs = plt.subplots(1, nplots, figsize=(int(6.5 * nplots), 6))
        counter = 0
        if 'mesh' in items:
            ax = axs[counter];
            counter += 1
            triplot(self.msh, axes=ax);
            ax.set_aspect('equal')
            ax.set_title(r"mesh", fontsize=20);
            ax.legend(fontsize=20)
        if 'u' in items:
            ax = axs[counter];
            counter += 1
            l = streamplot(u, axes=ax);
            #l = tricontourf(u, axes=ax)
            ax.set_aspect('equal')
            ax.set_title(r"$u$", fontsize=20);
            plt.colorbar(l)
        if 'p' in items:
            ax = axs[counter];
            counter += 1
            l = tricontourf(p, axes=ax);
            ax.set_aspect('equal')
            ax.set_title(r"$p$", fontsize=20);
            plt.colorbar(l)
        if 'div u' in items:
            ax = axs[counter];
            counter += 1
            P = p.function_space()
            div_u = project(div(u), P)
            l = tricontourf(div_u, axes=ax);
            ax.set_aspect('equal')
            ax.set_title(r"$\nabla\cdot u$", fontsize=20);
            plt.colorbar(l)
        if 'grad p' in items:
            ax = axs[counter];
            counter += 1
            U = u.function_space()
            grad_p = project(grad(p), U)
            l = tricontourf(grad_p, axes=ax);
            ax.set_aspect('equal')
            ax.set_title(r"$\nabla p$", fontsize=20);
            plt.colorbar(l)

        if save_name is not None:
            plt.savefig(save_name, format='pdf', bbox_inches='tight')
        else:
            plt.show()

    def save_function(self, up, file_name='up'):
        u, p = up.subfunctions
        u.rename("Velocity")
        p.rename("Pressure")
        File(f'{file_name}.pvd').write(u, p)

    def assemble_and_plot(self, bc_type, bc_markers):

        bcs = self.bcs(bc_type, bc_markers)
        nullspace = self.nullspace('lid-driven-cavity')
        up = Function(self.Z);
        up.assign(0)
        solve(self.residual() == self.rhs(self.Z), up, bcs=bcs, nullspace=nullspace)
        self.plot_solution(up)

        return up


if __name__ == "__main__":
    from firedrake import *
    import matplotlib.pyplot as plt

    mesh = UnitSquareMesh(10, 10)
    """
    # Taylor-Hood (2,1) on structured mesh w/ lid-driven cavity bcs
    problem = StokesProblem(mesh, elem_order=(2,1), elem_type=('CG', 'CG'))
    problem.assemble_and_plot('lid-driven-cavity', {'top': 4, 'bottom': 3, 'left': 1, 'right': 2})
    # Taylor-Hood (2,1) on structured mesh w/ in-out-flow bcs
    problem = StokesProblem(mesh, elem_order=(2,1), elem_type=('CG', 'CG'))
    problem.assemble_and_plot('in-out-flow', {'top': 4, 'bottom': 3, 'left': 1, 'right': 2})
    """
    # Scott-Vogelius (2,1) on structured mesh w/ lid-driven cavity bcs
    from sysmg.systems.util.bary_meshes import bary_reg_ref_mesh

    mesh = bary_reg_ref_mesh(mesh)
    problem = StokesProblem(mesh, elem_order=(2, 1), elem_type=('CG', 'DG'))
    problem.assemble_and_plot('lid-driven-cavity', {'top': 4, 'bottom': 3, 'left': 1, 'right': 2})
    # Scott-Vogelius (2,1) on structured mesh w/ in-out-flow bcs
    problem = StokesProblem(mesh, elem_order=(2, 1), elem_type=('CG', 'DG'))
    problem.assemble_and_plot('in-out-flow', {'top': 4, 'bottom': 3, 'left': 1, 'right': 2})
