import numpy as np
import scipy.sparse as sp
from firedrake import *

from .util.dof_handler import DoFHandler
from .util.mesh_wrapper import MeshWrapper


class System(MeshWrapper, DoFHandler):
    """Generic disretized system object.

    This class is to be extended by specific applications (e.g. diffusion,
    Stokes).
    """

    dim = None

    def __init__(self, system_param=None):
        """Initialize System Object.

        System class is responsible for dealing with domain topology and DoF
        ordering. It also includes a few functions common to most systems,
        such as computing and plotting solution; construction of mass and
        stiffness matrices from test/trial functions.

        Everything else should be handled by children classes.

        Allows construction of empty system objects, which are used to
        store coarse-grid operators.
        """
        if system_param is None:
            # empty system object
            # used to store MG coarse-grid operators in StokesMG class.
            return

        self.params = system_param
        mesh = system_param["mesh"]
        if hasattr(self, "elem_type"):
            if self.elem_type == ("CG", "DG") and not isinstance(mesh, dict):
                mesh = {"mesh": mesh, "bary": True}

        super().__init__(mesh)

    def _form_system(self, plot, solve, **kwargs):
        """Form system."""
        pass

    def _solve_sys(self, a, L, V, bcs, nullspace=None):
        fsol = Function(V)
        fsol.assign(0)
        solve(a == L, fsol, bcs=bcs, nullspace=nullspace)
        sol = np.copy(fsol.vector().array())

        return fsol, sol

    def _plot_sys(self):
        import matplotlib.pyplot as plt

        if self.dim == 1:
            ucoord = self.dof_coord[-1]["u"]
            nodes = self.bcs_nodes_hier[-1]["u"]
            yvals = np.zeros(len(ucoord))
            plt.scatter(ucoord, yvals)
            plt.scatter(ucoord[nodes], yvals[nodes], marker="x")
            plt.plot(ucoord, self.usol)
            plt.show()
            return

        spaces = self.space_hier[-1].subfunctions
        nplots = len(spaces)
        if nplots == 1:
            fig, axes = plt.subplots()
            l = tricontourf(self.fsol, axes=axes)
            triplot(self.mesh, axes=axes, interior_kw=dict(alpha=0.05))
            plt.legend()
            plt.colorbar(l)
            plt.show()
        else:
            # 3D plots don't work tetrahedron cells :(
            """
            fig = plt.figure(figsize=plt.figaspect(0.5))
            ax = fig.add_subplot(1, 2, 1, projection='3d')
            V, P = spaces
            v, p = self.fsol.subfunctions
            leg = tricontourf(v, axes=ax)
            ax.set_title(V.name)

            ax = fig.add_subplot(1, 2, 2, projection='3d')
            leg = tricontourf(p, axes=ax)
            ax.set_title(P.name)
            fig.tight_layout()
            plt.show()
            """
            if self.dim == 2:
                fig, axes = plt.subplots(
                    nrows=1, ncols=nplots, figsize=(3.5 * nplots, 3)
                )
                for ax, V, v in zip(axes, spaces, self.fsol.subfunctions):
                    leg = tricontourf(v, axes=ax)
                    triplot(self.mesh, axes=ax, interior_kw=dict(alpha=0.05))
                    ax.set_title(V.name)
                    plt.colorbar(leg, ax=ax)

                fig.tight_layout()
                plt.show()
            else:
                u, p = self.fsol.subfunctions
                u.rename("Velocity")
                p.rename("Pressure")
                File("stokes_3D.pvd").write(u, p)

    def plot_guess(self, u, mesh=False):
        """Plot solution.

        To be overwritten by implementing class.
        """
        pass

    def _homogeneous_bcs(self, bcs_type, component="u"):
        """Get homogeneous boundary condition.

        Homogeneous Dirichlet BCs are the most common choice/option.
        In case PDE system we may want to enforce these BCs just on
        one of the fiedlds, hence the space_hier input.
        """
        meshes = self.meshes
        bcs_hier = []
        bcs_nodes_hier = []

        if bcs_type is None:
            return
        elif bcs_type.lower() == "dirichlet":
            for (i, mesh), Vm in zip(enumerate(meshes), self.space_hier):
                # find the desired component
                Vm_comp = Vm.subfunctions
                for V in Vm_comp:
                    if V.name == component:
                        break

                # figure out the dimension of Vector space and construct
                Velem = V.ufl_element()
                dim = Velem.cell().geometric_dimension()
                val = tuple([0] * dim) if isinstance(Velem, VectorElement) else 0.0
                # val needs to be of the same number of components as the
                # Function space
                bcs = [DirichletBC(V, Constant(val), tuple(self.markers))]

                bcs_nodes = list(set(list(bcs[0].nodes)))

                bcs_hier.append({component: bcs})
                bcs_nodes_hier.append({component: bcs_nodes})
        else:
            raise ValueError("bcs=%s has not been defined yet." % bcs_type)

        return bcs_hier, bcs_nodes_hier

    def get_mass_matrix(self, test, trial, bcs, block=(0, 0)):
        """Get mass matrix."""
        integrand = inner(test, trial)
        Mass = assemble(integrand * dx, bcs=bcs, mat_type="nest", sub_mat_type="aij")
        if block == None:
            return sp.csr_matrix((Mass.petscmat.getValuesCSR())[::-1])
        else:
            return sp.csr_matrix(
                (Mass.petscmat.getNestSubMatrix(*block).getValuesCSR())[::-1]
            )

    def get_stiffness_matrix(self, test, trial, bcs, block=(0, 0), elem_type="CG"):
        """Get stiffness matrix.

        User needs to provide trial and test spaces and the boundary conditions.
        """

        if elem_type == "DG":
            mesh = test.ufl_domain()
            h = assemble(CellDiameter(mesh) * dx)
            myalpha = 20.0
            alpha = Constant(myalpha)
            n = FacetNormal(mesh)
            u, v = test, trial
            a = (
                inner(grad(u), grad(v)) * dx
                + alpha * (h ** (-1)) * inner(jump(u), jump(v)) * dS
                - inner(jump(u, n), avg(grad(v))) * dS
                - inner(jump(v, n), avg(grad(u))) * dS
                # Dirichlet terms were dropped
            )
        else:  # CG
            a = inner(grad(test), grad(trial)) * dx

        Stiffness = assemble(a, bcs=bcs, mat_type="nest", sub_mat_type="aij")
        if block == None:
            return sp.csr_matrix((Stiffness.petscmat.getValuesCSR())[::-1])
        else:
            return sp.csr_matrix(
                (Stiffness.petscmat.getNestSubMatrix(*block).getValuesCSR())[::-1]
            )

    def print_param(self, params=None, shift="\t"):
        """Print system parameters."""
        if params == None:
            # first call gets the parameters
            params = self.params
            # dump in dictionary form for easy copy/paste
            print(params, "\n---")

        # print in readable format
        if shift == "\t":
            print("Parameters:")

        for k, v in params.items():  # sorted(params):
            v = params[k]
            if isinstance(v, dict):
                print("%s-%s" % (shift, k))
                self.print_param(params=v, shift=(shift + "\t"))
            else:
                print("%s-%s\t=" % (shift, k), v)

    def get_setup_timings(self):
        """Return timings data for the Stokes system setup phase."""
        from sysmg.util.decorators import timings_data
        import pandas as pd

        # construct frame with all data: skip first 2 key identifiers ("setup:system:")
        setup_data = {}
        for k, v in timings_data.items():
            a, b, fun = k.split(":", 2)
            if f"{a}:{b}" == "setup:system":
                setup_data[fun] = v
        frame = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in setup_data.items()]))
        # sum up each column and sort decreasing order
        idx = np.argsort(np.nansum(frame.values, axis=0))[::-1]
        # reorder the frame
        sorted_frame = frame[frame.columns[idx]]

        return sorted_frame

    def reset_timers(self):
        from sysmg.util.decorators import timings_data

        to_pop = []
        for k, v in timings_data.items():
            if "setup:system" in k:
                to_pop.append(k)
        for k in to_pop:
            timings_data.pop(k)
