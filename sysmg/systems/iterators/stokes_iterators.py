import numpy as np
from time import time_ns
from firedrake import Mesh, UnitCubeMesh, UnitSquareMesh

from ..stokes import Stokes
from .system_iterator import SystemIterator

import os
import glob


class StructuredStokesIterator(SystemIterator):
    """Iterator object that generates structured Stokes systems on demand.


    Parameters
    ----------
    system_params : dict
        Dictionary containing the parameters for the system.
    start_idx : int
        Index of the first mesh to generate.
    end_idx : int
        Index of the last mesh to generate.
    quadrilateral : bool
        If True, generate quadrilateral meshes.
    max_dofs : int
        Maximum number of DoFs for the system.
    dim : int
        Dimension of the mesh.
    NEx : array
        Array containing the number of elements in each direction.
    shape : str
        Shape of the mesh. Currently only square and L-shaped  meshes
        are supported. If None, a square mesh is generated.


    Sample input:
    -------------
    mesh = { 'NE' : None, 'L' : (1,1),
                             'mesh_hierarchy'    : True,
                             'quadrilateral' : False}

    system_params = {'mesh'             : mesh
                    'discretization'    : { 'elem_type' : ('CG', 'CG),
                                            'order' : (2,1),
                                            'bcs': 'lid-driven-cavity'},
                    'dof_ordering'      : { 'split_by_component': True,
                                            'lexicographic': True},
                    'additional'        : { 'lo_fe_precond' :True},
                    }
    """

    def __init__(
        self,
        system_params,
        start_idx=None,
        end_idx=None,
        quadrilateral=False,
        max_dofs=1e5,
        dim=2,
        NEx=None,
        shape=None,
    ):
        """Initialize the iterator object."""
        super().__init__(system_params, max_dofs)

        self.start = 0 if start_idx == None else start_idx
        self.end = end_idx if NEx == None else len(NEx)
        assert self.end != None, "Either provide end_idx or NEx."

        self.NEx = NEx if NEx is not None else np.array([2**i for i in range(0, 20)])
        self.NEx = self.NEx[self.start : self.end]
        self.dim = dim
        self.shape = shape
        self.quadrilateral = quadrilateral

    def __repr__(self):
        return "Structured %dD Stokes: %s" % (
            self.dim,
            self.system_params["discretization"]["bcs"],
        )

    def __next__(self):
        if self.count < len(self.NEx):
            NEx = self.NEx[self.count]
            tic = time_ns()

            # Expected number of DoFs
            v_dofs = 1
            p_dofs = 1
            for _ in range(self.dim):
                v_dofs *= 2 * NEx + 1
                p_dofs *= NEx + 1
            v_dofs *= self.dim

            if v_dofs + p_dofs >= self.max_dofs:
                print("ISSUE: systems is bigger than expected")
                raise StopIteration

            if self.shape == "L":
                from sysmg.systems.util.other_meshes import RectangleMesh_MyCoord

                mesh = RectangleMesh_MyCoord(
                    NEx,
                    NEx,
                    2,
                    2,
                    quadrilateral=self.quadrilateral,
                    prescribed_coord=None,
                    shape="L",
                )
            elif self.dim == 2:
                mesh = UnitSquareMesh(NEx, NEx, quadrilateral=self.quadrilateral)
            else:
                mesh = UnitCubeMesh(
                    NEx,
                    NEx,
                    NEx,  # quadrilateral=self.quadrilateral
                )

            self.system_params["mesh"] = mesh
            stokes = Stokes(self.system_params)
            stokes.NE_hier = getattr(stokes, "NE_hier", [[NEx] * self.dim])

            self.build_time = (time_ns() - tic) / 1e9
            self.count += 1
            stokes.structured = True
            if hasattr(stokes, "lo_fe_sys"):
                stokes.lo_fe_sys.structured = True
            return stokes
        else:
            raise StopIteration


class UnstructuredStokesIterator(SystemIterator):
    """Iterator object that generates unstructured Stokes systems on demand.

    Parameters
    ----------
    system_params : dict
        Dictionary containing the parameters for the system.
    name_idx : int
        Index of the mesh to generate. See the code for the list of
        available meshes.
    start_idx : int
        Index of the first mesh to generate.
    end_idx : int
        Index of the last mesh to generate.
    max_dofs : int
        Maximum number of DoFs for the system.
    dim : int
        Dimension of the mesh.

    Sample input:
    -------------
    system_params = {'mesh'  : mesh,
                    'discretization'    : { 'elem_type' : ('CG', 'CG'),
                                            'order' : (2,1),
                                            'bcs': 'lid-driven-cavity'},
                    'dof_ordering'     : {  'split_by_component': True,
                                            'lexicographic': True},
                    'additional'       : {  'lo_fe_precond' :True},
                    }
    """

    # TODO: cleanup the values
    mesh_name2 = {
        2: {
            "unstructured square": "2D/square/square_h_",
            "flow past a cylinder": "2D/flow_past_cyl/flow_past_cyl_h_",
            "pinched channel": "2D/pinched_channel/pinched_channel_h_",
            "airfoil": "2D/airfoil/airfoil_h_",
        },
        3: {
            "pinched channel": "3D/pinched_channel/pinched_channel_h_",
            "split artery": "3D/split_artery/split_artery_h_",
            # "long artery"      : "3D/long_artery/long_artery_h_"
        },
    }
    mesh_ext2 = {}

    def __init__(
        self,
        system_params,
        name_id=0,
        dim=None,
        start_idx=None,
        end_idx=None,
        max_dofs=10000,
    ):
        """Initialize the iterator object."""
        super().__init__(system_params, max_dofs)

        if dim is None:
            raise ValueError("Please provide the expected dimension of the mesh")

        self.dim = dim
        self.start = 0 if start_idx == None else start_idx
        self.end = end_idx

        mesh_name = self.mesh_name2[self.dim]
        mesh_ext = self.mesh_ext2
        import os

        self.pream = os.path.split(__file__)[0] + "/../meshes/"
        import glob

        # print(mesh_name)
        for k, v in mesh_name.items():
            # print(f"{self.pream}{v}*.msh")
            paths = glob.glob(f"{self.pream}{v}*.msh")
            # print(paths)
            exts = [i.split(".")[-2].split("_h_")[1] for i in paths]
            exts = [int(i) for i in exts]
            # print(exts)
            exts = np.sort(exts)
            mesh_ext[k] = [str(i).zfill(2) for i in exts]
        # print('\n\nmesh_ext:\n', mesh_ext)
        # print("\n\nmesh_name:\n", mesh_name)

        self.name = list(mesh_name.keys())[name_id]
        self.file_name = mesh_name[self.name]
        self.exts = mesh_ext[self.name]

    def get_path(self):
        return self.pream + self.file_name + ("%s" % self.exts[self.count]) + ".msh"

    def __repr__(self):
        return "Unstructured %dD Stokes (%s): %s" % (
            self.dim,
            self.file_name,
            self.system_params["discretization"]["bcs"],
        )

    def __iter__(self):
        self.count = self.start
        return self

    def __next__(self):
        end = np.iinfo(np.uint64).max if self.end == None else self.end
        if self.count < min(len(self.exts), end):
            ext = self.exts[self.count]
            path = self.get_path()

            tic = time_ns()
            mesh = Mesh(path)
            # P2/P1: v_x+v_y+p
            # slightly overestimates # DoFs
            pdofs = mesh.coordinates.dat.data_ro.shape[0]
            vdofs = (np.power(pdofs, 1.0 / self.dim) * 2 + 1) ** self.dim
            ndofs_approx = vdofs * self.dim + pdofs
            if self.max_dofs <= ndofs_approx:
                raise StopIteration

            self.system_params["mesh"] = mesh
            stokes = Stokes(self.system_params)
            self.build_time = (time_ns() - tic) / 1e9

            self.count += 1
            stokes.structured = False
            if hasattr(stokes, "lo_fe_sys"):
                stokes.lo_fe_sys.structured = False
            return stokes
        else:
            raise StopIteration
