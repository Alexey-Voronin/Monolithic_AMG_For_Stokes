import numpy as np
from firedrake import MeshHierarchy, RectangleMesh
from firedrake.mesh import Mesh, MeshGeometry
from firedrake.utility_meshes import UnitIntervalMesh

from sysmg.util.decorators import timeit
from .bary_meshes import bary_reg_ref_mesh
from .other_meshes import RectangleMesh_MyCoord


class MeshWrapper(object):
    """SysMG's Mesh generating function.

    MeshWrapper stores some information about commonly used meshes to make system
    prototyping faster.
    """

    @timeit("setup:system:mesh:")
    def __init__(self, params):
        """Initialize MeshWrapper object."""
        if params == None:  # allows for construction of empty System objects
            return

        self.periodic = False  # check comes later
        self.mesh_hierarchy = False
        # paramas is a strong containing path to mesh
        if type(params) is str:
            self.mesh = Mesh(params)
            self.meshes = [self.mesh]
            self.quadrilateral = self.mesh.topology.ufl_cell().cellname() \
                                 == "quadrilateral"
            self.boundary_marker = {'left': 0, 'right': 1,
                                    'bottom': 2, 'top': 2}

        elif isinstance(params, MeshGeometry) or \
                isinstance(params, dict) and 'bary' in params:
            # paramas is a firedrake mesh or { 'mesh' : mesh, 'bary' : bool}
            if isinstance(params, MeshGeometry):
                self.mesh = params
                self.bary = False
            else:
                self.mesh = params['mesh']
                self.bary = params['bary']

            try:
                self.quadrilateral = self.mesh.topology.ufl_cell().cellname() \
                                     == "quadrilateral"
            except:
                # might fail with bary meshes
                self.quadrilateral = False

            if self.bary:
                assert not self.quadrilateral, \
                    'barycentric refinement is for P-elements.'
                self.mesh_orig = self.mesh
                self.meshes = [bary_reg_ref_mesh(self.mesh)]
                self.mesh = self.meshes[-1]
            else:
                self.meshes = [self.mesh]

            self.dim = self.mesh.topological.ufl_cell() \
                .geometric_dimension()
            self.markers = self._get_boundary_markers()
            n_markers = len(self.markers)

            if self.dim == 1:
                self.boundary_marker = {'left': self.markers[0],
                                        'right': self.markers[1],
                                        }
            elif self.dim == 2:



                if n_markers == 4:
                    self.boundary_marker = {'left': self.markers[0],
                                            'right': self.markers[1],
                                            'bottom': self.markers[2],
                                            'top': self.markers[3]}
                elif n_markers == 3:
                    # Airfoil hack
                    if np.min(self.markers) == 1:
                        self.boundary_marker = {'inflow': 1,
                                            'sides': 3,
                                            'outflow': 2}
                    else:
                        self.boundary_marker = {'inflow': 0,
                                            'sides': 2,
                                            'outflow': 1}
                else:
                    raise ValueError('new mesh markers?')
            elif self.dim == 3:

                if n_markers == 6:
                    self.boundary_marker = {'top': 6, 'bottom': 5,
                                            'left': 1, 'right': 2,
                                            'front': 3, 'back': 4}
                elif n_markers == 3:
                    self.boundary_marker = {'inflow': 1,
                                            'sides': 0,
                                            'outflow': 2}
                else:
                    raise ValueError('new mesh markers?')
            else:
                raise ValueError('dim not in (1,2,3).')

        else:
            assert isinstance(params, dict), """MeshWrapper input is not valid.
                    Valid inputs path to mesh (str), firedrake mesh object,
                    or dictionary ({'NE' : (2,2), 'L' : (1.,1.)})."""

            self.mesh_hierarchy = params.get('mesh_hierarchy', False)
            self.mesh_type = params.get('mesh_type', None)
            self.quadrilateral = params.get('quadrilateral', False)
            prescribed_vertices = params.get('V', None)
            NE = self.NE = params.get('NE', None)
            L = self.L = params.get('L', None)
            dim = self.dim = len(NE)
            assert dim == len(L), 'MeshWrapper: len(NE) does not match len(L).'

            if not self.mesh_type:
                if dim == 1:
                    self._setup_1D_interval_meshes()
                    self.markers = self._get_boundary_markers()
                elif dim == 2:
                    self._setup_2D_rectangular_meshes()
                    self.markers = self._get_boundary_markers()
                elif dim == 3:
                    self.mesh = BoxMesh(*NE, *L)
                    self.meshes = [self.mesh]
                    self.markers = self._get_boundary_markers()
                    n_markers = len(self.markers)

                    if n_markers == 6:
                        self.boundary_marker = {'top': 6, 'bottom': 5,
                                                'left': 1, 'right': 2,
                                                'front': 3, 'back': 4}
                    elif n_markers == 3:
                        self.boundary_marker = {'sides': 0,
                                                'inflow': 1,
                                                'outflow': 2,
                                                }
                    else:
                        raise ValueError("""You are probably passing in a new
                                            mesh. Make sure that boundary
                                            markers make sense.""")
                elif dim not in (2, 3):
                    raise ValueError("MeshWrapper: dim=%d is not supported." % dim)
            elif self.mesh_type == 'periodic':
                self.periodic = True
                assert dim == 2
                self._setup_2D_periodic_meshes()
                self.markers = self._get_boundary_markers()
                self.L = (1, 1)  # hardcoded mesh lengths
            elif self.mesh_type == 'bfs':  # back-ward facing step
                assert dim == 2
                # poor limitation. To be fixed
                assert self.NE[0] == self.NE[1], """Number of elements in
                            each directions must be the same."""
                self._setup_2D_bfs_meshes()
                self.markers = self._get_boundary_markers()
            elif self.mesh_type == 'non-uniform':
                assert prescribed_vertices is not None, 'provide vertices, V'
                assert dim == 2
                self._setup_2D_non_uniform_meshes(prescribed_vertices)
                self.markers = self._get_boundary_markers()
            else:
                raise ValueError('mesh_type=%s is not defined.' % self.mesh_type)

            assert len(self.markers) == len(set(self.boundary_marker.values())), \
                """Number of mesh markers (%d) does not match the
                    boundary marker (%d).%s""" % (len(self.markers),
                                                  len(self.boundary_marker.values()), str(self.boundary_marker))

    def _get_boundary_markers(self):
        self.mesh.coordinates  # triggers marker construction
        return self.mesh.exterior_facets.unique_markers

    def _get_refinement_info(self):
        """Get number of refinment levels from.

        TODO: Update.
        """
        NE = self.NE
        if self.periodic and self.dim == 2:
            assert NE[0] == NE[1], 'For periodic meshes NEx needs to equal to NEy.'

        ref_lvls = 1
        if self.mesh_hierarchy:
            if not np.alltrue(np.array([ne % 2 for ne in NE]) == 0):
                raise ValueError("""MeshWrapper->_get_refinement_info: Need even #
                                    of elements in each direction.""")
            ref_lvls = int(np.min(np.log2(NE)))

        NE_hier = [[ne] for ne in NE]
        for _ in range(ref_lvls - 1):
            for d in range(self.dim):
                NE_hier[d].append(NE_hier[d][-1] // 2)

        self.NE_hier = [ne[::-1] for ne in NE_hier]

        return ref_lvls - 1

    def _setup_1D_interval_meshes(self):
        ref_lvls = self._get_refinement_info()
        NE = self.NE_hier[0][0]

        mesh_c = UnitIntervalMesh(NE)

        self.boundary_marker = {'left': 1, 'right': 2}
        self.meshes = MeshHierarchy(mesh_c, ref_lvls, reorder=False).meshes
        self.mesh = self.meshes[-1]

    def _setup_2D_rectangular_meshes(self):
        ref_lvls = self._get_refinement_info()
        NE_hier = self.NE_hier

        mesh_c = RectangleMesh(NE_hier[0][0], NE_hier[1][0],
                               self.L[0], self.L[1],
                               quadrilateral=self.quadrilateral)

        self.boundary_marker = {'left': 1, 'right': 2, 'bottom': 3, 'top': 4}
        self.meshes = MeshHierarchy(mesh_c, ref_lvls, reorder=False).meshes
        self.mesh = self.meshes[-1]

    def _setup_2D_bfs_meshes(self):
        self.meshes = []
        for nex in self.NE_hier[0]:
            self.meshes.append(RectangleMesh_MyCoord(nex, nex, *self.L,
                                                     quadrilateral=self.quadrilateral,
                                                     prescribed_coord=None, shape="L"))
        self.mesh = self.meshes[-1]
        if not self.quadrilateral:
            raise Exception("""_setup_2D_bfs_meshes: Triangular Mesh looks
                                a bit off at the corner. Fix it.""")
        self.boundary_marker = {'left': 1, 'right': 3, 'bottom': 2, 'top': 2}

    def _setup_2D_non_uniform_meshes(self, prescribed_vertices):
        self.meshes = [RectangleMesh_MyCoord(self.NE_hier[0][0],
                                             self.NE_hier[1][0],
                                             *self.L,
                                             prescribed_coord=prescribed_vertices,
                                             quadrilateral=self.quadrilateral)
                       ]
        self.mesh = self.meshes[-1]
        self.boundary_marker = {'left': 1, 'right': 2, 'bottom': 3, 'top': 4}


if __name__ == "__main__":
    from firedrake import triplot, BoxMesh
    import matplotlib.pyplot as plt

    # case 1
    params = '/Users/lexey/UIUC/firedrake/break_MG/sysmg/systems/meshes/mesh_files/hole_mesh_tri_cf_p05.msh'
    mesh = MeshWrapper(params)
    # case 2
    params = BoxMesh(16, 16, 16, 1, 1, 1)
    mesh = MeshWrapper(params)
    # case 3
    params = {'NE': (16, 8), 'L': (1, 1), 'mesh_hierarchy': True}
    mesh = MeshWrapper(params)
    for m in mesh.meshes:
        triplot(m);
        plt.show()

    # case 4
    params = {'NE': (16, 8, 8), 'L': (2, 1, 1)}
    mesh = MeshWrapper(params)
    # case 6: Backward facing step
    params = {'NE': (8, 8), 'L': (2, 2), 'quadrilateral': True, 'mesh_type': 'bfs', 'mesh_hierarchy': True}
    mesh = MeshWrapper(params)
    # case 7: Non-uniform Mesh
    from firedrake import RectangleMesh, interpolate, SpatialCoordinate, VectorFunctionSpace

    m = RectangleMesh(2, 2, 1, 1, quadrilateral=True)
    V = interpolate(SpatialCoordinate(m), VectorFunctionSpace(m, "CG", 3))
    coord = V.dat.data_ro.copy()
    dof = coord.T.copy()
    dof = dof.round(12)
    ind = np.lexsort((dof[0], dof[1]))
    coord[:, 0] = coord[ind, 0]
    coord[:, 1] = coord[ind, 1]
    # Coordinates HAVE to be sorted
    params = {'NE': (6, 6), 'L': (1, 1), 'quadrilateral': True,
              'mesh_type': 'non-uniform', 'V': coord}
    mesh = MeshWrapper(params)
    triplot(mesh.mesh);
    plt.legend();
    plt.show()
