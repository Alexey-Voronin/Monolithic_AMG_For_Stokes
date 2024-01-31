import numpy as np
from firedrake import mesh
from firedrake.cython import dmcommon
from pyop2.mpi import COMM_WORLD


# Added argument to to specify vertex locations prescribed_coord needs to be lex-sorted.
# Warning: If the mesh connectivity comes out funky, it's because the coordinates
# in each col/ may not have exactly the same x coordinate (round-off error)
# Fix: just round all coardinates to some reasonable value.


def RectangleMesh_MyCoord(
    nx,
    ny,
    Lx,
    Ly,
    quadrilateral=False,
    reorder=None,
    shape="rectangle",
    diagonal="left",
    distribution_parameters=None,
    comm=COMM_WORLD,
    prescribed_coord=None,
):
    """Function adopted from firedrake source code.
    It allows one to specify the coordinates of the vertices of the mesh.
    In addition, one can generate flow over a step (or L-shaped domain) by
    specifying shape='L'.

    For all other arguments, see firedrake.mesh.RectangleMesh.
    """
    for n in (nx, ny):
        if n <= 0 or n % 1:
            raise ValueError("Number of cells must be a positive integer")

    xcoords = np.linspace(0.0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0.0, Ly, ny + 1, dtype=np.double)
    coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
    # cell vertices
    i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))
    if not quadrilateral and diagonal == "crossed":
        dx = Lx * 0.5 / nx
        dy = Ly * 0.5 / ny
        xs = np.linspace(dx, Lx - dx, nx, dtype=np.double)
        ys = np.linspace(dy, Ly - dy, ny, dtype=np.double)
        extra = np.asarray(np.meshgrid(xs, ys)).swapaxes(0, 2).reshape(-1, 2)
        coords = np.vstack([coords, extra])
        #
        # 2-----3
        # | \ / |
        # |  4  |
        # | / \ |
        # 0-----1
        cells = [
            i * (ny + 1) + j,
            i * (ny + 1) + j + 1,
            (i + 1) * (ny + 1) + j,
            (i + 1) * (ny + 1) + j + 1,
            (nx + 1) * (ny + 1) + i * ny + j,
        ]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 5)
        idx = [0, 1, 4, 0, 2, 4, 2, 3, 4, 3, 1, 4]
        cells = cells[:, idx].reshape(-1, 3)

    else:
        cells = [
            i * (ny + 1) + j,
            i * (ny + 1) + j + 1,
            (i + 1) * (ny + 1) + j + 1,
            (i + 1) * (ny + 1) + j,
        ]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
        if not quadrilateral:
            if diagonal == "left":
                idx = [0, 1, 3, 1, 2, 3]
            elif diagonal == "right":
                idx = [0, 1, 2, 0, 2, 3]
            else:
                raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
            # two cells per cell above...
            cells = cells[:, idx].reshape(-1, 3)

        if shape == "L":
            keep_nodes = []
            rm_nodes = []
            for i in range(coords.shape[0]):
                if (
                    coords[i, 0] >= 1
                    and coords[i, 1] >= 1
                    or coords[i, 0] >= 1
                    or coords[i, 1] >= 1
                ):
                    keep_nodes.append(i)

            keep_nodes = np.array(keep_nodes)
            keep_mask = np.zeros((coords.shape[0],), dtype=bool)
            keep_mask[keep_nodes] = True

            rm_nodes = np.linspace(0, coords.shape[0] - 1, coords.shape[0], dtype=int)
            rm_nodes = rm_nodes[~keep_mask]

            keep_cells_idx = []
            for i, c in enumerate(cells):
                if keep_mask[c].all():
                    keep_cells_idx.append(i)

            keep_cells = cells[keep_cells_idx]

            if not quadrilateral:
                # remove outer cell at the corner
                # only an issue with triangular elements
                center = np.array([0.49 * Lx, 0.49 * Ly])
                idx = -1
                dist = 2
                for i, c in enumerate(keep_cells):
                    dist_c = np.linalg.norm(coords[c, :] - center)
                    if dist_c < dist:
                        dist = dist_c
                        idx = i
                keep_cells = np.delete(keep_cells, idx, axis=0)

            dof_map = dict()
            for c, node in enumerate(keep_nodes):
                dof_map[node] = c

            coords = coords[keep_nodes, :]
            for c in keep_cells:
                for i in range(len(c)):
                    c[i] = dof_map[c[i]]
            cells = keep_cells

    from firedrake.mesh import plex_from_cell_list
    from pyop2.mpi import dup_comm

    comm = dup_comm(COMM_WORLD)
    if prescribed_coord is None:
        plex = plex_from_cell_list(2, cells, coords, comm)
    else:
        plex = plex_from_cell_list(2, cells, prescribed_coord, comm)

    # mark boundary facets
    plex.createLabel(dmcommon.FACE_SETS_LABEL)
    plex.markBoundaryFaces("boundary_faces")
    coords = plex.getCoordinates()
    coord_sec = plex.getCoordinateSection()
    if plex.getStratumSize("boundary_faces", 1) > 0:
        boundary_faces = plex.getStratumIS("boundary_faces", 1).getIndices()
        xtol = Lx / (2 * nx)
        ytol = Ly / (2 * ny)

        if shape == "L":
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if abs(face_coords[0]) < xtol and abs(face_coords[2]) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 0)
                if abs(face_coords[0] - Lx) < xtol and abs(face_coords[2] - Lx) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
                if abs(face_coords[0] - 1) < xtol and abs(face_coords[2] - 1) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)

                if abs(face_coords[1]) < ytol and abs(face_coords[3]) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
                if abs(face_coords[1] - Ly) < ytol and abs(face_coords[3] - Ly) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
                if abs(face_coords[1] - 1) < ytol and abs(face_coords[3] - 1) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
        else:
            for face in boundary_faces:
                face_coords = plex.vecGetClosure(coord_sec, coords, face)
                if abs(face_coords[0]) < xtol and abs(face_coords[2]) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 1)
                if abs(face_coords[0] - Lx) < xtol and abs(face_coords[2] - Lx) < xtol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 2)
                if abs(face_coords[1]) < ytol and abs(face_coords[3]) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 3)
                if abs(face_coords[1] - Ly) < ytol and abs(face_coords[3] - Ly) < ytol:
                    plex.setLabelValue(dmcommon.FACE_SETS_LABEL, face, 4)

    return mesh.Mesh(
        plex, reorder=reorder, distribution_parameters=distribution_parameters
    )


def get_entities(mesh, save=None):
    plex = mesh.topology_dm
    #     print('getDepthStratum:')
    #     for i, name in zip(range(3), ['vertex', 'edge', 'cell']):
    #         print('\t%s\t%d: %s' % (name, i, str(plex.getDepthStratum(i))))
    #     print('\n')
    #     print('getHeightStratum:')
    #     for i, name in zip(range(3), ['cell', 'edge', 'vertex']):
    #         print('\t%s\t%d: %s' % (name, i, str(plex.getHeightStratum(i))))

    dim = mesh.cell_dimension()
    coord_sec = plex.getCoordinateSection()
    coordinates = plex.getCoordinates()
    vStart, vEnd = plex.getDepthStratum(0)

    vertices = np.zeros((vEnd - vStart, dim))
    for i, v in enumerate(range(vStart, vEnd)):
        vertices[i, :] = plex.vecGetClosure(coord_sec, coordinates, v)

    # print(vertices)

    # 3D is broken
    #     #cell_id = 1 if dim == 2 else 2
    #     if dim == 2:
    #         eStart, eEnd = plex.getDepthStratum(1)
    #         edges = np.zeros((eEnd-eStart, 2))
    #         for i, v in enumerate(range(eStart, eEnd)):
    #             out =  plex.getCone(v)-eStart
    #             edges[i,:] = out + vertices.shape[0]
    #     else:
    #         eStart, eEnd = plex.getDepthStratum(2)
    #         edges = np.zeros((eEnd-eStart, 2))
    #         for i, v in enumerate(range(eStart, eEnd)):
    #             #out =  plex.getCone(v)-eStart
    #             #edges[i,:] = out + vertices.shape[0]
    #             print(plex.getTransitiveClosure(v)[:])

    #     print(edges)

    cell_id = 2 if dim == 2 else 3
    cStart, cEnd = plex.getDepthStratum(cell_id)
    cells = np.zeros((cEnd - cStart, dim + 1))
    for i, v in enumerate(range(cStart, cEnd)):
        cells[i, :] = plex.getTransitiveClosure(v)[0][-(dim + 1) :] - cEnd

    if save is not None:
        np.savez(save, vertices=vertices, cells=cells)

    return vertices, cells
