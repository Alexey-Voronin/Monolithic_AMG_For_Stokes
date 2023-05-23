from firedrake import *
import numpy as np
import scipy.sparse as sp
import networkx as nx
import matplotlib.pyplot as plt

def get_p1_interp(cf_map, neighbors_f):
    nf = len(neighbors_f)
    nc = cf_map.shape[0]

    rows = []; cols = []; data = []
    for c, f in enumerate(cf_map):
        for fi in neighbors_f[f]:
            rows.append(c)
            cols.append(fi)
            data.append(0.5)

        rows.append(c)
        cols.append(f)
        data.append(1)

    return sp.csr_matrix((data, (rows, cols)), shape=(nc, nf)).T

def get_p1_info(mesh, debug=False):
    P1           = FunctionSpace(mesh, "CG", 1)
    test, trial = TestFunction(P1), TrialFunction(P1)
    integrand   = inner(test, trial)
    Mass        = assemble(integrand*dx, bcs=[], mat_type='nest', sub_mat_type='aij')
    Mass        = sp.csr_matrix((Mass.petscmat.getValuesCSR())[::-1])

    #######################################
    vertices = interpolate(SpatialCoordinate(mesh), VectorFunctionSpace(mesh, "CG", 1)).dat.data_ro.copy()
    indices  = Mass.indices
    indptr   = Mass.indptr
    edges    = []
    for i in range(indptr.shape[0]-1):
        for j in range(indptr[i], indptr[i+1]):
            if i == indices[j] :
                continue
            edges.append((i,indices[j]) if i < indices[j] else (indices[j], i))

    edges_unique = np.unique(np.sort(np.array(edges), axis=1), axis=0)

    if debug:
        if mesh.topological_dimension() == 3:
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(vertices[:,0], vertices[:,1], vertices[:,2])
            for i, v in enumerate(vertices):
                ax.text(*tuple(v),str(i), fontsize=15, c='b')

            for i,j in edges:
                ax.plot([vertices[i,0], vertices[j,0]],
                            [vertices[i,1], vertices[j,1]],
                         zs=[vertices[i,2], vertices[j,2]], c='k', linewidth=0.5)
        else:
            fig = plt.figure()
            ax = fig.add_subplot()
            ax.scatter(vertices[:,0], vertices[:,1])
            for i, v in enumerate(vertices):
                ax.text(*tuple(v),str(i), fontsize=15, c='b')

            for i,j in edges:
                ax.plot([vertices[i,0], vertices[j,0]],
                        [vertices[i,1], vertices[j,1]],
                        c='k', linewidth=0.5)

    return P1, vertices, edges_unique

def get_neighbor_lists(edges_unique):
    G = nx.Graph()
    G.add_edges_from(edges_unique)
    nodes = np.max(np.ravel(edges_unique))+1

    return [[n for n in G.neighbors(i)] for i in range(nodes)]


def get_cf_mapping(meshc, meshf, debug=False):
    from firedrake.mg import utils
    from fractions import Fraction
    from firedrake.functionspacedata import entity_dofs_key
    from firedrake.cython import mgimpl as impl

    Vc, coord_c, edges_c = get_p1_info(meshc)
    Vf, coord_f, edges_f = get_p1_info(meshf)

    hierarchyf, levelf = utils.get_level(Vf.ufl_domain())
    hierarchyc, levelc = utils.get_level(Vc.ufl_domain())

    if hierarchyc != hierarchyf:
        raise ValueError("Can't map across hierarchies")

    hierarchy = hierarchyf
    increment = Fraction(1, hierarchyf.refinements_per_level)
    if levelc + increment != levelf:
        raise ValueError("Can't map between level %s and level %s" % (levelc, levelf))

    key = (entity_dofs_key(Vc.finat_element.entity_dofs())
        + entity_dofs_key(Vf.finat_element.entity_dofs())
        + (levelc, levelf))

    coarse_to_fine       = hierarchy.coarse_to_fine_cells[levelc]
    coarse_to_fine_nodes = impl.coarse_to_fine_nodes(Vc, Vf, coarse_to_fine)
    cf_map               = np.zeros((coarse_to_fine_nodes.shape[0],), dtype=int)
    coarse_to_fine_nodes = [np.unique(n) for n in coarse_to_fine_nodes]

    for c_point, potential_f_points in enumerate(coarse_to_fine_nodes):
        diff = coord_c[c_point]-coord_f[potential_f_points]
        norm = np.linalg.norm(diff, axis=1)
        match = np.where(norm == 0)[0]
        assert len(match) == 1, 'something is wrong with finding f_dof'
        cf_map[c_point] = potential_f_points[match]

        if debug:
            if meshc.cell_dimension() == 3:
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.scatter(coord_f[:,0], coord_f[:,1], coord_f[:,2])
                ax.scatter(coord_f[potential_f_points,0],
                            coord_f[potential_f_points,1],
                            coord_f[potential_f_points,2])
                ax.scatter(coord_c[c_point,0], coord_c[c_point,1],
                            coord_c[c_point,2], label='x')
                triplot(meshc, axes=ax,
                        interior_kw=dict(alpha=0.1, linewidth=2),
                        boundary_kw=dict( alpha=0.01, linewidths=[0.5]*4)
                       )
                plt.show()
            else:
                triplot(meshc)
                plt.scatter(coord_f[:,0], coord_f[:,1])
                plt.scatter(coord_f[potential_f_points,0],
                            coord_f[potential_f_points,1])
                plt.scatter(coord_c[c_point,0],
                            coord_c[c_point,1], marker='x')
                plt.show()

    return cf_map

#######################################################
# visualization
def plot_g(edges, coord):
    G = nx.Graph()
    G.add_edges_from(edges)
    pos = {}
    for i, v in enumerate(coord):
        pos[i] = v

    nx.draw(G, coord, with_labels=True)
    plt.show()

def plot_fxn(mesh, u, title=''):
    levels = np.linspace(0, 1, 51)
    contours = tricontourf(u, levels=levels, cmap="inferno")
    plt.colorbar(contours)
    plt.title(title +': u.shape=' + str(u.dat.data.shape))
    plt.show()


def contruct_unstruct_p1_interp(meshc, meshf):
    Vc, coord_c, edges_c = get_p1_info(meshc)
    Vf, coord_f, edges_f = get_p1_info(meshf)

    neighbors_f = get_neighbor_lists(edges_f)
    cf_map      = get_cf_mapping(meshc, meshf)

    return get_p1_interp(cf_map, neighbors_f)
