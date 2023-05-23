import numpy as np
import scipy.sparse as sp
from  firedrake import RectangleMesh, FunctionSpace, VectorFunctionSpace,\
    SpatialCoordinate, interpolate, DirichletBC, Constant
import matplotlib.pyplot as plt
import sympy as sym
import os


def basis_2d(d=2, symbolic=True, Omega=None, component='x'):
    """
    Return all local basis function phi as functions of the
    (X,Y) in a 2D element with d+1 nodes in each dim.
    point distribution is currently 'uniform'
    """
    if Omega is None or symbolic:
        Omega = [[0, 1], [0, 1]]

    # construct polynomial with undetermined coefficients
    x, y = None, None
    if component == 'x':
        y, x = sym.symbols('x y')
    else:
        x, y = sym.symbols('x y')

    h = sym.symbols('h')
    lin_fxns = [x ** i * y ** j for i in range(d) for j in range(d)]
    coef = []
    for i in range(len(lin_fxns)):
        coef.append(sym.symbols('a' + str(len(coef))))
        lin_fxns[i] *= coef[-1]

    lin_fxn = 0
    for f in lin_fxns:
        lin_fxn += f

    #######################################################
    # compute coefficients
    nx = d;
    ny = d
    nodes = nx * ny
    xv = np.linspace(Omega[0][0], Omega[0][1], nx)
    yv = np.linspace(Omega[1][0], Omega[1][1], ny)
    Xv, Yv = np.meshgrid(xv, yv)
    dof_x = np.ravel(Xv)
    dof_y = np.ravel(Yv)

    phis = []
    for i in range(nodes):
        if symbolic:
            phis.append(lin_fxn.subs([(x, h * dof_x[i]), (y, h * dof_y[i])]))
        else:
            phis.append(lin_fxn.subs([(x, dof_x[i]), (y, dof_y[i])]))

    # compute coefficients and plug them into polynomial
    basis = []
    for i in range(nodes):
        eqs = phis.copy()
        eqs[i] = eqs[i] - 1  # set fxn val equal to one
        args = sym.solve(eqs, coef)
        basis.append(sym.factor(lin_fxn.subs(args)))

    return basis


def get_q1isoq2_basis(component='y'):
    """ Q1isoQ2 basis functions over a macro element glued together from
    symbolic basis functions over induvidual element."""
    Lx = Ly = 1
    Omega1 = [[0   ,Lx/2], [0,Ly/2]]
    Omega2 = [[Lx/2,Lx],   [0,Ly/2]]
    Omega3 = [[0   ,Lx/2], [Ly/2,Ly]]
    Omega4 = [[Lx/2,Lx],   [Ly/2,Ly]]
    # keep sam component here and just renumber the basis at the end
    sq1    = basis_2d(d=2, Omega=Omega1, symbolic=False, component='y')
    sq2    = basis_2d(d=2, Omega=Omega2, symbolic=False, component='y')
    sq3    = basis_2d(d=2, Omega=Omega3, symbolic=False, component='y')
    sq4    = basis_2d(d=2, Omega=Omega4, symbolic=False, component='y')

    #corners
    x, y = sym.symbols('x y')
    sq1c =  sym.Piecewise( (sq1[0], (x <= Lx/2) & (y <= Ly/2)), (0, True))
    sq2c =  sym.Piecewise( (sq2[1], (x >= Lx/2) & (y <= Ly/2)), (0, True))
    sq3c =  sym.Piecewise( (sq3[2], (x <= Lx/2) & (y >= Ly/2)), (0, True))
    sq4c =  sym.Piecewise( (sq4[3], (x >= Lx/2) & (y >= Ly/2)), (0, True))

    phi_sq12 =  sym.Piecewise( (sq1[1], (x <= Lx/2) & (y <= Ly/2)),
                               (sq2[0], (x >= Lx/2) & (y <= Ly/2)),
                               (0, True))
    phi_sq24 =  sym.Piecewise( (sq2[3], (y <= Ly/2) & (x >= Lx/2)),
                               (sq4[1], (y >= Ly/2)  & (x >= Lx/2)),
                                (0, True))
    phi_sq13 =  sym.Piecewise( (sq1[2],  (y <= Ly/2) & (x <= Lx/2)),
                               (sq3[0], (y >= Ly/2)& (x <= Lx/2)),
                               (0, True))
    phi_sq34 =  sym.Piecewise( (sq3[3],  (x <= Lx/2)& (y >= Ly/2)),
                               (sq4[2], (x >= Lx/2) & (y >= Ly/2)),
                               (0,  True))
    phi_sqr1234 = sym.Piecewise( (sq1[3], (x <= Lx/2) & (y <= Ly/2)),
                                 (sq2[2], (Lx/2 <= x)  & (y <= Ly/2)),
                                 (sq3[1], (x <= Lx/2) & (y >= Ly/2)),
                                 (sq4[0], (x >= Lx/2) & (y >= Ly/2)))

    if component == 'y':
        basis_v  = [sq1c,        phi_sq12,     sq2c,
                    phi_sq13, phi_sqr1234, phi_sq24,
                    sq3c,        phi_sq34,     sq4c]
    else:
        basis_v  = [sq1c,        phi_sq13,     sq3c,
                    phi_sq12, phi_sqr1234, phi_sq34,
                    sq2c,        phi_sq24,     sq4c]

    return basis_v


def element_mass_matrix(phi, symbolic=True, Omega_e=None, pretty=False, name=''):
    n    = len(phi)

    if os.path.isfile(f'mass_matrix_{name}.npz'):
        A_e = np.load(f'mass_matrix_{name}.npz', allow_pickle=True)['A_e']
        assert A_e.shape == (n, n), 'shape check failed'
        return A_e

    A_e  = sym.zeros(n, n) if symbolic else np.zeros((n, n))
    X, Y = sym.symbols('x y')
    h = None

    if symbolic:
        h = sym.symbols('h')
    else:
        if Omega_e is None:
            Omega_e = [[0, 1], [0, 1]]
        h = Omega_e[0][1] - Omega_e[0][0]

    detJ = 1  # h*h
    for r in range(n):
        for s in range(r, n):
            I = sym.integrate(sym.integrate(phi[r] * phi[s] * detJ,
                                            (X, 0, 1)), (Y, 0, 1))
            A_e[r, s] = sym.nsimplify(I) if pretty else I
            A_e[s, r] = A_e[r, s]


    if not os.path.isfile(f'mass_matrix_{name}.npz'):
        np.savez(f'mass_matrix_{name}.npz', A_e=A_e)

    return A_e


def assemble_mass(vertices, elements, data, phi, symbolic=True, name=''):
    dof_coord, dof_map, internal_nodes = data

    N_n = len(list(set(dof_map.ravel())))
    N_e = len(elements)
    if symbolic:
        A = sym.zeros(N_n)
    else:
        A = np.zeros((N_n, N_n))

    A_e = None
    for e in range(N_e):
        if A_e is None:  # assumes all elements look the same
            # element dimensions
            xvals = vertices[elements[e]][:, 0]
            yvals = vertices[elements[e]][:, 1]
            Omega_e_x = [min(xvals), max(xvals)]
            Omega_e_y = [min(yvals), max(yvals)]
            Omega_e = [Omega_e_x, Omega_e_y]
            # construct element mass matrix
            hx = max(Omega_e_x[1] - Omega_e_x[0], Omega_e_y[1] - Omega_e_x[0])
            A_e = element_mass_matrix(phi, Omega_e, symbolic, name=name) * hx ** 2

        for r in range(len(dof_map[e])):
            for s in range(len(dof_map[e])):
                A[dof_map[e][r], dof_map[e][s]] += A_e[r, s]

    assert abs(np.sum(A) - 1) < 1e-7, 'mass matrix does not sum-up to one'
    return A


def my_mesh(nx, ny, x=[0, 1], y=[0, 1]):
    Lx = x[1];
    Ly = y[1]
    quadrilateral = True

    # pulled out of firedrake's RectangleMesh
    xcoords = np.linspace(0.0, Lx, nx + 1, dtype=np.double)
    ycoords = np.linspace(0.0, Ly, ny + 1, dtype=np.double)
    coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
    # cell vertices
    i, j = np.meshgrid(np.arange(nx, dtype=np.int32), np.arange(ny, dtype=np.int32))
    if quadrilateral:
        cells = [i * (ny + 1) + j, i * (ny + 1) + j + 1,
                 (i + 1) * (ny + 1) + j + 1, (i + 1) * (ny + 1) + j]
        cells = np.asarray(cells).swapaxes(0, 2).reshape(-1, 4)
        # if not quadrilateral:
        #     if diagonal == "left":
        #         idx = [0, 1, 3, 1, 2, 3]
        #     elif diagonal == "right":
        #         idx = [0, 1, 2, 0, 2, 3]
        #     else:
        #         raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
        #     # two cells per cell above...
        #     cells = cells[:, idx].reshape(-1, 3)

    mesh = RectangleMesh(nx, ny, Lx, Ly, quadrilateral=True)
    P = FunctionSpace(mesh, "CG", 1)
    V = VectorFunctionSpace(mesh, "CG", 2)
    p_dof_map = P.cell_node_map().values
    v_dof_map = V.cell_node_map().values

    coord_obj = SpatialCoordinate(mesh)
    vcoord_data = interpolate(coord_obj, VectorFunctionSpace(mesh, "CG", 2))
    v_dof_coord = vcoord_data.dat.data_ro

    pcoord_data = interpolate(coord_obj, VectorFunctionSpace(mesh, "CG", 1))
    p_dof_coord = pcoord_data.dat.data_ro

    bcs = [DirichletBC(FunctionSpace(mesh, "CG", 2), Constant(0.0), (1, 2, 3, 4))]
    bcs_v = set(list(bcs[0].nodes))
    internal_v = list(set(np.linspace(0, len(v_dof_coord) - 1,
                                      len(v_dof_coord), dtype=int)).difference(bcs_v))

    bcs = [DirichletBC(P, Constant(0.0), (1, 2, 3, 4))]
    bcs_p = set(list(bcs[0].nodes))
    internal_p = list(set(np.linspace(0, len(p_dof_coord) - 1,
                                      len(p_dof_coord), dtype=int)).difference(bcs_p))

    return coords, cells, (p_dof_coord, p_dof_map, internal_p), (v_dof_coord, v_dof_map, internal_v)


def element_divergence_matrix2(p_phi_in, v_phi_in, Omega_e, symbolic=True, component='x'):
    n_p = len(p_phi_in)
    n_v = len(v_phi_in)

    if  os.path.isfile('div_Ae.npz'):
        A_e = np.load('div_Ae.npz', allow_pickle=True)['A_e']
        assert A_e.shape == (n_p, n_v), 'element mass matrix has wrong shape'
        return A_e

    # need to ontroduce cordinate system for differentiation
    p_phi = p_phi_in
    v_phi = v_phi_in  # []
    x, y = sym.symbols('x y')

    A_e = sym.zeros(n_p, n_v) if symbolic else np.zeros((n_p, n_v))

    if symbolic:
        h = sym.symbols('h')
    else:
        h = Omega_e[0][1] - Omega_e[0][0]

    for r in range(n_p):
        for s in range(n_v):
            #I = sym.integrate(sym.integrate(p_phi[r] * v_phi[s],
            #                                (x, 0, 1)), (y, 0, 1))
            I = sym.integrate(p_phi[r] * v_phi[s],
                              (x, 0, 1), (y, 0, 1))

            A_e[r, s] = sym.nsimplify(I) if symbolic else I

    if not  os.path.isfile('div_Ae.npz'):
        np.savez('div_Ae', A_e=A_e)

    return A_e

def assemble_divergence2(vertices, elements, p_data, v_data,
                         p_phi, v_phi, symbolic=False, component='x'):
    p_dof_coord, p_dof_map, internal_p = p_data
    v_dof_coord, v_dof_map, internal_v = v_data

    N_p = len(list(set(np.array(p_dof_map).ravel())))
    N_v = len(list(set(np.array(v_dof_map).ravel())))
    N_e = len(elements)

    if symbolic:
        A = sym.zeros(N_p, N_v)
    else:
        A = np.zeros((N_p, N_v))

    Adiv_e = None
    for e in range(N_e):

        if Adiv_e is None:
            xvals = vertices[elements[e]][:, 0]
            yvals = vertices[elements[e]][:, 1]
            Omega_e_x = [min(xvals), max(xvals)]
            Omega_e_y = [min(yvals), max(yvals)]
            Omega_e = [Omega_e_x, Omega_e_y]
            hx = max(Omega_e_x[1] - Omega_e_x[0], Omega_e_y[1] - Omega_e_x[0])
            Adiv_e = element_divergence_matrix2(p_phi, v_phi, Omega_e=Omega_e,
                                                symbolic=False, component=component) * hx ** 2
            Adiv_e = np.array(Adiv_e, dtype=np.float64)
        Ae_tmp = Adiv_e.copy()

        #############################################################
        # Meshing makes necessary to renumber the basis
        # see if the basis needs renumbering
        v_dof_coord_cell = np.zeros((len(v_dof_map[0]), 2))
        for i, dof in enumerate(v_dof_map[e]):
            v_dof_coord_cell[i, :] = v_dof_coord[dof, :]

        renum = np.lexsort((v_dof_coord_cell[:, 1], v_dof_coord_cell[:, 0]))
        e_ndofs = len(v_dof_map[e])
        if not np.allclose(renum, np.linspace(0, e_ndofs - 1, e_ndofs, dtype=int)):
            Ae_tmp = Ae_tmp[:, renum]

        p_dof_coord_cell = np.zeros((len(p_dof_map[0]), 2))
        for i, dof in enumerate(p_dof_map[e]):
            p_dof_coord_cell[i, :] = p_dof_coord[dof, :]

        renum = np.lexsort((p_dof_coord_cell[:, 1], p_dof_coord_cell[:, 0]))
        e_ndofs = len(p_dof_map[e])
        if not np.allclose(renum, np.linspace(0, e_ndofs - 1, e_ndofs, dtype=int)):
            Ae_tmp = Ae_tmp[renum, :]

        ###############################################################
        # Contribute to Global Matrix
        for s in range(len(v_dof_map[e])):
            for r in range(len(p_dof_map[e])):
                A[p_dof_map[e][r], v_dof_map[e][s]] += Ae_tmp[r, s]

    #    A = A[internal_p, :]
    #    A = A[:, internal_p]

    return A


def plot_basis_grid(phis, Omega=[[0, 1], [0, 1]], component=None, basis_name=None):
    from matplotlib import cm
    N = 10
    xv = np.linspace(Omega[0][0], Omega[0][1], N)
    yv = np.linspace(Omega[1][0], Omega[1][1], N)
    Xv, Yv = np.meshgrid(xv, yv)

    xv_t = np.linspace(0, 1.0, N * 2)
    yv_t = np.linspace(0, 1.0, N * 2)
    Xv_t, Yv_t = np.meshgrid(xv_t, yv_t)
    Zv_t = np.zeros_like(Xv_t)

    order = 1
    if len(phis) == 9:
        order = 2
    elif len(phis) == 4:
        order = 1

    fig, axs = plt.subplots(order + 1, order + 1)
    x, y = sym.symbols('x y')
    for k, phi in enumerate(phis, start=0):
        Zv = np.zeros_like(Xv)
        for i in range(Zv.shape[0]):
            for j in range(Zv.shape[1]):
                Zv[i, j] = phi.subs([(x, Xv[i, j]), (y, Yv[i, j])])

        Zv_t *= 0
        start_i = int(Omega[0][0] * 2 * N)
        start_j = int(Omega[1][0] * 2 * N)
        Xv_t[start_j:(start_j + N), start_i:(start_i + N)] = Xv
        Yv_t[start_j:(start_j + N), start_i:(start_i + N)] = Yv
        Zv_t[start_j:(start_j + N), start_i:(start_i + N)] = Zv

        if component == 'y':
            ax = axs[order - int(k / (order + 1)), k % (order + 1)]
        elif component == 'x':
            ax = axs[order - k % (order + 1), int(k / (order + 1))]
        else:
            x = axs[int(k / (order + 1)), k % (order + 1)]

        # Plot the surface.
        surf = ax.contourf(Xv_t, Yv_t, Zv_t, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False, levels=300)

        ax.axhline(y=0.5, color='k')
        ax.axvline(x=0.5, color='k')
        # Customize the z axis.
        ax.set_xlim(0, 1.0)
        ax.set_ylim(0, 1.0)
        ax.set_title('Basis[%d]' % (k + 1))

    for ax in axs.flat:
        ax.set(xlabel='x-label', ylabel='y-label')
    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    if component is not None:
        fig.suptitle('DoF Ordering %s-component: %s' % (component, basis_name), y=1.002)

    plt.tight_layout()
    #if basis_name is not None:
    #    plt.savefig(basis_name + '_' + component + '.pdf')

    plt.show()

"""
if __name__ == '__main__':
    OUTPUT = True

    basis_q1 = basis_2d(d=2, symbolic=False, component='x')
    basis_q2 = basis_2d(d=3, symbolic=False, component='x')
    basis_q1isoq2 = get_q1isoq2_basis(component='x')

    if OUTPUT:
        plot_basis_grid(basis_q1, component='x', basis_name='basis_q1')
        plot_basis_grid(basis_q2, component='x', basis_name='basis_q2')
        plot_basis_grid(basis_q1isoq2, Omega=[[0, 1], [0, 1]], component='x',
                        basis_name='basis_q1isoq2')

    Lx  = Ly  = 1
    NEx = NEy = 16
    out = my_mesh(NEx, NEy, x=[0, Lx], y=[0, Ly])
    coords, cells, p_data, v_data = out

    NEx = NEy = 16
    out = my_mesh(NEx, NEy, x=[0, Lx], y=[0, Ly])
    coords, cells, p_data, v_data = out

    if OUTPUT:
        print('Domain--------------------')
        print('NEx=NEy=', NEx)
        print('Lx=Ly=', Lx)
        print('Matrices--------------------')

    # M_1 is the linear mass matrix on the h/2 mesh.
    M_iso1 = assemble_mass(coords, cells, v_data, basis_q1isoq2, symbolic=False)

    if OUTPUT:
        print('M_iso1.shape=', M_iso1.shape)
        print('sum=%.2f' % np.sum(M_iso1))
        eigvals = np.sort(np.linalg.eigvals(M_iso1))
        print('eigvals=[%.2e, %.2e]' % (eigvals[0], eigvals[-1]))

    coords, cells, p_data, v_data = my_mesh(NEx, NEy, x=[0, Lx], y=[0, Ly])
    M_iso12 = assemble_divergence2(coords, cells, v_data, v_data,
                                   basis_q1isoq2, basis_q2,
                                   symbolic=False, component='x')

    if OUTPUT:
        print('-----')
        print('M_iso12.shape=', M_iso12.shape)
        print('sum(M_iso12)=%.2f' % np.sum(M_iso12))
        eigvals = np.sort(np.abs(np.linalg.eigvals(M_iso12)))
        print('eigvals=[%.2e, %.2e]' % (eigvals[0], eigvals[-1]))

        print('----')
        print('M_iso1_inv:')
        M_iso1_inv = np.linalg.inv(M_iso1)
        # print('sum=', np.sum(M_iso1_inv))
        eig0 = np.sort(np.linalg.eigvals(M_iso1_inv))
        print('eigvals=[%.2e, %.2e]' % (eig0[0], eig0[-1]))

        print('----')
        print('T=M_iso1_inv@M_iso12:')
        T = M_iso1_inv @ M_iso12
        print('sum=', np.sum(T))
        eig0 = np.sort(np.abs(np.linalg.eigvals(T)))
        print('eigvals=[%.2e, %.2e]' % (eig0[0], eig0[-1]))

        plt.spy(T.round(10))
        plt.title('T')
        plt.show()
"""
