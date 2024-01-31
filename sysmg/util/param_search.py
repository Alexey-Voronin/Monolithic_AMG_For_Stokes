import itertools

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

mpl.rcParams.update(mpl.rcParamsDefault)


def fgmres_solver(problem, mg, tol, maxiter):
    """
    Solve the problem using FGMRES preconditioned with the multigrid solver.

    Parameters
    ----------
    problem : sysmg.systems.stokes.StokesSystem
        The problem to solve.
    mg : pyamg.multilevel_solver
        The multigrid solver.
    tol : float
        The tolerance for the solver.
    maxiter : int
        The maximum number of iterations for the solver.
    """

    # System
    A0 = problem.A_bmat.tocsr()
    b = problem.b
    x0 = np.zeros_like(b)
    # Solver
    x, resids = mg.solve(
        b,
        x0,
        A=A0,
        maxiter=maxiter,
        tol=tol / np.linalg.norm(b),
        cycle_type="V",
        accel={"module": ("pyamg", "fgmres"), "resid": "abs"},
    )

    norm_resid = resids / resids[0]
    norm_resid = norm_resid[np.where(norm_resid >= tol)]
    return norm_resid


def geom_mean(iters):
    scaled = (iters[1:] / iters[:-1])[-5:]
    return scaled.prod() ** (1.0 / float(len(scaled)))


def parameter_scan(
    sys_iterator, mg_fxn, mg_params, setter, param_search, tol=1e-8, max_iter=50
):
    """Generalizes parameter search functionality

    tau_u_range    = np.linspace(0.5, 2., 3)
    tau_p_range    = np.linspace(0.5, 2., 3)
    omega_range  = np.linspace(0.5, 1., 2)
    param_search = {'tau'   : (tau_u_range, tau_p_range),
                    'omega' : (omega_range,), # TUPLE!
                   }

    def setter(mg, params):
        # the variables are in the same order as
        # in the input param_search

        tau_u, tau_p, omega = params
        mg.wrapper.set_tau((tau_u, tau_p))
        mg.wrapper.outer_relaxation.set_omega(omega)
    """

    param_names = list(param_search.keys())
    name_to_range = dict()
    param_idxs = []
    param_arrs = []
    for k in param_names:
        ids = []
        name_to_range[k] = []
        for vi in param_search[k]:
            ids.append(np.array(range(len(vi))))
            param_arrs.append(vi)
            name_to_range[k].append(vi)
        param_idxs += ids

    np.save("param_ranges.npy", name_to_range)

    def get_params(param_idxs):
        return tuple([arr[i] for i, arr in zip(param_idxs, param_arrs)])

    # 'main.log' output
    # figure out the length of each line for formatting
    results = {}
    sv_max = 0
    for idx in itertools.product(*param_idxs):
        args = get_params(idx)
        results[args] = {}
        sv_max = max(sv_max, len(str(args)))
    sv_max += 1
    sn = str(param_names)
    sn_len = len(sn)

    # assemble the sysstem
    total_problems = 0
    for problem in sys_iterator:
        total_problems += 1
        # assembel the solver
        mg = mg_fxn(problem, mg_params)
        f = open("hierarchy.log", "a")
        f.write(mg.__repr__())
        f.close()

        arr_shape = tuple([len(p) for p in param_arrs])
        iters_data = np.zeros(arr_shape)
        cf_data = np.zeros(arr_shape)
        # iterate over all parameter combinations
        for idx in itertools.product(*param_idxs):
            args = get_params(idx)
            setter(mg, args)
            # collect data
            resids = fgmres_solver(problem, mg, tol, max_iter)
            # output data
            ndofs = problem.A_bmat.tocsr().shape[0]
            results[args][ndofs] = resids
            cf = geom_mean(resids)
            sv = str(args)

            # save to array
            iters_data[idx] = len(resids)
            cf_data[idx] = cf
            # write to disc
            f = open("main.log", "a")
            f.write(
                f"ndofs={ndofs:10}: {sn:{sn_len}}={sv:{sv_max}} | resids[{len(resids):3}]={resids[-1]:2.2e} cf={cf:2.3f}\n"
            )
            f.close()

        np.save("iters_data_%d.npy" % total_problems, iters_data)
        np.save("cf_data_%d.npy" % total_problems, cf_data)

    if total_problems > 1:
        f = open("main_sorted.log", "a")
        for args, resids in results.items():
            sv = str(args)
            for ndofs, resid in resids.items():
                cf = geom_mean(resid)
                f.write(
                    f"ndofs={ndofs:10}: {sn:{sn_len}}={sv:{sv_max}} | resids[{len(resid):3}]={resid[-1]:2.2e} cf={cf:2.3f}\n"
                )
        f.close()


def parameter_scan_old(sys_iterator, mg_fxn, mg_params, param_search):
    """Generalizes parameter search functionality
    def set_tau(amg, tau):
        amg.wrapper.set_tau(tau)

    def set_omega(amg, omega):
        amg.wrapper.outer_relaxation.set_omega(omega)

    tau_range    = np.linspace(0.5, 2., 3)
    omega_range  = np.linspace(0.5, 1., 2)
    param_search = {'tau'   : {'range' : (tau_range, tau_range),
                               'setter' : set_tau},
                    'omega' : {'range' : (omega_range),
                               'setter' : set_omega}
                   }
    """

    param_names = list(param_search.keys())
    param_ranges = []
    param_setters = []
    for k in param_names:
        v = param_search[k]["range"]
        s = param_search[k]["setter"]
        param_ranges.append(list(itertools.product(*v)) if type(v) == tuple else v)
        param_setters.append(s)

    # output crap
    results = {}
    sv_max = 0
    for args in itertools.product(*param_ranges):
        results[args] = {}
        sv_max = max(sv_max, len(str(args)))
    sv_max += 1
    sn = str(param_names)
    sn_len = len(sn)

    # assemble the sysstem
    total_problems = 0
    for problem in sys_iterator:
        total_problems += 1
        # assembel the solver
        mg = mg_fxn(problem, mg_params)
        f = open("hierarchy.log", "a")
        f.write(mg.__repr__())
        f.close()

        # iterate over all parameter combinations
        for args in itertools.product(*param_ranges):
            # update parameters
            for setter, arg in zip(param_setters, args):
                setter(mg, arg)
            # collect data
            resids = fgmres_solver(problem, mg, tol)
            # output data
            ndofs = problem.A.shape[0]
            results[args][ndofs] = resids
            cf = geom_mean(resids)
            sv = str(args)
            # write to disc
            f = open("main.log", "a")
            f.write(
                f"ndofs={ndofs:10}: {sn:{sn_len}}={sv:{sv_max}} | resids[{len(resids):3}]={resids[-1]:2.2e} cf={cf:2.3f}\n"
            )
            f.close()

    if total_problems > 1:
        f = open("main_sorted.log", "a")
        for args, resids in results.items():
            sv = str(args)
            for ndofs, resid in resids.items():
                cf = geom_mean(resid)
                f.write(
                    f"ndofs={ndofs:10}: {sn:{sn_len}}={sv:{sv_max}} | resids[{len(resid):3}]={resid[-1]:2.2e} cf={cf:2.3f}\n"
                )
        f.close()


def param_search(
    fxn,
    x0,
    args=(),
    callback=None,
    bounds=None,
    tol=1e-8,
    options={"maxiter": 50, "disp": False},
):
    """
    Automatic parameter optimization.

    Arguments:
        fxn: objective function that takes x0 as input and outputs a scalar value.
        x0: array with parameters being optimized
        args: additional/optional parameters fed to fxn

    Returns:
        test_param: parameters evaluated
        test_obj_vals: respective objective functions
        dict: output of the minimization algorithm

    Example:

    def fun(x, geom_mean):

        tau_u, tau_p, omega = x
        amg.wrapper.outer_relaxation.set_omega(omega)
        amg.wrapper.set_tau((tau_u, tau_p,))

        MAX_ITER = 30; TOL = 1e-6
        b        = stokes.b
        x0       = np.zeros_like(b)
        out      = amg.solve(b, x0, A=stokes.A,
                          maxiter=MAX_ITER, tol=TOL,
                          cycle_type='V',
                          accel={'module': ('pyamg', 'fgmres'),
                                              'resid': 'abs'},
                          )
        _, resid = out[:2]
        cf       = geom_mean(resid)

        test_param.append(x)
        test_obj_vals.append(cf)

        return cf

    x0   = np.ones((3))
    bnds = [(0.1, 3.0) for _ in range(len(x0))]
    out  = param_search(fun, x0, args=(geom_mean,),
                         bounds=bnds, tol=1e-8)
    """

    test_param = []
    test_obj_vals = []

    if not callback:
        output_template = "iter={}:\t"
        for _ in range(len(x0)):
            output_template += "{}, "
        output_template += "\tf(x)={}"

        def cb(xk):
            f = open("main.log", "a")
            out = output_template.format(
                "%3d" % len(test_param),
                *tuple(["%2.4f" % i for i in xk]),
                ("%2.5f" % test_obj_vals[-1]),
            )
            out += "\n"
            f.write(out)
            f.close()

        callback = cb

    def wrapped_fun(x, *args):
        test_param.append(x)
        test_obj_vals.append(fxn(x, *args))
        return test_obj_vals[-1]

    out = minimize(
        wrapped_fun,
        x0,
        args=args,
        method=None,
        jac=None,
        hess=None,
        hessp=None,
        bounds=bounds,
        constraints=(),
        tol=tol,
        callback=callback,
        options=options,
    )

    return np.array(test_param), np.array(test_obj_vals), out


def plot_min_results(
    xvec, fval, labels=["$\\tau_u$", "$\\tau_p$", "$\\omega$"], title="", filename="tmp"
):
    lines = []
    fig, ax = plt.subplots()

    # Rho values
    ax.plot(fval, linestyle="-.", c="k")
    ax.set_ylabel("$\\rho$", fontsize=20)
    ax.set_xlabel("iteration \#", fontsize=20)
    # lines+=line
    ax.annotate(
        str(round(fval[-1], 3)), (len(fval) * 0.93, fval[-1] * 1.05), fontsize=13
    )

    # Parameters
    ax = ax.twinx()
    for i in range(xvec.shape[1]):
        line = ax.plot(xvec[:, i], label=labels[i])
        lines += line
    ax.set_ylabel("parameter values", fontsize=20)
    ax.grid("on")

    # Title
    title += ": p_opt=("
    for i in range(xvec.shape[1] - 1):
        title += "%2.3f," % round(xvec[-1, i], 3)
    title += "%2.3f)" % round(xvec[-1, -1], 3)
    ax.set_title(title, fontsize=20)
    # Legend
    labs = [line.get_label() for line in lines]
    lgd = ax.legend(
        lines, labs, loc="center left", bbox_to_anchor=(1.2, 0.5), fontsize=20
    )
    # Save
    plt.savefig(filename + ".pdf", bbox_extra_artists=(lgd,), bbox_inches="tight")
