import gc
import json
from time import time_ns

import numpy as np

from sysmg import StokesMG, BlockDiagMG


def collect_conv_data(
    stokes_iter,
    MG_PARAMS,
    TOL=1e-12,
    MAX_ITER=80,
    solver_type={"module": ("pyamg", "fgmres"), "resid": "abs"},
    rerun=None,
    cycle_type="V",
    save_solution=False,
):
    """
    Collect convergence data.

    Args:
        stokes_iter: sysmg.systems.stokes_iterators
            An iterators that for Stokes systems that
            produces increasinly larger problems.

        MG_PARAMS: list
            Contains a list of MG setup parameters.


    Returns:
        See code for details.

        conv_data = {'resid_hist' : RESID_HIST,
                  'stokes' : stokes, 'nDOFs' : nDOFs,
                  'mg_lvls' : MG_LVLs, 'amg_params' : AMG_PARAM}

    Notes:
        Work in progress..
    """

    RESID_HIST = {}
    MG_PARAM = {}
    MG_LVLs = {}

    mg_hier_json = {}
    solve_timings_json = {}
    mg_setup_timings_json = {}
    system_setup_timings_json = {}
    patch_info_json = {}
    print(stokes_iter)  # Log-file output
    first_write = True  # dump amg parameters only for one problem size..
    #############################################
    # setup system
    for stokes in stokes_iter:
        A0_csr = stokes.A_bmat.tocsr()
        # print(A0_csr.shape)
        # continue
        #########################################
        # MG SET-UP
        tic = time_ns()
        precond_type = MG_PARAMS.get("type", None)
        if precond_type.lower() == "monolithic":
            amg = StokesMG(stokes, MG_PARAMS, keep=False)
        elif precond_type.lower() in ["uzawa", "physics"]:
            amg = BlockDiagMG(stokes, MG_PARAMS)
        else:
            raise ValueError(
                "Parameter list needs to include the following key/values: {'type': 'monolithic' or 'block-diagonal'}"
            )
        t_amg = (time_ns() - tic) / 1e9

        b0 = stokes.b
        stokes_setup_time = stokes.get_setup_timings().to_json()
        del stokes.A_bmat._bmat
        if hasattr(stokes, "lo_fe_sys"):
            del stokes.lo_fe_sys.A_bmat._bmat
        # del stokes.lo_fe_sys.dof_coord
        # del stokes.mesh
        # del stokes.lo_fe_sys.mesh
        # del stokes.meshes
        # del stokes.lo_fe_sys.meshes
        if not save_solution:
            del stokes.problems
            del stokes
        gc.collect()

        timing_tables = []
        up = None
        resid = None
        for _ in range(rerun):
            #########################################
            # Solve Phase
            b = b0.copy()

            up0 = np.zeros_like(b)
            up, resid = amg.solve(
                b,
                up0,
                A=A0_csr,
                maxiter=MAX_ITER,
                tol=TOL,
                cycle_type=cycle_type,
                accel=solver_type,
            )

            timing_tables.append(amg.get_solution_timings())
            amg.reset_timers(reset=["solution:solver"])

        if save_solution:
            stokes.save_solution(up, "solution")

        ###########################################
        # Save data for later.
        # (overwrite file each system size in case the job
        # is ended prematurely)
        mg_id = len(up)
        mg_setup_timings_json[mg_id] = amg.get_setup_timings().to_json()
        system_setup_timings_json[mg_id] = stokes_setup_time
        amg.reset_timers(reset=["setup:solver"])
        # stokes.reset_timers()
        json.dump(mg_setup_timings_json, open("mg_setup_timings.json", "w"), indent=2)
        json.dump(
            system_setup_timings_json, open("stokes_setup_timings.json", "w"), indent=2
        )
        ###########################################
        # Save solution phase timings
        fastest_id = np.argmin([t["mg:solve"].sum() for t in timing_tables])
        fastest_slv = timing_tables[fastest_id]
        solve_timings_json[len(up)] = fastest_slv.to_json()
        json.dump(solve_timings_json, open("solve_timings.json", "w"), indent=2)
        ###########################################
        # Hierarchy Information
        MG_PARAM[mg_id] = MG_PARAMS
        MG_LVLs[mg_id] = len(amg.ml.levels)
        mg_hier_json[len(up)] = amg.to_pandas().to_json()
        json.dump(mg_hier_json, open("hierarchy_info.json", "w"), indent=2)
        ###########################################
        # convergence histories
        RESID_HIST[mg_id] = resid / resid[0]
        np.save("resid_hist.npy", RESID_HIST)
        ##########################################
        # Log-file output
        f = open("main.log", "a")
        if first_write:
            out = "AMG Params:\n" + str(MG_PARAMS) + "\n" + "\n"
            f.write(out)
            first_write = False
        out = f"ndofs={A0_csr.shape[0]:>12}, resid[{len(resid):>3}]={resid[-1]:2.2e}: "
        out += "system_setup=%2.2f, " % (stokes_iter.get_build_time())
        t_solve = fastest_slv["mg:solve"].sum()
        out += f"mg_setup[lvls={MG_LVLs[mg_id]:>2d}]={t_amg:>4.2f}, solve[{len(resid):>2}]={t_solve:>4.2f}"
        print(out)
        f.write(out + "\n")
        f.close()
        ##########################################
        # Patch info output
        # if MG_PARAMS['relaxation'][0].lower() == "vanka":
        #     vanka_hier = {}
        #     count = 0
        #     if hasattr(amg, "wrapper") and amg.wrapper is not None:
        #         vanka_hier[count] = amg.wrapper.outer_relaxation.patch_stats
        #         count += 1
        #     for lvl in amg.ml.levels[:1]:
        #         try:
        #             vanka_hier[count] = lvl.relax_obj.patch_stats
        #         except:
        #             pass
        #         count += 1
        #     json.dump(vanka_hier, open("patch_info.json", 'w'), indent=2)
        ##########################################
        # Clean-up potential memory leaks
        del amg
        del A0_csr
        # del stokes
        gc.collect()

    return None
