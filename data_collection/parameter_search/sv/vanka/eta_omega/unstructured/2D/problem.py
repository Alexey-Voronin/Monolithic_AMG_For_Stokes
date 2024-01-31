from sysmg import Structured_Stokes_Iterator, Unstructured_Stokes_Iterator


def get_problem_iterator(system_params):
    system_params["discretization"]["bcs"] = "in-out-flow"
    return Unstructured_Stokes_Iterator(
        system_params, name_id=1, start_idx=3, end_idx=4, max_dofs=3e6
    )
