from sysmg import Structured_Stokes_Iterator, Unstructured_Stokes_Iterator


def get_problem_iterator(system_params):
    return Structured_Stokes_Iterator(
        system_params, start_idx=6, end_idx=7, dim=2, max_dofs=5e6
    )
