from sysmg import StructuredStokesIterator, UnstructuredStokesIterator


############################################################
# Options
############################################################
def unstructured_3D_iterator(system_params):
    system_params["discretization"]["bcs"] = "in-out-flow"
    return UnstructuredStokesIterator(
        system_params,
        name_id=1,
        dim=3,
        start_idx=0,
        end_idx=12,
        # start_idx=0, end_idx=11,
        # start_idx=10, end_idx=11,
        max_dofs=7e7,
    )


############################################################
# Access function
############################################################
def get_problem_iterator(msh_type, dim):
    structured = None
    msh_type = msh_type.lower()
    if msh_type == "structured":
        structured = True
    elif msh_type == "unstructured":
        structured = False
    else:
        raise ValueError(f"Unknown problem type {msh_type}.")

    if dim not in [2, 3]:
        raise ValueError("Dimension is wrong")

    from disc import system_params

    return globals()[f"{msh_type}_{dim}D_iterator"](system_params)
