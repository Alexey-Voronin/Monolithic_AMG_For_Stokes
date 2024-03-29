from sysmg import StructuredStokesIterator, UnstructuredStokesIterator


############################################################
# Options
############################################################
def structured_2D_iterator(system_params):
    system_params["discretization"]["bcs"] = "in-out-flow"
    NEx = [16, 32, 64, 128, 256, 512, 700]  # , 800]
    return StructuredStokesIterator(
        system_params,
        start_idx=0,
        end_idx=len(NEx),
        NEx=NEx,
        dim=2,
        max_dofs=5e7,
        shape="L",
    )


def unstructured_2D_iterator(system_params):
    system_params["discretization"]["bcs"] = "in-out-flow"
    return UnstructuredStokesIterator(
        system_params,
        name_id=1,
        dim=2,
        start_idx=0,
        end_idx=9,
        max_dofs=5e8,
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
