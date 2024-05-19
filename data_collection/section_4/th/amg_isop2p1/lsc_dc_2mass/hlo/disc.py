##########################################################
# Stokes Parameters
low_order_prec = True
system_params = {
    "mesh": None,  # assigned by stokes_iterator
    "discretization": {
        "elem_type": ("CG", "CG"),
        "order": (2, 1),
        "bcs": None,  # assigned by problem_iterator
    },
    "dof_ordering": {"split_by_component": True, "lexicographic": True},
    "additional": {
        "lo_fe_precond": low_order_prec,
        "lo_stiffness": ("p",),
        "ho_mass": ("u",),
        "lo_mass": ("u",),
    },
    "keep": False,
}
