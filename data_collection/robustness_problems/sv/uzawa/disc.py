##########################################################
# Stokes Parameters
low_order_prec= False
system_params = {   'mesh'          : None, # assigned by stokes_iterator
                    'discretization': {'elem_type' : ('CG','DG'),
                                        'order' : (2,1),
                                        'bcs': None, # assigned by problem_iterator
                                       },
                    'dof_ordering'  : {'split_by_component': True,
                                        'lexicographic': True},
                    'additional'    : {'lo_fe_precond' : low_order_prec,
                                       'ho_mass' : ('p',), # needed for SV Vanka and Uzawa
                                       },
                    'keep' : False
                    }
