##############################################################
# DO NOT CHANGE THE ORDER OF ANYTHING IN the following 3 ARRAYS
# (unless you know what you're doing)
# Feel free to change names, colors and markers, but make sure
# they are unique.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
_names = [
    "HO-AMG",
    "DC-skip0",
    "DC-skip1",
    "DC-all",
    "BTP",
]

_colors = [
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:purple",
]
_markers = [
    "|",
    "o",
    "^",
    "s",
    "x",
]
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##############################################################

_name_to_color = dict()
_name_to_marker = dict()
for name, color, marker in zip(_names, _colors, _markers):
    _name_to_color[name] = color
    _name_to_marker[name] = marker


def get_plot_attr(name):
    return (_name_to_color[name], _name_to_marker[name])


def get_color(name):
    return _name_to_color[name]


def get_marker(name):
    return _name_to_marker[name]


def get_sensitivity_plot_names(disc):
    if disc == "p2p1":
        return (_names[2], _names[3])
    else:
        raise Exception("Unknown disc=%s" % disc)


def get_robustness_plot_names(disc):
    if disc == "p2p1":
        return (_names[0], _names[2], _names[1], _names[3])
    if disc == "p2p1_lsc":
        return (_names[4], _names[0], _names[2], _names[3])
    elif disc == "p2p1disc":
        return (_names[4], _names[3], _names[5])
    else:
        raise Exception("Unknown disc=%s" % disc)


def get_hero_plot_names(disc):
    if disc == "p2p1":
        return (
            f"{_names[0]}(Vanka)",
            f"{_names[0]}(LSC-DGS)",
            f"{_names[2]}(Vanka)",
            f"{_names[2]}(LSC-DGS)",
        )
    elif disc == "p2p1disc":
        return (_names[4], _names[3])
    else:
        raise Exception("Unknown disc=%s" % disc)
