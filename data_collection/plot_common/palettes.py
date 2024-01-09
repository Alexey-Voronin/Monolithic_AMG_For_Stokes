##############################################################
# DO NOT CHANGE THE ORDER OF ANYTHING IN the following 3 ARRAYS
# (unless you know what you're doing)
# Feel free to change names, colors and markers, but make sure
# they are unique.
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
__names = [
    "HO-AMG",  # 0
    "DC-skip0",  # 1
    "DC-skip1",  # 2
    "DC-all",  # 3
    "BTP",  # 4
]

_names = {
    "Vanka": [f"{name}(Vanka)" for name in __names[:-1]],
    "LSC-DGS": [f"{name}(LSC-DGS)" for name in [__names[0], __names[2], __names[3]]],
    # "mixed"   : [f"{name}(Vanka($\ell=0$),LSC-DGS($\ell>0$))" for name in\
    #                         [__names[3]]],
    "mixed": [
        f"{__names[3]}(Vanka)",
        f"{__names[3]}(Vanka($\ell=0$),LSC-DGS($\ell>0$))",
    ],
}

__colors = [
    "tab:blue",
    "tab:green",
    "tab:orange",
    "tab:red",
    "tab:purple",
]
_colors = {
    "Vanka": __colors[:-1],
    "LSC-DGS": [__colors[0], __colors[2], __colors[3]],
    "mixed": [__colors[3], __colors[3]],
}
__markers = [
    "|",
    "o",
    "^",
    "s",
    "x",
]
_markers = {
    "Vanka": __markers[:-1],
    "LSC-DGS": [__markers[0], __markers[2], __markers[3]],
    "mixed": [__markers[3], __markers[3]],
}

__linestyles = ["solid", "dotted", "dashdot"]
_linestyles = {
    "Vanka": [__linestyles[0]] * len(_names["Vanka"]),
    "LSC-DGS": [__linestyles[1]] * len(_names["LSC-DGS"]),
    "mixed": [__linestyles[0], __linestyles[2]],
}

# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
##############################################################

"""
_name_to_color = dict()
_name_to_marker = dict()
for name, color, marker in zip(_names, _colors, _markers):
    _name_to_color[name] = color
    _name_to_marker[name] = marker


def get_plot_attr(name):
    return (_name_to_color[name], _name_to_marker[name])
"""


def get_sensitivity_th_attr(rlx="Vanka"):
    if rlx == "Vanka":
        names = [__names[2], __names[3]]
        colors = [__colors[2], __colors[3]]
        markers = [__markers[2], __markers[3]]
        linestyles = [__linestyles[0]] * 2
    else:
        names = [__names[2], __names[3]]
        colors = [__colors[2], __colors[3]]
        markers = [__markers[2], __markers[3]]
        linestyles = [__linestyles[1]] * 2

    return names, colors, markers, linestyles


def get_section_4_th_attr(rlx="Vanka"):
    if rlx == "Vanka":
        names = [__names[4]] + _names[rlx]
        colors = [__colors[4]] + _colors[rlx]
        markers = [__markers[4]] + _markers[rlx]
        linestyles = [__linestyles[0]] + _linestyles[rlx]
    else:
        names = [__names[4]] + _names[rlx]
        colors = [__colors[4]] + _colors[rlx]
        markers = [__markers[4]] + _markers[rlx]
        linestyles = [__linestyles[0]] + _linestyles[rlx]

    return names, colors, markers, linestyles


def get_section_4_sv_attr():
    rlx = "mixed"
    names = [__names[4]] + _names[rlx]
    colors = [__colors[4]] + _colors[rlx]
    markers = [__markers[4]] + _markers[rlx]
    linestyles = [__linestyles[0]] + _linestyles[rlx]

    return names, colors, markers, linestyles


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
