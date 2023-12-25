import numpy as np
import pandas as pd
import json


def get_data(path_name):
    # mg hierarchy info: ndofs -> pandas data frame / dict
    json_str = json.load(open(path_name + "hierarchy_info.json"))
    hier = {int(k): pd.DataFrame.from_dict(json.loads(v)) for k, v in json_str.items()}
    # hier = {k: hier[k] for k in sorted(hier.keys())}
    # resid hist: ndofs -> array[nIterations]
    resids = np.load(path_name + "resid_hist.npy", allow_pickle=True).ravel()[0]
    # [::-1]
    # timings: 'str' -> array[nSystems]
    json_str = json.load(open(path_name + "solve_timings.json"))
    timings = {int(k): json.loads(v) for k, v in json_str.items()}
    # timings =  {k: timings[k] for k in sorted(timings.keys())}
    return hier, resids, timings  # , solve_timings


def load_data(names, paths):
    all_data = {"hierarchy": {}, "timings": {}, "residuals": {}}

    for name, path in zip(names, paths):
        data = get_data(path)
        hier, resids, timings = data

        hier = {k: hier[k] for k in sorted(hier.keys())}
        resids = {k: resids[k] for k in sorted(resids.keys())}
        timings = {k: timings[k] for k in sorted(timings.keys())}

        all_data["hierarchy"][name] = hier
        all_data["timings"][name] = timings
        all_data["residuals"][name] = resids
        # print(resids)

    return all_data
