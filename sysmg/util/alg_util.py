
def dropSmallEntries(M, tau):
    Min = M.copy()
    D = Min.diagonal()
    from math import sqrt
    if tau < 1e-8:
        return Min

    data = Min.data.copy()
    indices = Min.indices.copy()
    indptr = Min.indptr.copy()

    for n in range(Min.shape[0]):
        start, end = indptr[n], indptr[n+1]
        diag_idx = -1
        lump = 0
        for idx in range(start, end):
            if indices[idx] == n:
                diag_idx = idx
            elif abs(data[idx]) <= tau*sqrt(abs(D[n]*D[indices[idx]])):
                lump += data[idx]
                data[idx] = 0.0

        if diag_idx >= 0:
            data[diag_idx] += lump

    Min.data = data
    Min.eliminate_zeros()

    return Min.tocsr()
