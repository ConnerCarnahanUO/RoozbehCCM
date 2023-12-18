import numpy as np
import scipy as sci
import collections

def SymbolizeSet(S,grain = 0.1):
    """ Convert a set of vectors in to the finite set of symbols that span the set 
    S: an T x N matrix representing T vectors with N coordinates 
    grain = 0.1: The graining (resolution) on the space for symbolization"""

    def symbolizepoint(x):
        f = 0
        n = 1
        for i in range(len(x)-1):
            f += int( ((x[0] > grain) and (x[i] > grain)) or ((x[0] < grain) and (x[i] < grain )))*n
            n << 1
        return f

    Gamma = collections.defaultdict(list)
    SGamma = np.zeros(S.shape[0])*np.nan

    for i in range(S.shape[0]):
        SGamma[i] = f = symbolizepoint(S[i,:])
        Gamma[f].append(i)

    return SGamma, Gamma

