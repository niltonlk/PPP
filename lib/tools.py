import pandas                           as pd
from   brian2                           import *
from   scipy.integrate                  import simps
from   scipy.stats                      import ks_2samp
from   scipy.signal                     import hilbert

def fixed_indegree(Ce, Ci, Nexct, Ninhb, W, g):
    pre_ex  = np.random.uniform(0, Nexct, Ce*(Nexct+Ninhb))
    pos_ex  = np.repeat( np.arange(0, Nexct+Ninhb), Ce)

    pre_in  = np.random.uniform(Nexct, Nexct+Ninhb, Ci*(Nexct+Ninhb))
    pos_in  = np.repeat( np.arange(0, Nexct+Ninhb), Ci)


    pre_idx = np.concatenate((pre_ex, pre_in))
    pos_idx = np.concatenate((pos_ex, pos_in))

    weights = np.concatenate( (W*np.ones(pre_ex.shape[0]), -g*W*np.ones(pre_in.shape[0])) )

    return pre_idx, pos_idx, weights