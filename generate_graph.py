import numpy        as     np
from   set_params   import *
#from   lib.tools    import *
# from scipy.sparse import coo_matrix

np.random.seed(rseed)

def fixed_indegree(Ce, Ci, Nexct, Ninhb, W, g, d_e, d_i):
    pre_ex  = np.random.uniform(0, Nexct, Ce*(Nexct+Ninhb))
    pos_ex  = np.repeat( np.arange(0, Nexct+Ninhb), Ce)
    
    pre_in  = np.random.uniform(Nexct, Nexct+Ninhb, Ci*(Nexct+Ninhb))
    pos_in  = np.repeat( np.arange(0, Nexct+Ninhb), Ci)
    
    pre_idx = np.concatenate((pre_ex, pre_in))
    pos_idx = np.concatenate((pos_ex, pos_in))
    
    pre_idx = pre_idx.astype(int)
    pos_idx = pos_idx.astype(int)
    
    #weights = np.concatenate( (W*np.ones(pre_ex.shape[0]), 
    #                           -g*W*np.ones(pre_in.shape[0])) )
    #delays = np.concatenate( (d_ex*np.ones(pre_ex.shape[0]), 
    #                          d_in*np.ones(pre_in.shape[0])) )
    weights = np.concatenate( (
            np.random.normal( W, W*0.1, pre_ex.shape[0] ), 
            np.random.normal( -g*W, g*W*0.1, pre_in.shape[0] ) ) )
    delays = np.concatenate( (
            np.random.normal( d_ex, d_ex*0.1, pre_ex.shape[0] ), 
            np.random.normal( d_in, d_in*0.1, pre_in.shape[0] ) ) )

    return pre_idx, pos_idx, weights, delays


def brunel_graph(N, w_ex, g, d_ex, d_in, save_graph=False):
    f     = 0.8            # fraction of excitatory neurons
    p	  = 0.1	           # probability of connection
    Nexct = int(f * N)     # number of excitatory neurons
    Ninhb = N-Nexct        # number of inhibitory neurons
    Ce    = int(Nexct*p)
    Ci    = int(Ce*0.25)
    
    pre_idx, pos_idx, W, D = fixed_indegree(Ce, Ci, Nexct, Ninhb, 
                                            w_ex, g, d_ex, d_in)
    
    post_list = []
    for i in range(Nexct+Ninhb):
        post_list.append( sorted( np.stack(
                [pos_idx[pre_idx==i],W[pre_idx==i],D[pre_idx==i]],
                axis=-1).tolist(), key=lambda x:x[2]) )

    if save_graph == True:
        np.save('graph/brunel_seed_'+str(s)+'.npy', post_list)

    return post_list


def all_to_all(N, w, d):
    conn_mat = np.ones((N,N))*w
    delay_mat = np.ones((N,N))*d
    
    return conn_mat, delay_mat
