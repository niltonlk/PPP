import numpy        as     np
from   scipy.sparse import coo_matrix
from   set_params   import *
from   lib.tools    import *

np.random.seed(s)

pre_idx, pos_idx, W = fixed_indegree(Ce, Ci, Nexct, Ninhb, w_ex, g)

conn_mat = coo_matrix((W, (pre_idx, pos_idx)), shape=(N,N)).toarray()
conn_mat.setdiag(np.zeros(N))  # Deleting autapses

np.save('graph/brunel_seed_'+str(s)+'.npy', conn_mat)