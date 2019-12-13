import numpy        as     np
from   scipy.sparse import coo_matrix
from   set_params   import *
from   lib.tools    import *

np.random.seed(s)

def brunel_graph(Ce, Ci, Nexct, Ninhb, w_ex, g, save_graph=False):

	pre_idx, pos_idx, W = fixed_indegree(Ce, Ci, Nexct, Ninhb, w_ex, g)

	# conn_mat = coo_matrix((W, (pre_idx, pos_idx)), shape=(N,N))
	# conn_mat.setdiag(np.zeros(N))  # Deleting autapses

	conn_mat = np.zeros([N,N])
	conn_mat[pre_idx.astype('int'), pos_idx.astype('int')] = W
	np.fill_diagonal(conn_mat, 0.0)

	if save_graph == True:
		np.save('graph/brunel_seed_'+str(s)+'.npy', conn_mat)

	return conn_mat
