import numpy        as     np
from   set_params   import *
from   lib.tools    import *
# from scipy.sparse import coo_matrix

np.random.seed(rseed)

def fixed_indegree(Ce, Ci, Nexct, Ninhb, W, g):

	pre_ex  = np.random.uniform(0, Nexct, Ce*(Nexct+Ninhb))
	pos_ex  = np.repeat( np.arange(0, Nexct+Ninhb), Ce)

	pre_in  = np.random.uniform(Nexct, Nexct+Ninhb, Ci*(Nexct+Ninhb))
	pos_in  = np.repeat( np.arange(0, Nexct+Ninhb), Ci)

	pre_idx = np.concatenate((pre_ex, pre_in))
	pos_idx = np.concatenate((pos_ex, pos_in))

	pre_idx = pre_idx.astype(int)
	pos_idx = pos_idx.astype(int)

	weights = np.concatenate( (W*np.ones(pre_ex.shape[0]), -g*W*np.ones(pre_in.shape[0])) )

	return pre_idx, pos_idx, weights

# def brunel_graph(N, w_ex, g, save_graph=False):
#
# 	f     = 0.8         	# fraction of excitatory neurons
# 	p	  = 0.1				# probability of connection
# 	Nexct = int(f * N)  	# number of excitatory neurons
# 	Ninhb = N-Nexct			# number of inhibitory neurons
# 	Ce    = int(Nexct*p)
# 	Ci    = int(Ce*0.25)
#
# 	pre_idx, pos_idx, W = fixed_indegree(Ce, Ci, Nexct, Ninhb, w_ex, g)
#
#	# sparse matrix
# 	conn_mat = coo_matrix((W, (pre_idx, pos_idx)), shape=(N,N))
# 	conn_mat.setdiag(np.zeros(N))  # Deleting autapses
# 	conn_mat = conn_mat.tolil()
#
# 	# conn_mat = np.zeros([N,N])
# 	# conn_mat[pre_idx.astype('int'), pos_idx.astype('int')] = W
# 	# np.fill_diagonal(conn_mat, 0.0)
#
# 	if save_graph == True:
# 		np.save('graph/brunel_seed_'+str(s)+'.npy', conn_mat)
#
# 	return conn_mat

def brunel_graph(N, w_ex, g, save_graph=False):

	f     = 0.8         	# fraction of excitatory neurons
	p	  = 0.1				# probability of connection
	Nexct = int(f * N)  	# number of excitatory neurons
	Ninhb = N-Nexct			# number of inhibitory neurons
	Ce    = int(Nexct*p)
	Ci    = int(Ce*0.25)

	pre_idx, pos_idx, W = fixed_indegree(Ce, Ci, Nexct, Ninhb, w_ex, g)

	post_list = []
	for i in range(Nexct+Ninhb):
		post_list.append([pos_idx[pre_idx==i],W[pre_idx==i]])

	if save_graph == True:
		np.save('graph/brunel_seed_'+str(s)+'.npy', post_list)

	return post_list

def all_to_all(N, w):
	conn_mat = np.ones((N,N))*w

	return conn_mat
