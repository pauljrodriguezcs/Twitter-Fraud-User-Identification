import math
import numpy as np 
import scipy.sparse as sparse
import scipy.spatial.distance as spd
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt

def main():
	### Part 1: reads in file and plots sparsity ###
	### 1)
	print("importing file ...")
	gr = np.loadtxt('assignment_graph.txt',delimiter=',',dtype=int)
	print("done ...")

	c = gr[:,0].size

	print("creating 1's...")
	ones = np.ones((c,),dtype=int)
	print("done ...")

	s = tuple(gr.max(axis=0)[:2]+1)

	print("creating adj_matrix ...")
	adj_matrix = sparse.csr_matrix((ones,(gr[:,0],gr[:,1])),shape=s,dtype=float)
	print("done ...")

	### 3)
	print("creating spy plot...")
	plt.title("Sparsity Graph")
	plt.spy(adj_matrix,marker='.',markersize=1)
	plt.show()

	### Part 2: finds the degree and plots distribution vs node count ###
	### 1)
	print("\nfinding degree distribution ...")
	# map with where key value is the degree of node and value is the count 
	deg_dist = {}
	node_sum = []
	for i in range(s[0]):
		deg = adj_matrix.getrow(i).sum(axis=1).sum()
		node_sum.append(deg)
		if((deg in deg_dist) and deg > 0):
			deg_dist[deg] += 1

		else:
			deg_dist[deg] = 1

	node_deg = list(deg_dist.keys())
	node_cnt = []

	for i in node_deg:
		node_cnt.append(deg_dist.get(i))

	print("done ...")

	### 2)
	# print("plotting degree distribution ...")
	# plt.loglog(node_deg,node_cnt,'b.')
	# plt.title("degree distribution")
	# plt.xlabel("degree")
	# plt.ylabel("node count")
	# plt.show()

	### 4)
	# plt.plot(node_sum,'bo',markersize=1)
	# plt.title("abnormal nodes")
	# plt.xlabel("node")
	# plt.ylabel("degree")
	# plt.xticks(np.arange(0,len(node_sum),1000))
	# plt.show()

	### Part 3: singular value decomposition ###
	### 2a) find k that reproduces 90% of original data
	# k = 80
	# p_data = 0.0

	# while(p_data < 0.90):
	# 	print("\nsolving svds for k =", k)
	# 	U, S, V = spl.svds(adj_matrix,k=k)
	# 	u = np.array(U)
	# 	s = np.diag(S)
	# 	v = np.array(V)
	# 	m = (u @ s @ v)

	# 	print(" SQE calc ...")
	# 	sqe = spd.sqeuclidean(m.flatten(),adj_matrix.toarray().flatten())

	# 	adj_tot = len(adj_matrix.nonzero()[0])

	# 	p_data = 1 - (sqe/adj_tot)
	# 	print(" Data reconstruction: ", p_data, "%")

	# 	print("\nPlotting graphs ...")
		
	# 	plt.subplot(121)
	# 	plt.title("90% of Reconstructed Data")
	# 	plt.spy(m,markersize=1,precision=0.1)
	# 	plt.subplot(122)
	# 	plt.title("Full Graph")
	# 	plt.spy(adj_matrix,markersize=1)
	# 	plt.show()

	# 	k += 1

	top_k = 5

	### 2b) 
	# print("\nsolving svds for k =", top_k)
	# U, S, V = spl.svds(adj_matrix,k=top_k)
	# print("plotting ... " )

	# for i in range(5):
	# 	plt.subplot(2,3,(i+1))
	# 	plt.title('Left Singular Vector #%i' % (i+1))
	# 	plt.plot(U[:,i],markersize=1)

	# plt.subplot(2,3,6)
	# plt.title('Left Singular Vector Top 5')
	# plt.plot(U,markersize=1)
	# plt.legend(('LSV - 1','LSV - 2','LSV - 3','LSV - 4','LSV - 5'),loc='best')
	# plt.show()

	### 2d)
	# print("\nsolving svds for k =", top_k)
	# U, S, V = spl.svds(adj_matrix,k=top_k)

	# u_abs = np.array(U)
	# u_abs = np.absolute(np.array(U))

	# indices = np.ones([100,top_k],dtype=int)

	# print("finding top 100 ...")
	# for i in range(top_k):
	# 	ind = np.argpartition(u_abs[:,i], -100)[-100:]
	# 	indices[:,i] = ind[np.argsort(u_abs[:,i][ind])]

	# u_top = np.zeros(U.shape)
	# v_top = np.zeros(V.shape)

	# print("obtaining top 100 ...")
	# for i in range(top_k):
	# 	for j in range(indices.shape[0]):
	# 		ind = indices[j,i]
	# 		u_top[ind,i] = U[ind,i]
	# 		v_top[i,ind] = V[i,ind]

	# print("graphing top 100 ...")
	# for i in range(5):
	# 	plt.subplot(2,3,(i+1))
	# 	plt.title('Matrix #%i' % (i+1))
	# 	m = np.reshape(S[i] * u_top[:,i],(u_top.shape[0],1)) * v_top[i,:]
	# 	plt.spy(m,markersize=1)

	# plt.subplot(2,3,6)
	# plt.title('Matrix Top 5')
	# s_dag = np.diag(S)
	# m_top = u_top @ s_dag @ v_top
	# print(m_top.shape)
	# plt.spy(m_top,markersize=1)
	# plt.show()
	
	
if __name__=="__main__":
	main()