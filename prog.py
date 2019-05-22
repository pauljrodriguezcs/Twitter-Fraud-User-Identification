import math
import numpy as np 
import scipy.sparse as sparse
import scipy.spatial.distance as spd
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt

def main():
	### Part 1: reads in file and plots sparsity ###
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

	# print("creating spy plot...")
	# plt.title("Sparsity Graph")
	# plt.spy(adj_matrix,marker='.',markersize=1)
	# plt.show()

	### Part 2: finds the degree and plots distribution vs node count ###

	# print("\nfinding degree distribution ...")
	# deg_dist = {}
	# for i in range(s[0]):
	# 	deg = adj_matrix.getrow(i).sum(axis=1).sum()
	# 	if((deg in deg_dist) and deg > 0):
	# 		deg_dist[deg] += 1

	# 	else:
	# 		deg_dist[deg] = 1

	# node_deg = list(deg_dist.keys())
	# node_cnt = []

	# for i in node_deg:
	# 	node_cnt.append(deg_dist.get(i))

	# print("done ...")

	# print("plotting degree distribution ...")
	# plt.loglog(node_deg,node_cnt,'b.')
	# plt.title("degree distribution")
	# plt.xlabel("degree")
	# plt.ylabel("node count")
	# plt.show()

	### Part 3: singular value decomposition ###

	# a) find k that reproduces 90% of original data
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

	# b) 
	# print("\nsolving svds for k =", top_k)
	# U, S, V = spl.svds(adj_matrix,k=top_k)
	# print("plotting ... " )
	# plt.subplot(131)
	# plt.title("Left Singular Vectors (U)")
	# plt.plot(U,markersize=1)
	# plt.subplot(132)
	# plt.title("Singular Values (S)")
	# plt.plot(S,markersize=1)
	# plt.subplot(133)
	# plt.title("Right Singular Vectors (V)")
	# plt.plot(V,markersize=1)
	# plt.show()

	# d)
	
	print("\nsolving svds for k =", top_k)
	U, S, V = spl.svds(adj_matrix,k=top_k)

	u_abs = np.array(U)
	# u_abs = np.absolute(np.array(U))

	indices = np.ones([100,top_k],dtype=int)

	for i in range(top_k):
		ind = np.argpartition(u_abs[:,i], -100)[-100:]
		indices[:,i] = ind[np.argsort(u_abs[:,i][ind])]


	
	u_top = np.ones([100,top_k])
	v_top = np.ones([top_k,100])

	m = np.argmax(u_abs[:,0])
	print(U[:,0][indices[99,0]])
	print(U[:,0][m])

	for i in range(top_k):
		u_top[:,i] = U[:,i][indices[:,i]]
		v_top[i,:] = V[i,:].T[indices[:,i]].T

	s_dag = np.diag(S)
	
	m_top = u_top @ s_dag @ v_top

	print(u_top)
	print(s_dag)
	print(v_top)
	print(m_top)

	plt.spy(m_top,markersize=1)
	plt.show()

	

	# plt.spy(m,markersize=1,precision=0.1)

	# plt.subplot(341)

	# for i in range(12):
	# 	print("Plot: ", (i+1))
	# 	plt.subplot(3,4,(i+1))

	# 	u = U[:,i]
	# 	s = S[i]
	# 	v = V[i,:]

	# 	u = np.array(u).reshape(len(u),1)
	# 	s = np.array(s).reshape(1,1)
	# 	v = np.array(v).reshape(1,len(v))

	# 	m = (u @ s @ v)
	# 	plt.spy(m,markersize=1,precision=0.1)

	# plt.show()

	
	
if __name__=="__main__":
	main()