import math
import numpy as np 
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import matplotlib.pyplot as plt

def main():
	# Part 1: reads in file and plots sparsity
	print("importing file...")
	gr = np.loadtxt('assignment_graph.txt',delimiter=',',dtype=int)
	print("done...")

	c = gr[:,0].size

	print("creating 1's...")
	ones = np.ones((c,),dtype=int)
	print("done...")

	s = tuple(gr.max(axis=0)[:2]+1)

	print("creating adj_matrix...")
	adj_matrix = sp.csr_matrix((ones,(gr[:,0],gr[:,1])),shape=s,dtype=float)
	print("done...")

	# print("creating spy plot...")
	# plt.spy(adj_matrix,marker='.',ms=0.75)
	# plt.show()

	# Part 2: finds the degree and plots distribution vs node count
	# print("finding degree distribution...")
	# deg_dist = {}
	# for i in range(s[0]):
	# 	deg = adj_matrix.getrow(i).sum(axis=1).sum()
	# 	if((deg in deg_dist) and deg > 0):
	# 		deg_dist[deg] += 1

	# 	else:
	# 		deg_dist[deg] = 1
	# print("done...")

	# node_deg = list(deg_dist.keys())
	# node_cnt = []

	# for i in node_deg:
	# 	node_cnt.append(deg_dist.get(i))


	# plt.loglog(node_deg,node_cnt,'b.')
	# plt.title("degree distribution")
	# plt.xlabel("degree")
	# plt.ylabel("node count")
	# plt.show()

	# Part 3: singular value decomposition
	# k = int(math.sqrt(s[0] * s[1] * 0.90))

	print("solving svds...")
	U, S, V = spl.svds(adj_matrix,k=10)
	u0 = U[:,0]
	s0 = S[0][0]
	v0 = V[0,:]

	print(u0.shape)
	print(s0)
	print(v0.shape)

	u0 = np.array(u0).reshape(len(u0),1)
	s0 = np.array(s0).reshape(len(s0),1)
	v0 = np.array(v0).reshape(1,len(v0))

	print(u.shape)
	print(s)
	print(v.shape)


	print("multiplying matrix...")

	M = u0 @ s0 @ v0

	m = sp.csr_matrix(M)

	print("creating spy plot...")
	plt.spy(m,marker='.',ms=0.75,precision=0.1)
	plt.show()

	
	
if __name__=="__main__":
	main()