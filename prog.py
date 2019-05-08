import numpy as np 
import scipy.sparse as sparse
import matplotlib.pyplot as plt

def main():
	# r = 4
	# c = 3

	print("importing file...")
	gr = np.loadtxt('graph.txt',dtype=int)
	print("done...")

	c = gr[:,0].size

	print("creating 1's...")
	ones = np.ones((c,),dtype=int)
	print("done...")

	s = tuple(gr.max(axis=0)[:2]+1)

	print("creating adj_matrix...")
	adj_matrix = sparse.csr_matrix((ones,(gr[:,0],gr[:,1])),shape=s,dtype=int)
	print("done...")

	# print("creating spy plot...")
	# plt.spy(adj_matrix,marker=',')
	# plt.show()

	print("finding degree distribution...")
	# deg = adj_matrix.getrow(1).sum(axis=1)
	# print(deg)
	# print(deg.shape)
	# print(deg.sum())
	deg_dist = {}
	for i in range(s[0]):
		deg = adj_matrix.getrow(i).sum(axis=1).sum()
		if((deg in deg_dist) and deg > 0):
			deg_dist[deg] += 1

		else:
			deg_dist[deg] = 1
	print("done...")

	node_deg = list(deg_dist.keys())
	node_cnt = []

	for i in node_deg:
		node_cnt.append(deg_dist.get(i))


	plt.loglog(node_deg,node_cnt,'r.')
	plt.title("degree distribution")
	plt.xlabel("degree")
	plt.ylabel("node count")
	plt.show()

	# degree = {}
	# for i in adj_matrix.indices:
	# 	deg = adj_matrix.getcol(i).sum(axis=1)
	# 	print(deg)

	# 	if(deg > 0):
	# 		degree[i] = deg

	# print(degree)
	# print(adj_matrix.getrow(1).sum(axis=1))
	# print(adj_matrix.getrow(1))
	# print(adj_matrix.getrow(2))
	#print(adj_matrix[1][1])

	# print(adj_matrix.nnz)
	# print(adj_matrix.indices)

	# print("creating spy plot...")
	# plt.spy(adj_matrix)
	# plt.show()

	
if __name__=="__main__":
	main()