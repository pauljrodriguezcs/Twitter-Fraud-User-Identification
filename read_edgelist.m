% part 1%
edge_list = dlmread('assignment_graph.txt',',');
[m,n] = size(edge_list);
onearray = ones(m,1);
adj = sparse(edge_list(:,1),edge_list(:,2), onearray(:,1));
% spy(adj);

% part 3%
[U,S,V] = svds(adj,6);

M = U(:,1:1)*S(1:1,1:1)*V(:,1:1)';
m = sparse(M);
spy(m);



