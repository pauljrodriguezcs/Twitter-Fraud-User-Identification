edge_list = dlmread('assignment_graph.txt',',');
[m,n] = size(edge_list);
onearray = ones(m,1);
adj = sparse(edge_list(:,1),edge_list(:,2), onearray(:,1));
spy(adj);

% part 2%
% [m,n] = size(adj);
% deg_dist = containers.Map;
%for i=1:m
%    deg = sum(sum(adj(i,:)));
%    deg(1,1)
%    if(deg > 0)
%        fprintf('true\n')
%        if(isKey(deg_dist,deg))
%            oldv = deg_dist(deg);
%            oldv
%            deg_dist(deg) = oldv + 1;
%        else
%            deg_dist(deg) = 1;
%        end
%    end
%end

%loglog(deg_dist)
