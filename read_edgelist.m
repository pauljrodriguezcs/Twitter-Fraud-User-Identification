graph = fopen('graph.txt', 'r');
edge_list = fscanf(graph, '%i %i', [2 inf]);
edge_list = edge_list';
adj = sparse(edge_list(:,1),edge_list(:,2), 1);
%spy(adj);
%ax = gca;
%ax.YDir = 'normal';
fclose(graph);

% part 2
[m,n] = size(adj);
deg_dist = containers.Map;
for i=1:m
    deg = sum(sum(adj(i,:)));
    deg(1,1)
    if(deg > 0)
        fprintf('true\n')
        if(isKey(deg_dist,deg))
            oldv = deg_dist(deg);
            oldv
            deg_dist(deg) = oldv + 1;
        else
            deg_dist(deg) = 1;
        end
    end
end

loglog(deg_dist)
