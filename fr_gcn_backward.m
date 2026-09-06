function grads = fr_gcn_backward(net, cache, dZ)
%FR_GCN_BACKWARD  Analytic gradients for the network of FR_GCN_FORWARD.
%   The normalised adjacency of Equation (8) is symmetric, so A' is A and the
%   transpose is skipped.
    L = net.layers;
    A = cache.A;
    grads = cell(1, L);
    d = dZ;
    for i = L:-1:1
        grads{i} = cache.inp{i}' * d;
        if i == 1, break; end
        dH = A * (d * net.W{i}');
        if ~isempty(cache.masks{i-1})
            dH = dH .* cache.masks{i-1};
        end
        d = dH .* double(cache.pre{i-1} > 0);
    end
end
