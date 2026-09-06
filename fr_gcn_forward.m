function [P, cache, net] = fr_gcn_forward(net, X, A, istrain)
%FR_GCN_FORWARD  Propagation rule of Equations (9)-(10).
%   Each layer computes ReLU(A H W); the last layer is linear + softmax.
    if nargin < 4 || isempty(istrain), istrain = false; end
    L = net.layers;
    cache.inp   = cell(1, L);
    cache.pre   = cell(1, L);
    cache.masks = cell(1, L - 1);
    H = X;
    for i = 1:L
        Hin = A * H;
        cache.inp{i} = Hin;
        Z = Hin * net.W{i};
        cache.pre{i} = Z;
        if i < L
            H = max(Z, 0);
            if istrain && net.p > 0
                [u, net.rs] = fr_rand(net.rs, size(H,1), size(H,2));
                mk = double(u > net.p) / (1 - net.p);
                H  = H .* mk;
                cache.masks{i} = mk;
            else
                cache.masks{i} = [];
            end
        else
            H = Z;
        end
    end
    cache.A = A;
    P = fr_softmax(H);
end
