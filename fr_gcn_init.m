function net = fr_gcn_init(fin, hidden, nclass, seed, dropout, layers)
%FR_GCN_INIT  He-initialised L-layer graph convolutional network.
%   Weights, Adam moments and the deterministic stream are carried in NET.
    if nargin < 3 || isempty(nclass),  nclass  = 2;   end
    if nargin < 4 || isempty(seed),    seed    = 0;   end
    if nargin < 5 || isempty(dropout), dropout = 0.0; end
    if nargin < 6 || isempty(layers),  layers  = 2;   end

    dims = [fin, repmat(hidden, 1, layers - 1), nclass];
    s = fr_seed(seed + 991);
    net.W = cell(1, layers);
    for i = 1:layers
        [g, s] = fr_randn(s, dims(i), dims(i+1));
        net.W{i} = g * sqrt(2 / dims(i));
    end
    net.rs      = s;
    net.p       = dropout;
    net.m       = cellfun(@(w) zeros(size(w)), net.W, 'UniformOutput', false);
    net.v       = cellfun(@(w) zeros(size(w)), net.W, 'UniformOutput', false);
    net.t       = 0;
    net.layers  = layers;
end
