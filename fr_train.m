function net = fr_train(trGraphs, cols, adjName, useUnc, cfg, seed, vaGraphs, dropout)
%FR_TRAIN  One Adam step per training graph, cycling all graphs each epoch,
%   with validation-based early stopping (Section 7).
%   A fixed long budget overshoots the optimum on this dataset, so the best
%   checkpoint by validation node-Dice is kept and its epoch reported.
    if nargin < 8 || isempty(dropout), dropout = 0; end
    fin = numel(cols);
    net = fr_gcn_init(fin, cfg.hidden, 2, seed, dropout, cfg.layers);
    rs  = fr_seed(seed + 17);
    uncCol = size(trGraphs{1}.X, 2);          % boundary is the last column
    best = struct('val', -1, 'epoch', 0, 'W', {net.W});
    for ep = 1:cfg.maxEpochs
        [ord, rs] = fr_randperm(rs, numel(trGraphs));
        for gi = ord
            Gf = trGraphs{gi};
            [P, cache, net] = fr_gcn_forward(net, Gf.X(:, cols), Gf.(adjName), true);
            [~, dZ] = fr_loss(P, Gf.y, Gf.X(:, uncCol), cfg.alpha, cfg.lambda, useUnc);
            net = fr_gcn_step(net, fr_gcn_backward(net, cache, dZ), cfg.lr);
        end
        if ~isempty(vaGraphs) && (mod(ep, cfg.valEvery) == 0 || ep == cfg.maxEpochs)
            d = fr_valdice(net, vaGraphs, cols, adjName);
            if d > best.val
                best.val = d; best.epoch = ep; best.W = net.W;
            end
        end
    end
    if ~isempty(vaGraphs)
        net.W = best.W; net.bestEpoch = best.epoch; net.bestVal = best.val;
    end
end

function d = fr_valdice(net, graphs, cols, adjName)
    tp = 0; fp = 0; fn = 0;
    for i = 1:numel(graphs)
        p = fr_predict(net, graphs{i}, cols, adjName) > 0.5;
        yy = graphs{i}.y > 0.5;
        tp = tp + sum(p & yy); fp = fp + sum(p & ~yy); fn = fn + sum(~p & yy);
    end
    d = 2*tp / max(2*tp + fp + fn, 1e-9);
end
