function d = fr_meandice(mdl, graphs, idx, cols, lo, hi)
%FR_MEANDICE  Mean pixel-level Dice of a per-node classifier over IDX.
%   Used by the descriptor-ceiling diagnostic (Section 8.8).
    v = zeros(numel(idx), 1);
    for j = 1:numel(idx)
        G = graphs{idx(j)};
        p = fr_applylinear(mdl, fr_applyscaler(G.featAll(:, cols), lo, hi)) > 0.5;
        m = fr_metrics(p(G.L), G.gt);
        v(j) = m.dice;
    end
    d = mean(v);
end
