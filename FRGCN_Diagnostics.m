%FRGCN_DIAGNOSTICS  Locate the residual error rather than attribute it
%   (Section 8.8).
%
%   1. The superpixel oracle ceiling: the best Dice any node classifier could
%      reach on this graph, swept over SLIC compactness.
%   2. The descriptor ceiling: fit a strong non-graph classifier to the same
%      node descriptors.  If it plateaus where the graph network does, the
%      descriptor is the binding constraint, not the model.
%   3. The distribution of the fuzzy edge weights, which is what distinguishes
%      selective gating from uniform suppression.

cfg = fr_config();
graphs = fr_requirecache(cfg);
n = numel(graphs);

% ---- 1. oracle ceiling vs compactness ---------------------------------
comps = [0.05 0.10 0.15 0.30 1.00];
rows = {};
sub = 1:min(40, n);
for c = comps
    d = zeros(numel(sub), 1); ns = zeros(numel(sub), 1);
    for j = 1:numel(sub)
        G = graphs{sub(j)};
        L = fr_slic(G.gray, cfg.nSuperpix, c, cfg.slicIters);
        K = max(L(:)); ns(j) = K;
        pred = false(size(L));
        for k = 1:K
            m = (L == k);
            if mean(G.gt(m)) > 0.5, pred = pred | m; end
        end
        mm = fr_metrics(pred, G.gt); d(j) = mm.dice;
    end
    rows(end+1,:) = {c, mean(d), mean(ns)}; %#ok<SAGROW>
    fprintf('compactness=%.2f  oracle Dice=%.4f  mean superpixels=%.0f\n', c, mean(d), mean(ns));
end
fr_writecsv(fullfile(cfg.outDir,'D_oracle_sweep.csv'), ...
    {'compactness','oracle_dice','mean_superpixels'}, rows);

% ---- 2. descriptor ceiling: v1 (6 descriptors) vs v2 (20) --------------
patients = cellfun(@(G) {G.patient}, graphs);
folds = fr_patientfolds(patients, cfg.kFolds);
rows = {};
for variant = 1:2
    if variant == 1, cols = 1:6; nm = 'v1: 6 hand-designed descriptors';
    else,            cols = 1:size(graphs{1}.featAll,2); nm = 'v2: 20 descriptors'; end
    te_d = zeros(cfg.kFolds,1); tr_d = zeros(cfg.kFolds,1);
    for f = 1:cfg.kFolds
        te = find(folds == f); tr = find(folds ~= f);
        [lo, hi] = fr_fitscaler(cellfun(@(G) {G.featAll(:,cols)}, graphs(tr)));
        Xtr = []; ytr = [];
        for i = tr(:)'
            Xtr = [Xtr; fr_applyscaler(graphs{i}.featAll(:,cols), lo, hi)]; %#ok<AGROW>
            ytr = [ytr; graphs{i}.y];                                        %#ok<AGROW>
        end
        mdl = fr_fitlinear(Xtr, ytr, 800, 0.05);
        te_d(f) = fr_meandice(mdl, graphs, te, cols, lo, hi);
        tr_d(f) = fr_meandice(mdl, graphs, tr(1:min(60,numel(tr))), cols, lo, hi);
    end
    rows(end+1,:) = {nm, mean(te_d), mean(tr_d)}; %#ok<SAGROW>
    fprintf('  %-34s test Dice %.4f   train Dice %.4f\n', nm, mean(te_d), mean(tr_d));
end
fr_writecsv(fullfile(cfg.outDir,'D_descriptor_ceiling.csv'), ...
    {'features','test_dice','train_dice'}, rows);

% ---- 3. fuzzy edge-weight distribution --------------------------------
[lo, hi] = fr_fitscaler(cellfun(@(G) {G.featAll}, graphs));
adjR = [];
for i = 1:min(25, n)
    Xs = fr_applyscaler(graphs{i}.featAll, lo, hi);
    R  = exp(-(fr_pdist2(Xs, Xs).^2) ./ (2*cfg.sigmaE^2));
    adjR = [adjR; R(graphs{i}.adj)]; %#ok<AGROW>
end
fr_writecsv(fullfile(cfg.outDir,'D_edge_weights.csv'), ...
    {'R_adjacent_mean','R_adjacent_median','R_adjacent_p10','R_adjacent_p90'}, ...
    {mean(adjR), median(adjR), fr_prctile(adjR,10), fr_prctile(adjR,90)});
fprintf('edge weights on adjacent pairs: mean %.3f (p10 %.3f, p90 %.3f)\n', ...
        mean(adjR), fr_prctile(adjR,10), fr_prctile(adjR,90));
