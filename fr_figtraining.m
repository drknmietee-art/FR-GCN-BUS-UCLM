function fr_figtraining(cfg)
%FR_FIGTRAINING  Figure 6: training loss and validation Dice against epoch.
%   The loss falls monotonically while validation accuracy plateaus early,
%   which is why a fixed long budget overshoots and early stopping is used.
%
%   This figure needs one short extra training run on fold 1, recording both
%   curves per epoch; the main experiment keeps only the best checkpoint.
    if nargin < 1, cfg = fr_config(); end
    graphs = fr_requirecache(cfg);
    patients = cellfun(@(G) {G.patient}, graphs);
    folds = fr_patientfolds(patients, cfg.kFolds);

    trAll = find(folds ~= 1);
    tp = unique(patients(trAll));
    rs = fr_seed(1000 + 1);
    [pp, ~] = fr_randperm(rs, numel(tp), max(1, round(cfg.valFraction*numel(tp))));
    isVal = ismember(patients, tp(pp));
    va = trAll(isVal(trAll)); tr = trAll(~isVal(trAll));

    [lo, hi] = fr_fitscaler(cellfun(@(G) {G.featAll}, graphs(tr)));
    idx = [tr(:); va(:)];
    Gf = cell(1, numel(graphs));
    fprintf('  building fold-1 features for the training curve...\n');
    for k = 1:numel(idx)
        i = idx(k);
        [X, Af, Ab, Ai] = fr_foldfeatures(graphs{i}, lo, hi, cfg);
        Gf{i} = struct('X', X, 'Af', Af, 'Ab', Ab, 'Ai', Ai, 'y', graphs{i}.y);
    end

    RAW  = size(graphs{1}.featAll, 2);
    cols = getfield(fr_featuresets(RAW), 'full'); %#ok<GFLD>
    net  = fr_gcn_init(numel(cols), cfg.hidden, 2, 0, 0, cfg.layers);
    rs2  = fr_seed(17);
    L = zeros(cfg.maxEpochs, 1); V = zeros(cfg.maxEpochs, 1);
    fprintf('  training %d epochs (recording both curves)...\n', cfg.maxEpochs);
    for ep = 1:cfg.maxEpochs
        [ord, rs2] = fr_randperm(rs2, numel(tr));
        tot = 0;
        for gi = ord
            G = Gf{tr(gi)};
            [P, cache, net] = fr_gcn_forward(net, G.X(:,cols), G.Af, true);
            [l, dZ] = fr_loss(P, G.y, G.X(:,end), cfg.alpha, cfg.lambda, true);
            net = fr_gcn_step(net, fr_gcn_backward(net, cache, dZ), cfg.lr);
            tot = tot + l;
        end
        L(ep) = tot / numel(tr);
        tp_ = 0; fp_ = 0; fn_ = 0;
        for j = va(:)'
            p = fr_predict(net, Gf{j}, cols, 'Af') > 0.5; yy = Gf{j}.y > 0.5;
            tp_ = tp_ + sum(p & yy); fp_ = fp_ + sum(p & ~yy); fn_ = fn_ + sum(~p & yy);
        end
        V(ep) = 2*tp_ / max(2*tp_ + fp_ + fn_, 1e-9);
    end

    fr_writecsv(fullfile(cfg.outDir, 'training_curve.csv'), ...
        {'epoch','train_loss','val_dice'}, num2cell([(1:cfg.maxEpochs)', L, V]));

    [~, best] = max(V);
    fh = figure('Position', [100 100 700 420], 'Color', 'w');
    subplot(2,1,1);
    plot(1:cfg.maxEpochs, L, '-', 'Color', [0.12 0.36 0.55], 'LineWidth', 1.5);
    ylabel('Training loss'); title('FR-GCN convergence, fold 1'); grid on
    xlim([1 cfg.maxEpochs]);
    subplot(2,1,2); hold on
    plot(1:cfg.maxEpochs, V, '-', 'Color', [0.24 0.56 0.43], 'LineWidth', 1.5);
    yl = ylim; plot([best best], yl, ':', 'Color', [0.69 0.19 0.19], 'LineWidth', 1.3);
    text(best, yl(1), sprintf('  early stopping selects epoch %d', best), ...
         'VerticalAlignment', 'bottom', 'FontSize', 8, 'Color', [0.69 0.19 0.19]);
    xlabel('Epoch'); ylabel('Validation Dice (nodes)'); grid on;
    xlim([1 cfg.maxEpochs]); hold off
    fr_savefig(fh, cfg, 'Figure6_training_curve.png');
end
