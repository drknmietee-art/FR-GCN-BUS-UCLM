%FRGCN_RUNALL  The single seeded execution behind every table (Section 6.5).
%
%   One run produces every graph model, every ablation, the no-propagation
%   control and the classical baselines, over 5 patient-grouped folds x 5
%   random initialisations.  Reporting the same model from separate runs in
%   different tables makes the tables mutually incomparable, so this script
%   is deliberately the only source of results/perimage_all.csv.
%
%   Requires: FRGCN_Extract to have been run first.

cfg = fr_config();
% Quick mode for a first end-to-end check: set FRGCN_QUICK = true before
% running.  It reduces seeds, folds and epochs by roughly 20x.  Use it to
% confirm the pipeline runs; it is NOT the protocol reported in the paper.
if exist('FRGCN_QUICK','var') && FRGCN_QUICK
    cfg.seeds = 0:1; cfg.maxEpochs = 30; cfg.kFolds = 3;
    fprintf('*** FRGCN_QUICK: reduced budget, not the published protocol ***\n');
end
graphs = fr_requirecache(cfg);
n   = numel(graphs);
patients = cellfun(@(G) {G.patient}, graphs);
folds    = fr_patientfolds(patients, cfg.kFolds);
RAW      = size(graphs{1}.featAll, 2);
SETS     = fr_featuresets(RAW);

fprintf('%d lesion images, %d patients, fold sizes %s\n', n, ...
        numel(unique(patients)), mat2str(accumarray(folds, 1)'));
fprintf('raw descriptor width %d, sigmaE %.2f, sigmaB %.2f\n', RAW, cfg.sigmaE, cfg.sigmaB);

% name , feature set , adjacency , uncertainty-weighted loss , dropout
CONFIGS = {
 'FR-GCN',                     'full',  'Af', true,  0
 'Plain GCN',                  'base',  'Ab', false, 0
 'FR-GCN-lean',                'lean',  'Af', false, 0
 'abl: no fuzzy membership',   'nomem', 'Af', true,  0
 'abl: no rough boundary',     'nobnd', 'Af', true,  0
 'abl: no uncertainty loss',   'full',  'Af', false, 0
 'abl: no fuzzy edge weights', 'full',  'Ab', true,  0
 'MLP (no propagation)',       'full',  'Ai', true,  0     % control
 'FR-GCN (MC-dropout)',        'full',  'Af', true,  cfg.mcDropout
};

% ---- superpixel oracle ceiling (Section 8.8) ---------------------------
orc = zeros(n,1);
for i = 1:n
    pred = false(cfg.res);
    for k = find(graphs{i}.y > 0.5)'
        pred = pred | (graphs{i}.L == k);
    end
    m = fr_metrics(pred, graphs{i}.gt); orc(i) = m.dice;
end
fprintf('superpixel oracle Dice = %.4f\n', mean(orc));
fr_writecsv(fullfile(cfg.outDir,'oracle_ceiling.csv'), ...
    {'superpixel_oracle_dice_mean','std','n'}, {mean(orc), std(orc), n});

rows = {}; postStore = struct();
t0 = tic;
for f = 1:cfg.kFolds
    te    = find(folds == f);
    trAll = find(folds ~= f);
    % patient-disjoint validation split, carved from the TRAINING fold only
    tp    = unique(patients(trAll));
    rs    = fr_seed(1000 + f);
    [pp, ~] = fr_randperm(rs, numel(tp), max(1, round(cfg.valFraction*numel(tp))));
    vp    = tp(pp);
    isVal = ismember(patients, vp);
    va    = trAll(isVal(trAll));
    tr    = trAll(~isVal(trAll));

    lo_hi = cellfun(@(G) {G.featAll}, graphs(tr));
    [lo, hi] = fr_fitscaler(lo_hi);

    Gf = cell(1, n);
    for i = 1:n
        [X, Af, Ab, Ai] = fr_foldfeatures(graphs{i}, lo, hi, cfg);
        Gf{i} = struct('X', X, 'Af', Af, 'Ab', Ab, 'Ai', Ai, 'y', graphs{i}.y);
    end
    fprintf('[fold %d] features built (%.0fs)\n', f, toc(t0));

    % ---- classical baselines (seed-independent) -----------------------
    Xtr = []; ytr = [];
    for i = trAll(:)'
        Xtr = [Xtr; Gf{i}.X(:, 1:RAW)]; ytr = [ytr; Gf{i}.y]; %#ok<AGROW>
    end
    svm = fr_fitlinear(Xtr, ytr);
    for i = te(:)'
        pm = fr_applylinear(svm, Gf{i}.X(:, 1:RAW)) > 0.5;
        rows(end+1,:) = fr_pad(fr_record('SLIC + SVM', -1, f, graphs{i}, ...
                        fr_node2pix(double(pm), graphs{i}, 0.5)), 11); %#ok<SAGROW>
        rows(end+1,:) = fr_pad(fr_record('FCM + threshold', -1, f, graphs{i}, ...
                        fr_fcmthreshold(graphs{i}, cfg)), 11); %#ok<SAGROW>
    end

    % ---- graph models -------------------------------------------------
    for s = cfg.seeds
        for c = 1:size(CONFIGS, 1)
            cname = CONFIGS{c,1}; cols = SETS.(CONFIGS{c,2});
            adjN  = CONFIGS{c,3};  useU = CONFIGS{c,4}; dp = CONFIGS{c,5};
            net = fr_train(Gf(tr), cols, adjN, useU, cfg, s, Gf(va), dp);
            for i = te(:)'
                post = fr_predict(net, Gf{i}, cols, adjN);
                pred = fr_node2pix(post, graphs{i}, cfg.tau);
                rec  = fr_record(cname, s, f, graphs{i}, pred);
                if strcmp(cname, 'FR-GCN')
                    rec{end+1} = fr_hd95(fr_largestcc(pred), graphs{i}.gt); %#ok<SAGROW>
                end
                rows(end+1,:) = fr_pad(rec, 11); %#ok<SAGROW>
                if s == cfg.seeds(1)
                    key = fr_key(graphs{i}.name);
                    if strcmp(cname, 'FR-GCN')
                        postStore.(key).post = post;
                        postStore.(key).bnd  = Gf{i}.X(:, end);
                        postStore.(key).y    = graphs{i}.y;
                        postStore.(key).frac = graphs{i}.frac;
                        postStore.(key).npix = accumarray(graphs{i}.L(:), 1);
                        postStore.(key).name = graphs{i}.name;
                    elseif strcmp(cname, 'FR-GCN (MC-dropout)')
                        [~, sd] = fr_predict_mc(net, Gf{i}, cols, adjN, 20);
                        postStore.(key).mc = sd;
                    end
                end
            end
            % training-fold Dice + selected epoch (under/overfitting diagnostic)
            if s == cfg.seeds(1)
                sub = tr(1:min(40, numel(tr)));
                d = zeros(numel(sub), 1);
                for j = 1:numel(sub)
                    i = sub(j);
                    m = fr_metrics(fr_node2pix(fr_predict(net, Gf{i}, cols, adjN), ...
                                               graphs{i}, cfg.tau), graphs{i}.gt);
                    d(j) = m.dice;
                end
                rows(end+1,:) = {[cname ' [TRAIN-FOLD]'], s, f, '__train__', ...
                                 mean(d), NaN, NaN, NaN, NaN, NaN, ...
                                 fr_get(net, 'bestEpoch')}; %#ok<SAGROW>
            end
        end
        fprintf('[fold %d] seed %d done (%.0fs)\n', f, s, toc(t0));
    end
    clear Gf
end

hdr = {'method','seed','fold','image','dice','jaccard','sensitivity', ...
       'specificity','precision','hd95','extra'};
fr_writecsv(fullfile(cfg.outDir, 'perimage_all.csv'), hdr, rows);
save(fullfile(cfg.outDir, 'posteriors.mat'), 'postStore', '-v7');
fprintf('DONE in %.0fs -> %s\n', toc(t0), fullfile(cfg.outDir, 'perimage_all.csv'));
