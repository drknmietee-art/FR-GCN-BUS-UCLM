%FRGCN_TABLES  Build every manuscript table from the one results file.
%   Reads results/perimage_all.csv (+ dense_baselines.csv when present) and
%   writes T5..T10, the controls table and the seed-variation table.

cfg = fr_config();
rf = fullfile(cfg.outDir, 'perimage_all.csv');
if ~exist(rf, 'file')
    error('frgcn:noResults', ['\n\nperimage_all.csv not found in\n    %s\n\n' ...
        'Run FRGCN_RunAll first.\n'], cfg.outDir);
end
[hdr, D] = fr_readcsv(rf);
dense = fullfile(cfg.outDir, 'dense_baselines.csv');
if exist(dense, 'file')
    [~, D2] = fr_readcsv(dense); D = [D; D2];
end
col = @(nm) find(strcmp(hdr, nm), 1);
method = D(:, col('method')); seed = cell2mat(D(:, col('seed')));
image_ = D(:, col('image'));
isTrain = ~cellfun(@isempty, strfind(method, '[TRAIN-FOLD]')); %#ok<STRCLFH>
metrics = {'dice','jaccard','sensitivity','specificity','precision','hd95'};

Dtr = D(isTrain, :); D = D(~isTrain, :);
method = method(~isTrain); seed = seed(~isTrain); image_ = image_(~isTrain);
vals = cell2mat(D(:, cellfun(col, metrics)));

% ---- per-image score averaged over seeds: the unit of the paired tests
[keys, ~, gi] = unique(strcat(method, '||', image_));
per = zeros(numel(keys), numel(metrics));
for m = 1:numel(metrics)
    per(:, m) = accumarray(gi, vals(:, m), [], @(v) mean(v(~isnan(v))));
end
pm = cellfun(@(k) {k(1:strfind(k,'||')-1)}, keys);
pi_ = cellfun(@(k) {k(strfind(k,'||')+2:end)}, keys);

um = unique(pm);
getv = @(mth, mi) per(strcmp(pm, mth), mi);
getim = @(mth) pi_(strcmp(pm, mth));

% ---- Table 5: FR-GCN headline ----------------------------------------
rows = {};
for m = 1:5
    x = getv('FR-GCN', m); x = x(~isnan(x));
    ci = fr_bca(x, cfg.nBoot);
    rows(end+1,:) = {metrics{m}, mean(x), std(x), 1.96*std(x)/sqrt(numel(x)), ci(1), ci(2), numel(x)}; %#ok<SAGROW>
end
h = getv('FR-GCN', 6); h = h(~isnan(h)); ci = fr_bca(h, cfg.nBoot);
rows(end+1,:) = {'HD95 (pixels)', mean(h), std(h), 1.96*std(h)/sqrt(numel(h)), ci(1), ci(2), numel(h)};
fr_writecsv(fullfile(cfg.outDir,'T5_headline.csv'), ...
    {'Metric','Mean','Std','CI_half_normal','BCa_lo','BCa_hi','n'}, rows);

% ---- Table 6: comparison ---------------------------------------------
order = {'FCM + threshold','SLIC + SVM','Plain GCN','FR-GCN','FR-GCN-lean', ...
         'MLP (no propagation)','U-Net','nnU-Net-style'};
rows = {};
for i = 1:numel(order)
    if ~any(strcmp(pm, order{i})), continue; end
    r = {order{i}};
    for m = 1:6
        x = getv(order{i}, m); r{end+1} = mean(x(~isnan(x))); %#ok<SAGROW>
    end
    % seed-to-seed SD of the fold-pooled mean
    sel = strcmp(method, order{i}) & seed >= 0;
    if any(sel)
        us = unique(seed(sel)); sm = arrayfun(@(s) mean(vals(sel & seed==s, 1)), us);
        r{end+1} = std(sm); %#ok<SAGROW>
    else
        r{end+1} = NaN; %#ok<SAGROW>
    end
    rows(end+1,:) = r; %#ok<SAGROW>
end
fr_writecsv(fullfile(cfg.outDir,'T6_comparison.csv'), ...
    [{'Method'} metrics {'dice_seed_sd'}], rows);

% ---- Table 7: paired tests vs FR-GCN ---------------------------------
refv = getv('FR-GCN', 1); refi = getim('FR-GCN');
cmp = setdiff(order, {'FR-GCN'}, 'stable'); rows = {}; pw = []; pt = [];
for i = 1:numel(cmp)
    if ~any(strcmp(pm, cmp{i})), continue; end
    [a, b] = fr_alignpair(refi, refv, getim(cmp{i}), getv(cmp{i}, 1));
    pw(end+1) = fr_signrank(a, b); pt(end+1) = fr_ttest_rel(a, b); %#ok<SAGROW>
    dirn = 'FR-GCN better'; if mean(a) < mean(b), dirn = 'FR-GCN worse'; end
    rows(end+1,:) = {['FR-GCN vs ' cmp{i}], pw(end), NaN, pt(end), NaN, dirn}; %#ok<SAGROW>
end
hw = fr_holm(pw); ht = fr_holm(pt);
for i = 1:size(rows,1), rows{i,3} = hw(i); rows{i,5} = ht(i); end
fr_writecsv(fullfile(cfg.outDir,'T7_tests.csv'), ...
    {'Comparison','Wilcoxon_raw','Wilcoxon_Holm','Paired_t_raw','Paired_t_Holm','Direction'}, rows);

% ---- Table 8: ablation ------------------------------------------------
abl = {'FR-GCN','abl: no fuzzy membership','abl: no rough boundary', ...
       'abl: no uncertainty loss','abl: no fuzzy edge weights', ...
       'MLP (no propagation)','FR-GCN-lean'};
full = mean(getv('FR-GCN', 1)); rows = {}; praw = [];
for i = 1:numel(abl)
    if ~any(strcmp(pm, abl{i})), continue; end
    x = getv(abl{i}, 1);
    if strcmp(abl{i}, 'FR-GCN')
        p = NaN;
    else
        [a, b] = fr_alignpair(refi, refv, getim(abl{i}), x);
        p = fr_signrank(a, b); praw(end+1) = p; %#ok<SAGROW>
    end
    sel = strcmp(method, abl{i}) & seed >= 0;
    us = unique(seed(sel)); sm = arrayfun(@(s) mean(vals(sel & seed==s,1)), us);
    rows(end+1,:) = {abl{i}, mean(x), mean(getv(abl{i},2)), mean(x)-full, std(sm), p, NaN}; %#ok<SAGROW>
end
hp = fr_holm(praw); k = 0;
for i = 1:size(rows,1)
    if ~isnan(rows{i,6}), k = k + 1; rows{i,7} = hp(k); end
end
fr_writecsv(fullfile(cfg.outDir,'T8_ablation.csv'), ...
    {'Configuration','Dice','Jaccard','Delta_Dice','Seed_SD','Wilcoxon_vs_full','Wilcoxon_Holm'}, rows);

% ---- Table 9: controls (the decisive comparison) ----------------------
pairs = {'FR-GCN-lean','MLP (no propagation)'; 'FR-GCN','MLP (no propagation)'; ...
         'MLP (no propagation)','Plain GCN';  'FR-GCN-lean','Plain GCN'; ...
         'FR-GCN-lean','FR-GCN';              'MLP (no propagation)','abl: no fuzzy edge weights'};
rows = {};
for i = 1:size(pairs,1)
    if ~any(strcmp(pm,pairs{i,1})) || ~any(strcmp(pm,pairs{i,2})), continue; end
    [a, b] = fr_alignpair(getim(pairs{i,1}), getv(pairs{i,1},1), ...
                          getim(pairs{i,2}), getv(pairs{i,2},1));
    rows(end+1,:) = {pairs{i,1}, pairs{i,2}, mean(a), mean(b), mean(a)-mean(b), ...
                     fr_signrank(a,b), fr_ttest_rel(a,b)}; %#ok<SAGROW>
end
fr_writecsv(fullfile(cfg.outDir,'T9_controls.csv'), ...
    {'A','B','mean_A','mean_B','delta','wilcoxon','paired_t'}, rows);

% ---- Table 10: stratified --------------------------------------------
[mh, M] = fr_readcsv(fullfile(cfg.outDir,'dataset_manifest.csv'));
mn = M(:, find(strcmp(mh,'image'),1)); mt = M(:, find(strcmp(mh,'type'),1));
mp = cell2mat(M(:, find(strcmp(mh,'lesion_pixels'),1)));
[tf, loc] = ismember(getim('FR-GCN'), mn);
d = getv('FR-GCN',1); area = nan(size(d)); typ = repmat({''}, size(d));
area(tf) = mp(loc(tf)); typ(tf) = mt(loc(tf));
med = median(area(~isnan(area)));
S = {'Lesion area below median', d(area <  med); ...
     'Lesion area at or above median', d(area >= med); ...
     'Benign lesions',    d(strcmp(typ,'benign')); ...
     'Malignant lesions', d(strcmp(typ,'malignant'))};
rows = {};
for i = 1:size(S,1)
    v = S{i,2}; rows(end+1,:) = {S{i,1}, numel(v), mean(v), std(v)}; %#ok<SAGROW>
end
rows(end+1,:) = {'Images with an empty prediction', sum(d == 0), 0, 0};
fr_writecsv(fullfile(cfg.outDir,'T10_stratified.csv'), ...
    {'Stratum','Images','Dice_mean','Dice_std'}, rows);

% ---- training fit -----------------------------------------------------
if ~isempty(Dtr)
    tm = Dtr(:, col('method')); td = cell2mat(Dtr(:, col('dice')));
    te_ = cell2mat(Dtr(:, col('extra')));
    ut = unique(tm); rows = {};
    for i = 1:numel(ut)
        s = strcmp(tm, ut{i});
        base = strrep(ut{i}, ' [TRAIN-FOLD]', '');
        tv = getv(base, 1);
        rows(end+1,:) = {base, mean(td(s)), mean(tv(~isnan(tv))), mean(te_(s))}; %#ok<SAGROW>
    end
    fr_writecsv(fullfile(cfg.outDir,'T_trainfit.csv'), ...
        {'Configuration','train_fold_dice','test_fold_dice','selected_epoch'}, rows);
end
fprintf('All tables written to %s\n', cfg.outDir);
