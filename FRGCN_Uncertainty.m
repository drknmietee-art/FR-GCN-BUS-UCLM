%FRGCN_UNCERTAINTY  Is the fuzzy-rough boundary region a usable confidence
%   signal?  (Section 8.7.)
%
%   Three candidate scores are compared on identical predictions: the
%   closed-form descriptor B(p), which costs nothing at inference; the
%   predictive entropy of the same trained model, which also costs nothing;
%   and the standard deviation over 20 Monte-Carlo dropout passes, which
%   costs twenty forward passes.  Every node is weighted by its pixel count
%   so the measures refer to image area rather than to graph nodes.

cfg = fr_config();
pf = fullfile(cfg.outDir, 'posteriors.mat');
if ~exist(pf, 'file')
    error('frgcn:noPosteriors', ['\n\nposteriors.mat not found in\n    %s\n\n' ...
        'Run FRGCN_RunAll first; it writes the stored posteriors this study needs.\n'], cfg.outDir);
end
P = load(pf); PS = P.postStore; clear P
keys = fieldnames(PS);
keys = keys(cellfun(@(k) isfield(PS.(k), 'post') && isfield(PS.(k), 'mc'), keys));
fprintf('%d images with stored posteriors\n', numel(keys));

post = []; y = []; w = []; bnd = []; mc = [];
for i = 1:numel(keys)
    r = PS.(keys{i});
    post = [post; r.post]; y = [y; r.y]; w = [w; r.npix(:)]; %#ok<AGROW>
    bnd  = [bnd;  r.bnd];  mc = [mc; r.mc];                  %#ok<AGROW>
end
pc  = min(max(post, 1e-6), 1 - 1e-6);
ent = -(pc .* log(pc) + (1 - pc) .* log(1 - pc));
err = double((post > 0.5) ~= (y > 0.5));

% ---- calibration ------------------------------------------------------
fr_writecsv(fullfile(cfg.outDir, 'U1_calibration.csv'), {'quantity','value'}, ...
    {'ECE (node posterior, pixel-weighted)', fr_ece(post, y, w); ...
     'Node error rate (pixel-weighted)',     sum(err .* w) / sum(w); ...
     'Mean boundary uncertainty B(p)',       mean(bnd)});

% ---- error detection --------------------------------------------------
names = {'Fuzzy-rough boundary B(p)  [closed form]', ...
         'Predictive entropy         [free]', ...
         'MC-dropout std, T=20       [20 forward passes]'};
scores = {bnd, ent, mc};
det = {};
for i = 1:3
    [a, ap] = fr_auroc(err, scores{i}, w);
    det(end+1,:) = {names{i}, a, ap}; %#ok<SAGROW>
    fprintf('  %-52s AUROC %.4f  AP %.4f\n', names{i}, a, ap);
end
fr_writecsv(fullfile(cfg.outDir, 'U2_error_detection.csv'), ...
    {'score','AUROC','AveragePrecision'}, det);

% ---- selective prediction --------------------------------------------
fracs = [0 .02 .05 .10 .15 .20 .30 .40 .50];
labels = {'B(p)', 'Entropy', 'MC-dropout', 'Random'};
curves = zeros(numel(fracs), 4);
rs = fr_seed(7);
for i = 1:numel(keys)
    r = PS.(keys{i});
    pc2 = min(max(r.post, 1e-6), 1 - 1e-6);
    e2  = -(pc2 .* log(pc2) + (1 - pc2) .* log(1 - pc2));
    [rnd, rs] = fr_rand(rs, numel(r.y), 1);
    sc = {r.bnd, e2, r.mc, rnd};
    for j = 1:4
        curves(:, j) = curves(:, j) + fr_referral(r, sc{j}, fracs) / numel(keys);
    end
end
rows = cell(numel(fracs), 5);
for i = 1:numel(fracs)
    rows(i,:) = {sprintf('%.0f%%', 100*fracs(i)), curves(i,1), curves(i,2), curves(i,3), curves(i,4)};
end
fr_writecsv(fullfile(cfg.outDir, 'U3_referral.csv'), ...
    [{'referred_area_fraction'} labels], rows);

% ---- decision-threshold sweep ----------------------------------------
taus = [0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8];
sw = cell(numel(taus), 5);
for t = 1:numel(taus)
    acc = zeros(numel(keys), 4);
    for i = 1:numel(keys)
        r = PS.(keys{i}); ww = r.npix(:); fr = r.frac(:);
        p = double(r.post > taus(t));
        tp = sum(ww.*p.*fr); fp = sum(ww.*p.*(1-fr));
        fn = sum(ww.*(1-p).*fr); tn = sum(ww.*(1-p).*(1-fr));
        acc(i,:) = [2*tp/max(2*tp+fp+fn,1e-9), tp/max(tp+fn,1e-9), ...
                    tn/max(tn+fp,1e-9), tp/max(tp+fp,1e-9)];
    end
    m = mean(acc, 1);
    sw(t,:) = {taus(t), m(1), m(2), m(3), m(4)};
end
fr_writecsv(fullfile(cfg.outDir, 'U5_threshold.csv'), ...
    {'threshold','Dice','Sensitivity','Specificity','Precision'}, sw);
fprintf('Uncertainty study written to %s\n', cfg.outDir);
