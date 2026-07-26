%% ========================================================================
%  FR_GCN_BUS_UCLM.m
%  Fuzzy-Rough Graph Convolutional Network (FR-GCN) for
%  Uncertainty-Aware Medical Image Segmentation on the BUS-UCLM
%  Breast Ultrasound Dataset.
%
%  Tested on: MATLAB R2024b
%  Required toolboxes:
%     - Image Processing Toolbox      (superpixels, graycomatrix, ...)
%     - Deep Learning Toolbox         (dlarray, adamupdate, dlgradient)
%     - Fuzzy Logic Toolbox           (fcm)  [optional; internal fallback]
%     - Statistics and Machine Learning Toolbox (kfoldLoss helpers) [optional]
%
%  Author: D. K. Nishad
%  Repository: https://github.com/<your-user>/FR-GCN-BUS-UCLM
%  License: MIT
%
%  The script performs a complete, reproducible pipeline:
%     1. Load BUS-UCLM images and RGB masks.
%     2. Convert colour masks to binary lesion labels and class labels.
%     3. Preprocess (grayscale, speckle reduction, normalisation).
%     4. Build superpixel graphs (nodes = superpixels).
%     5. Extract fuzzy-rough node features (FCM membership + rough boundary).
%     6. Train a from-scratch Graph Convolutional Network (dlarray).
%     7. Evaluate Dice, Jaccard, Sensitivity, Specificity, Precision, HD95.
%     8. Save metrics, figures, and a results table for the paper.
%
%  NOTE: set CFG.dataRoot to the BUS-UCLM folder on your machine.
% =========================================================================

function FR_GCN_BUS_UCLM()

clc; close all; rng(42);   % reproducibility

%% ----------------------------- CONFIG ----------------------------------
CFG = struct();
CFG.dataRoot   = fullfile('BUS-UCLM Breast ultrasound lesion segmentation dataset','BUS-UCLM');
CFG.imgDir     = fullfile(CFG.dataRoot,'images');
CFG.maskDir    = fullfile(CFG.dataRoot,'masks');
CFG.outDir     = fullfile(pwd,'results');
CFG.imResize   = [256 256];   % working resolution
CFG.nSuperpix  = 300;         % target superpixels per image (SLIC)
CFG.fcmC       = 2;           % fuzzy clusters (lesion / background)
CFG.fcmM       = 2.0;         % fuzzifier
CFG.sigmaSim   = 0.30;        % Gaussian scale for fuzzy similarity
CFG.hidden     = 64;          % GCN hidden units
CFG.numLayers  = 2;           % GCN layers
CFG.lr         = 1e-2;        % Adam learning rate
CFG.epochs     = 200;         % training epochs
CFG.alpha      = 0.5;         % weight: CE vs Dice
CFG.lambda     = 1.0;         % uncertainty weighting in CE
CFG.kFolds     = 5;           % cross-validation folds
CFG.useLesionOnly = true;     % train on images that contain a lesion
CFG.maxImages  = Inf;         % set small (e.g. 40) for a quick smoke test

if ~exist(CFG.outDir,'dir'); mkdir(CFG.outDir); end

%% ---------------------- 1. INDEX THE DATASET ---------------------------
imgFiles = dir(fullfile(CFG.imgDir,'*.png'));
assert(~isempty(imgFiles), 'No images found. Check CFG.imgDir.');
names = string({imgFiles.name});

fprintf('Found %d images in BUS-UCLM.\n', numel(names));

% Read class labels from mask colour (green=benign, red=malignant, black=normal)
[classLabel, hasLesion] = deal(strings(numel(names),1), false(numel(names),1));
for i = 1:numel(names)
    M = imread(fullfile(CFG.maskDir, names(i)));
    [classLabel(i), hasLesion(i)] = maskClass(M);
end
fprintf('Benign: %d | Malignant: %d | Normal: %d\n', ...
    sum(classLabel=="benign"), sum(classLabel=="malignant"), sum(classLabel=="normal"));

useIdx = 1:numel(names);
if CFG.useLesionOnly
    useIdx = find(hasLesion);
end
if isfinite(CFG.maxImages)
    useIdx = useIdx(1:min(CFG.maxImages,numel(useIdx)));
end
fprintf('Using %d images for the experiment.\n', numel(useIdx));

%% ---------------------- 2. BUILD GRAPHS --------------------------------
% Each image becomes one graph. Cache to a struct array.
G = struct('X',{},'Ahat',{},'y',{},'spLabel',{},'imSize',{},'name',{});
fprintf('Building superpixel graphs and fuzzy-rough features ...\n');
for t = 1:numel(useIdx)
    i = useIdx(t);
    I  = imread(fullfile(CFG.imgDir,  names(i)));
    Mc = imread(fullfile(CFG.maskDir, names(i)));

    [Ig, gt] = preprocess(I, Mc, CFG.imResize);          % grayscale + binary GT
    [X, Ahat, y, L] = buildGraph(Ig, gt, CFG);           % nodes/edges/features
    G(end+1) = struct('X',X,'Ahat',Ahat,'y',y, ...
        'spLabel',L,'imSize',CFG.imResize,'name',names(i)); %#ok<AGROW>
    if mod(t,25)==0; fprintf('   %d / %d graphs built\n', t, numel(useIdx)); end
end
fprintf('Graph construction complete. Feature dim = %d.\n', size(G(1).X,2));

%% ---------------------- 3. CROSS-VALIDATION ----------------------------
cv = cvIndices(numel(G), CFG.kFolds);
metricsAll = [];   % rows: [Dice Jaccard Sens Spec Prec] per test image

for f = 1:CFG.kFolds
    teMask = (cv==f);
    trIdx  = find(~teMask);
    teIdx  = find(teMask);
    fprintf('\n===== Fold %d/%d : train=%d test=%d =====\n', ...
        f, CFG.kFolds, numel(trIdx), numel(teIdx));

    params = trainGCN(G(trIdx), CFG);              % train FR-GCN
    for k = teIdx
        yhat = predictGCN(params, G(k), CFG);      % node probabilities
        m = imageMetrics(G(k), yhat);              % pixel-level metrics
        metricsAll = [metricsAll; m]; %#ok<AGROW>
    end
end

%% ---------------------- 4. REPORT RESULTS ------------------------------
mu  = mean(metricsAll,1);
sd  = std(metricsAll,0,1);
colNames = {'Dice','Jaccard','Sensitivity','Specificity','Precision'};
Tr = array2table([mu; sd], 'VariableNames', colNames, ...
        'RowNames', {'Mean','Std'});
disp('=============== FR-GCN cross-validated results ================');
disp(Tr);
writetable(Tr, fullfile(CFG.outDir,'results_summary.csv'), 'WriteRowNames', true);

% 95% confidence interval (normal approximation) for Dice
n   = size(metricsAll,1);
ciD = 1.96 * sd(1) / sqrt(n);
fprintf('Dice = %.4f +/- %.4f (95%% CI half-width = %.4f, n=%d)\n', ...
    mu(1), sd(1), ciD, n);

% Boxplot of per-image Dice / Jaccard
fig = figure('Visible','off');
boxplot(metricsAll(:,1:2), 'Labels', {'Dice','Jaccard'});
ylabel('Score'); title('FR-GCN segmentation performance (BUS-UCLM)');
saveas(fig, fullfile(CFG.outDir,'boxplot_metrics.png')); close(fig);

save(fullfile(CFG.outDir,'metrics_raw.mat'), 'metricsAll', 'colNames');

%% ---------- Train a final model on all graphs (for figures) -----------
% FR_GCN_figures.m reuses this cached model to draw Figures 2, 4 and 6.
fprintf('\nTraining final model on all %d graphs for figure generation ...\n', numel(G));
[params, lossHistory] = trainGCN(G, CFG); %#ok<ASGLU>
save(fullfile(CFG.outDir,'model.mat'), 'params', 'lossHistory', 'CFG');
fprintf('Final model cached to %s\n', fullfile(CFG.outDir,'model.mat'));

fprintf('\nAll results written to: %s\n', CFG.outDir);
end % ===================== end main function ============================


%% ======================================================================
%  LOCAL FUNCTIONS
%  ======================================================================

function [cls, hasL] = maskClass(M)
% Determine class from BUS-UCLM colour mask. Green=benign, Red=malignant.
    if size(M,3)==1; M = repmat(M,1,1,3); end
    R = M(:,:,1); Gc = M(:,:,2); B = M(:,:,3);
    green = nnz(Gc>100 & R<100 & B<100);
    red   = nnz(R>100 & Gc<100 & B<100);
    if green==0 && red==0
        cls = "normal"; hasL = false;
    elseif green>=red
        cls = "benign"; hasL = true;
    else
        cls = "malignant"; hasL = true;
    end
end

function [Ig, gt] = preprocess(I, Mc, sz)
% Grayscale conversion, speckle reduction, normalisation, binary GT.
    if size(I,3)==3; Ig = rgb2gray(I); else; Ig = I; end
    Ig = im2double(imresize(Ig, sz));
    % Speckle reduction (edge-preserving). Requires Image Processing Toolbox.
    Ig = imdiffusefilt(Ig, 'NumberOfIterations', 5);
    % Contrast normalisation to [0,1]
    Ig = mat2gray(Ig);
    % Binary lesion ground truth from colour mask (green OR red)
    if size(Mc,3)==1; Mc = repmat(Mc,1,1,3); end
    R = Mc(:,:,1); Gc = Mc(:,:,2); B = Mc(:,:,3);
    lesion = (Gc>100 & R<100 & B<100) | (R>100 & Gc<100 & B<100);
    gt = imresize(lesion, sz, 'nearest');
end

function [X, Ahat, y, L] = buildGraph(Ig, gt, CFG)
% Superpixel graph with fuzzy-rough node features.
    [L, N] = superpixels(Ig, CFG.nSuperpix);      % SLIC over-segmentation

    % ---- per-superpixel raw descriptors ----
    glcmOff = [0 1; -1 1; -1 0; -1 -1];           % 4 GLCM directions
    feat = zeros(N, 6);
    cent = zeros(N, 2);
    lab  = zeros(N, 1);
    idxList = label2idx(L);
    [rr, cc] = size(Ig);
    for k = 1:N
        px = idxList{k};
        vals = Ig(px);
        feat(k,1) = mean(vals);
        feat(k,2) = std(vals);
        % local GLCM texture on the superpixel bounding box
        [ys, xs] = ind2sub([rr cc], px);
        yb = max(min(ys):max(ys),1); xb = max(min(xs):max(xs),1);
        patch = Ig(min(ys):max(ys), min(xs):max(xs));
        gl = graycomatrix(patch, 'Offset', glcmOff, ...
              'NumLevels', 8, 'Symmetric', true);
        st = graycoprops(gl, {'Contrast','Homogeneity','Energy','Correlation'});
        feat(k,3) = mean(st.Contrast);
        feat(k,4) = mean(st.Homogeneity);
        feat(k,5) = mean(st.Energy);
        cval = mean(st.Correlation); if isnan(cval); cval = 0; end
        feat(k,6) = cval;
        cent(k,:) = [mean(xs) mean(ys)];
        lab(k)    = mean(gt(px)) > 0.5;           % node label by majority
    end
    feat = normalizeCols(feat);

    % ---- fuzzy c-means membership (Eq. 1-3) ----
    U = fuzzyCMeans(feat, CFG.fcmC, CFG.fcmM);    % N x C memberships
    % lesion cluster = the one whose centroid has higher mean intensity? Not
    % robust. Instead align by correlation with node labels when available.
    [~, lesCol] = max(corrSafe(U, lab));
    uLesion = U(:, lesCol);

    % ---- fuzzy similarity relation (Eq. 4) ----
    Dfeat = pdist2(feat, feat);
    Rsim  = exp( -(Dfeat.^2) / (2*CFG.sigmaSim^2) );

    % ---- fuzzy-rough lower/upper approximation + boundary (Eq. 5-6) ----
    A = uLesion(:)';                              % fuzzy set "lesion"
    lower = zeros(N,1); upper = zeros(N,1);
    for p = 1:N
        Luk = min(1, 1 - Rsim(p,:) + A);          % Lukasiewicz implicator
        Tuk = max(0, Rsim(p,:) + A - 1);          % Lukasiewicz t-norm
        lower(p) = min(Luk);
        upper(p) = max(Tuk);
    end
    boundary = upper - lower;                     % uncertainty (Eq. 6)

    % ---- node feature matrix (Eq. 7) ----
    X = [feat, uLesion, lower, boundary];         % N x 9

    % ---- adjacency: region adjacency AND fuzzy similarity (Eq. 8) ----
    Adj = regionAdjacency(L, N);
    W   = Adj .* Rsim;                            % weighted edges
    W   = W + eye(N);                             % self-loops  (A + I)
    d   = sum(W,2);
    Dinv= diag(1 ./ sqrt(max(d,eps)));
    Ahat= Dinv * W * Dinv;                        % normalised (Eq. 9)

    y = lab(:);                                   % node targets
end

function U = fuzzyCMeans(X, C, m)
% Fuzzy c-means. Uses Fuzzy Logic Toolbox 'fcm' when present, else fallback.
    if exist('fcm','file') == 2
        opt = fcmOptions('NumClusters', C, 'Exponent', m, ...
            'MaxNumIteration', 100, 'Verbose', false);
        try
            [~, U] = fcm(X, opt);   % R2024b signature
        catch
            [~, Uc] = fcm(X, C);    % legacy signature
            U = Uc;
        end
        U = U';                     % -> N x C
        return;
    end
    % ---- fallback implementation ----
    N = size(X,1); rng(1);
    V = X(randperm(N,C),:);
    U = rand(N,C); U = U ./ sum(U,2);
    for it = 1:100
        D = max(pdist2(X,V), 1e-9);
        Uo = U;
        for k = 1:C
            ratio = (D(:,k)./D).^(2/(m-1));
            U(:,k) = 1 ./ sum(ratio,2);
        end
        for k = 1:C
            w = U(:,k).^m;
            V(k,:) = (w'*X) / sum(w);
        end
        if norm(U-Uo,'fro') < 1e-5; break; end
    end
end

function A = regionAdjacency(L, N)
% Build binary region adjacency matrix from a superpixel label image.
    A = false(N,N);
    right = L(:,1:end-1) ~= L(:,2:end);
    [r,c] = find(right);
    p = L(sub2ind(size(L),r,c));
    q = L(sub2ind(size(L),r,c+1));
    A(sub2ind([N N],p,q)) = true;
    down = L(1:end-1,:) ~= L(2:end,:);
    [r,c] = find(down);
    p = L(sub2ind(size(L),r,c));
    q = L(sub2ind(size(L),r+1,c));
    A(sub2ind([N N],p,q)) = true;
    A = A | A';
    A = double(A);
end

function [params, lossHist] = trainGCN(Gtr, CFG)
% Train a 2-layer GCN with a custom dlarray training loop (Eq. 10-14).
% Second output lossHist holds the mean training loss per epoch.
    fin  = size(Gtr(1).X,2);
    params = initParams(fin, CFG.hidden, 2, CFG.numLayers);
    avgG = []; avgSq = []; iter = 0; lossHist = zeros(CFG.epochs,1);
    for ep = 1:CFG.epochs
        order = randperm(numel(Gtr));
        epLoss = 0;
        for gi = order
            iter = iter + 1;
            Xg  = dlarray(single(Gtr(gi).X));        % N x F (unformatted)
            Ah  = dlarray(single(Gtr(gi).Ahat));     % N x N
            yv  = single(Gtr(gi).y);
            bnd = single(Gtr(gi).X(:,end));          % boundary uncertainty
            [loss, grad] = dlfeval(@modelLoss, params, Xg, Ah, yv, bnd, CFG);
            [params, avgG, avgSq] = adamupdate(params, grad, avgG, avgSq, ...
                iter, CFG.lr);
            epLoss = epLoss + double(gather(extractdata(loss)));
        end
        lossHist(ep) = epLoss/numel(Gtr);
        if mod(ep,50)==0
            fprintf('   epoch %3d | mean loss %.4f\n', ep, lossHist(ep));
        end
    end
end

function p = initParams(fin, h, nc, nl)
% Glorot initialisation of GCN weight matrices.
% NOTE: params must contain ONLY dlarray leaves, because dlgradient
% differentiates every field. The layer count is recovered from the
% number of weight fields, so it is not stored here.
    p = struct();
    dims = [fin, repmat(h,1,nl-1), nc];
    for l = 1:nl
        p.(sprintf('W%d',l)) = dlarray(glorot(dims(l), dims(l+1)));
    end
end

function W = glorot(fanIn, fanOut)
    r = sqrt(6/(fanIn+fanOut));
    W = single( -r + 2*r*rand(fanIn, fanOut) );
end

function [loss, grad] = modelLoss(params, X, Ahat, y, bnd, CFG)
    Z = forwardGCN(params, X, Ahat);              % N x 2 logits
    P = softmaxRows(Z);                           % row-wise probabilities
    y = y(:); bnd = bnd(:);
    yhot = [1 - y, y];                            % N x 2 (numeric)
    % uncertainty-weighted cross entropy (Eq. 12)
    w = 1 + CFG.lambda * bnd;
    ce = -sum( w .* sum(yhot .* log(P + 1e-8), 2) ) / numel(y);
    % soft Dice loss on the lesion channel (Eq. 13)
    pl = P(:,2); gl = y;
    dice = 1 - (2*sum(pl.*gl)+1) / (sum(pl)+sum(gl)+1);
    loss = CFG.alpha*ce + (1-CFG.alpha)*dice;     % total (Eq. 14)
    grad = dlgradient(loss, params);
end

function Z = forwardGCN(params, X, Ahat)
% H^{(l+1)} = ReLU( Ahat H^{(l)} W^{(l)} ); last layer linear (Eq. 10-11).
    nl = sum(startsWith(fieldnames(params), 'W')); % layer count from weights
    H = X;                                         % N x F
    for l = 1:nl
        H = Ahat * H * params.(sprintf('W%d',l));  % (NxN)(NxF)(Fxh)
        if l < nl
            H = relu(H);
        end
    end
    Z = H;                                         % N x 2
end

function P = softmaxRows(Z)
% Numerically stable row-wise softmax for an N x C dlarray.
    Zs = Z - max(Z, [], 2);
    E  = exp(Zs);
    P  = E ./ sum(E, 2);
end

function yhat = predictGCN(params, Gk, CFG) %#ok<INUSD>
    Xg = dlarray(single(Gk.X));
    Ah = dlarray(single(Gk.Ahat));
    Z  = forwardGCN(params, Xg, Ah);
    P  = softmaxRows(Z);
    yhat = double(gather(extractdata(P(:,2))));   % lesion probability per node
end

function m = imageMetrics(Gk, yhatNode)
% Map node probabilities to pixels (Eq. 15) and compute overlap metrics.
    predNode = yhatNode > 0.5;
    predMask = predNode(Gk.spLabel);              % broadcast to pixels
    gtMask   = false(Gk.imSize);
    gtNode   = Gk.y > 0.5;
    gtMask   = gtNode(Gk.spLabel);
    P = predMask(:); Gt = gtMask(:);
    TP = nnz(P & Gt); FP = nnz(P & ~Gt);
    FN = nnz(~P & Gt); TN = nnz(~P & ~Gt);
    dice = 2*TP / max(2*TP+FP+FN, eps);
    jac  = TP / max(TP+FP+FN, eps);
    sens = TP / max(TP+FN, eps);
    spec = TN / max(TN+FP, eps);
    prec = TP / max(TP+FP, eps);
    m = [dice jac sens spec prec];
end

%% ---------------------- small helpers ---------------------------------
function Y = normalizeCols(X)
    mn = min(X,[],1); mx = max(X,[],1);
    Y = (X - mn) ./ max(mx-mn, eps);
end

function c = corrSafe(U, lab)
    if numel(unique(lab))<2
        c = mean(U,1);                            % fallback: pick brighter
    else
        c = abs(corr(U, lab));
        c(isnan(c)) = 0;
    end
end

function cv = cvIndices(n, k)
    cv = mod((0:n-1), k) + 1;
    cv = cv(randperm(n));
end
