%% ========================================================================
%  run_baselines.m
%  Baseline comparison and statistical tests for the FR-GCN paper.
%  All methods are evaluated on the 264 lesion images of BUS-UCLM under
%  one shared five-fold cross-validation split.
%
%  Methods
%    1. FCM + threshold    (unsupervised pixel clustering)
%    2. SLIC + SVM         (superpixel features, RBF SVM)
%    3. Plain GCN          (6-D texture features, binary graph, CE loss)
%    4. FR-GCN (proposed)  (9-D fuzzy-rough features, fuzzy graph, unc. loss)
%
%  Also runs an ablation of the proposed model.
%
%  Outputs written to results/
%    comparison.csv   methods x [Dice Jaccard Sensitivity Specificity Precision]
%    stats.csv        FR-GCN vs each baseline: Wilcoxon and paired t p-values
%    ablation.csv     configuration x [Dice Jaccard]
%    perimage_dice.mat  per-image Dice for every method (for the tests)
%
%  Tested on MATLAB R2024b. Toolboxes: Image Processing, Deep Learning,
%  Statistics and Machine Learning (fitcsvm, signrank), Fuzzy Logic (opt.).
% =========================================================================

function run_baselines()

clc; close all; rng(42);

%% --------------------------- CONFIG ------------------------------------
CFG = struct();
CFG.dataRoot   = fullfile('BUS-UCLM Breast ultrasound lesion segmentation dataset','BUS-UCLM');
CFG.imgDir     = fullfile(CFG.dataRoot,'images');
CFG.maskDir    = fullfile(CFG.dataRoot,'masks');
CFG.outDir     = fullfile(pwd,'results');
CFG.imResize   = [256 256];
CFG.nSuperpix  = 300;
CFG.fcmC       = 2;
CFG.fcmM       = 2.0;
CFG.sigmaSim   = 0.30;
CFG.hidden     = 64;
CFG.numLayers  = 2;
CFG.lr         = 1e-2;
CFG.epochs     = 150;    % reduce for a quick run; raise for final numbers
CFG.alpha      = 0.5;
CFG.lambda     = 1.0;
CFG.kFolds     = 5;
CFG.maxImages  = Inf;    % set small (e.g. 40) for a smoke test
if ~exist(CFG.outDir,'dir'); mkdir(CFG.outDir); end

%% ---------------------- INDEX LESION IMAGES ----------------------------
imgFiles = dir(fullfile(CFG.imgDir,'*.png'));
names = string({imgFiles.name});
hasL = false(numel(names),1);
for i = 1:numel(names)
    [~,hasL(i)] = maskClass(imread(fullfile(CFG.maskDir,names(i))));
end
useNames = names(hasL);
if isfinite(CFG.maxImages)
    useNames = useNames(1:min(CFG.maxImages,numel(useNames)));
end
fprintf('Using %d lesion images.\n', numel(useNames));

%% ---------------------- BUILD GRAPH CACHE ------------------------------
fprintf('Building graph cache ...\n');
G = struct('X9',{},'Af',{},'Ab',{},'y',{},'L',{},'name',{});
for t = 1:numel(useNames)
    [Ig,gt] = preprocess(imread(fullfile(CFG.imgDir,useNames(t))), ...
                         imread(fullfile(CFG.maskDir,useNames(t))), CFG.imResize);
    [X9,Af,Ab,y,L] = buildGraphFull(Ig,gt,CFG);
    G(end+1) = struct('X9',X9,'Af',Af,'Ab',Ab,'y',y,'L',L,'name',useNames(t)); %#ok<AGROW>
    if mod(t,25)==0; fprintf('   %d / %d\n',t,numel(useNames)); end
end
nImg = numel(G);
cv = mod(0:nImg-1, CFG.kFolds) + 1; cv = cv(randperm(nImg));

%% ---------------------- METRIC CONTAINERS ------------------------------
methods = {'FCM_threshold','SLIC_SVM','Plain_GCN','FR_GCN'};
Mmet = struct();                 % per-method metric rows
for k = 1:numel(methods); Mmet.(methods{k}) = []; end
diceByImg = nan(nImg, numel(methods));   % per-image Dice for the tests
imgOrder  = strings(nImg,1);

%% ---------------------- 1) FCM + THRESHOLD -----------------------------
fprintf('\n[1] FCM + threshold (unsupervised) ...\n');
for i = 1:nImg
    [Ig,gt] = preprocess(imread(fullfile(CFG.imgDir,G(i).name)), ...
                         imread(fullfile(CFG.maskDir,G(i).name)), CFG.imResize);
    pred = fcmThreshold(Ig, CFG);
    m = pixelMetrics(pred, gt);
    Mmet.FCM_threshold = [Mmet.FCM_threshold; m];
    diceByImg(i,1) = m(1); imgOrder(i) = G(i).name;
end

%% ---------------------- CV FOR SUPERVISED METHODS ----------------------
fprintf('\n[2-4] Supervised methods over %d folds ...\n', CFG.kFolds);
for f = 1:CFG.kFolds
    teIdx = find(cv==f); trIdx = find(cv~=f);
    fprintf('  Fold %d/%d (train %d, test %d)\n', f, CFG.kFolds, numel(trIdx), numel(teIdx));

    % ---- 2) SLIC + SVM ----
    if exist('fitcsvm','file')==2
        Xtr = []; Ytr = [];
        for i = trIdx; Xtr=[Xtr; G(i).X9(:,1:6)]; Ytr=[Ytr; G(i).y]; end %#ok<AGROW>
        mdl = fitcsvm(Xtr, Ytr, 'KernelFunction','rbf', ...
                      'Standardize',true, 'KernelScale','auto');
        for i = teIdx
            yp = predict(mdl, G(i).X9(:,1:6));
            m = nodeToPixelMetrics(yp>0.5, G(i));
            Mmet.SLIC_SVM = [Mmet.SLIC_SVM; m]; diceByImg(i,2)=m(1);
        end
    else
        warning('fitcsvm not found. Skipping SLIC + SVM.');
    end

    % ---- 3) Plain GCN : 6-D features, binary graph, plain CE ----
    p3 = trainGCNgeneric(G(trIdx), CFG, 'base6', 'bin', false);
    for i = teIdx
        yp = predictGCNgeneric(p3, G(i), 'base6', 'bin');
        m = nodeToPixelMetrics(yp>0.5, G(i));
        Mmet.Plain_GCN = [Mmet.Plain_GCN; m]; diceByImg(i,3)=m(1);
    end

    % ---- 4) FR-GCN : 9-D features, fuzzy graph, uncertainty loss ----
    p4 = trainGCNgeneric(G(trIdx), CFG, 'full9', 'fuzzy', true);
    for i = teIdx
        yp = predictGCNgeneric(p4, G(i), 'full9', 'fuzzy');
        m = nodeToPixelMetrics(yp>0.5, G(i));
        Mmet.FR_GCN = [Mmet.FR_GCN; m]; diceByImg(i,4)=m(1);
    end
end

%% ---------------------- COMPARISON TABLE -------------------------------
colNames = {'Dice','Jaccard','Sensitivity','Specificity','Precision'};
rows = {}; data = [];
for k = 1:numel(methods)
    Mk = Mmet.(methods{k});
    if isempty(Mk); continue; end
    rows{end+1} = methods{k}; %#ok<AGROW>
    data = [data; mean(Mk,1)]; %#ok<AGROW>
end
Tcmp = array2table(data,'VariableNames',colNames,'RowNames',rows);
disp('================ Comparison (mean over test images) ================');
disp(Tcmp);
writetable(Tcmp, fullfile(CFG.outDir,'comparison.csv'), 'WriteRowNames', true);

%% ---------------------- STATISTICAL TESTS ------------------------------
% FR-GCN (col 4) vs each baseline on per-image Dice.
fprintf('\nPaired tests on per-image Dice (FR-GCN vs baseline)\n');
srows = {}; sdata = {};
ref = diceByImg(:,4);
for k = 1:3
    d = diceByImg(:,k);
    ok = ~isnan(d) & ~isnan(ref);
    if nnz(ok) < 5; continue; end
    pW = signrank(ref(ok), d(ok));           % Wilcoxon signed-rank
    [~,pT] = ttest(ref(ok), d(ok));          % paired t-test
    sig = (min(pW,pT) < 0.05);
    srows{end+1} = sprintf('FR-GCN vs %s', strrep(methods{k},'_',' ')); %#ok<AGROW>
    sdata(end+1,:) = {pW, pT, mat2str(sig)}; %#ok<AGROW>
    fprintf('  %-22s Wilcoxon p=%.4g | paired-t p=%.4g | sig=%d\n', ...
        methods{k}, pW, pT, sig);
end
Tsta = cell2table(sdata, 'VariableNames', {'Wilcoxon_p','Paired_t_p','Significant_0p05'}, ...
                  'RowNames', srows);
writetable(Tsta, fullfile(CFG.outDir,'stats.csv'), 'WriteRowNames', true);
save(fullfile(CFG.outDir,'perimage_dice.mat'),'diceByImg','methods','imgOrder');

%% ---------------------- ABLATION STUDY ---------------------------------
fprintf('\nAblation of the proposed model ...\n');
ablConfigs = {
    'Full FR-GCN',                     'full9','fuzzy', true
    'Without fuzzy membership',        'nomem','fuzzy', true
    'Without rough boundary',          'nobnd','fuzzy', false
    'Without uncertainty-weighted loss','full9','fuzzy', false
    'Without fuzzy edge weights',      'full9','bin',   true };
Tabl = []; ablRows = {};
for c = 1:size(ablConfigs,1)
    dsc = zeros(nImg,2); mask = false(nImg,1);
    for f = 1:CFG.kFolds
        teIdx = find(cv==f); trIdx = find(cv~=f);
        p = trainGCNgeneric(G(trIdx), CFG, ablConfigs{c,2}, ablConfigs{c,3}, ablConfigs{c,4});
        for i = teIdx
            yp = predictGCNgeneric(p, G(i), ablConfigs{c,2}, ablConfigs{c,3});
            m = nodeToPixelMetrics(yp>0.5, G(i));
            dsc(i,:) = m(1:2); mask(i) = true;
        end
    end
    ablRows{end+1} = ablConfigs{c,1}; %#ok<AGROW>
    Tabl = [Tabl; mean(dsc(mask,:),1)]; %#ok<AGROW>
    fprintf('  %-34s Dice=%.4f Jaccard=%.4f\n', ablConfigs{c,1}, mean(dsc(mask,1)), mean(dsc(mask,2)));
end
Tab = array2table(Tabl,'VariableNames',{'Dice','Jaccard'},'RowNames',ablRows);
writetable(Tab, fullfile(CFG.outDir,'ablation.csv'), 'WriteRowNames', true);

fprintf('\nAll baseline outputs written to: %s\n', CFG.outDir);
fprintf('Send comparison.csv, stats.csv and ablation.csv to fill Tables 6, 7 and the ablation table.\n');
end % ===================== end main function ============================


%% ======================================================================
%  METHOD-SPECIFIC HELPERS
%  ======================================================================

function pred = fcmThreshold(Ig, CFG)
% Unsupervised: cluster pixel intensity into 2, lesion = darker cluster.
    v = Ig(:);
    U = fuzzyCMeans(v, CFG.fcmC, CFG.fcmM);
    cen = zeros(CFG.fcmC,1);
    for k = 1:CFG.fcmC; cen(k) = sum(U(:,k).*v)/sum(U(:,k)); end
    [~,darkK] = min(cen);                 % hypoechoic lesion assumption
    [~,lab] = max(U,[],2);
    pred = reshape(lab==darkK, size(Ig));
    pred = imopen(pred, strel('disk',2)); % light cleanup
end

function m = nodeToPixelMetrics(predNode, Gi)
    predMask = predNode(Gi.L);
    gtMask   = (Gi.y>0.5); gtMask = gtMask(Gi.L);
    m = pixelMetrics(predMask, gtMask);
end

function m = pixelMetrics(P, G)
    P = logical(P(:)); G = logical(G(:));
    TP=nnz(P&G); FP=nnz(P&~G); FN=nnz(~P&G); TN=nnz(~P&~G);
    dice=2*TP/max(2*TP+FP+FN,eps); jac=TP/max(TP+FP+FN,eps);
    sens=TP/max(TP+FN,eps); spec=TN/max(TN+FP,eps); prec=TP/max(TP+FP,eps);
    m = [dice jac sens spec prec];
end

function F = selectFeatures(X9, mode)
    switch mode
        case 'full9'; F = X9;
        case 'base6'; F = X9(:,1:6);
        case 'nomem'; F = X9(:,[1:6 8 9]);   % drop fuzzy membership (col 7)
        case 'nobnd'; F = X9(:,1:8);          % drop boundary (col 9)
        otherwise;    F = X9;
    end
end

function A = selectAdj(Gi, mode)
    if strcmp(mode,'bin'); A = Gi.Ab; else; A = Gi.Af; end
end

%% ---------------------- GCN (generic) ---------------------------------

function params = trainGCNgeneric(Gtr, CFG, featMode, adjMode, useUnc)
    fin = size(selectFeatures(Gtr(1).X9, featMode),2);
    params = initParams(fin, CFG.hidden, 2, CFG.numLayers);
    avgG=[]; avgSq=[]; iter=0;
    for ep = 1:CFG.epochs
        for gi = randperm(numel(Gtr))
            iter = iter + 1;
            Fm = selectFeatures(Gtr(gi).X9, featMode);
            Xg = dlarray(single(Fm));
            Ah = dlarray(single(selectAdj(Gtr(gi), adjMode)));
            yv = single(Gtr(gi).y);
            if useUnc; bnd = single(Gtr(gi).X9(:,9)); else; bnd = zeros(size(yv),'single'); end
            [~,grad] = dlfeval(@modelLoss, params, Xg, Ah, yv, bnd, CFG, useUnc);
            [params,avgG,avgSq] = adamupdate(params,grad,avgG,avgSq,iter,CFG.lr);
        end
    end
end

function yhat = predictGCNgeneric(params, Gi, featMode, adjMode)
    Fm = selectFeatures(Gi.X9, featMode);
    Z = forwardGCN(params, dlarray(single(Fm)), dlarray(single(selectAdj(Gi,adjMode))));
    P = softmaxRows(Z);
    yhat = double(gather(extractdata(P(:,2))));
end

function [loss, grad] = modelLoss(params, X, Ahat, y, bnd, CFG, useUnc)
    Z = forwardGCN(params, X, Ahat); P = softmaxRows(Z);
    y = y(:); bnd = bnd(:); yhot = [1-y, y];
    if useUnc; w = 1 + CFG.lambda*bnd; else; w = ones(numel(y),1); end
    ce = -sum(w .* sum(yhot .* log(P+1e-8),2))/numel(y);
    pl = P(:,2); dice = 1 - (2*sum(pl.*y)+1)/(sum(pl)+sum(y)+1);
    loss = CFG.alpha*ce + (1-CFG.alpha)*dice;
    grad = dlgradient(loss, params);
end

function Z = forwardGCN(params, X, Ahat)
    nl = sum(startsWith(fieldnames(params),'W')); H = X;
    for l = 1:nl
        H = Ahat * H * params.(sprintf('W%d',l));
        if l < nl; H = relu(H); end
    end
    Z = H;
end

function P = softmaxRows(Z)
    Zs = Z - max(Z,[],2); E = exp(Zs); P = E ./ sum(E,2);
end

function p = initParams(fin, h, nc, nl)
    p = struct(); dims = [fin, repmat(h,1,nl-1), nc];
    for l = 1:nl; p.(sprintf('W%d',l)) = dlarray(glorot(dims(l),dims(l+1))); end
end

function W = glorot(a,b)
    r = sqrt(6/(a+b)); W = single(-r + 2*r*rand(a,b));
end

%% ---------------------- SHARED PIPELINE -------------------------------

function [cls, hasL] = maskClass(M)
    if size(M,3)==1; M = repmat(M,1,1,3); end
    R=M(:,:,1); Gc=M(:,:,2); B=M(:,:,3);
    green=nnz(Gc>100 & R<100 & B<100); red=nnz(R>100 & Gc<100 & B<100);
    if green==0 && red==0; cls="normal"; hasL=false;
    elseif green>=red; cls="benign"; hasL=true;
    else; cls="malignant"; hasL=true; end
end

function [Ig, gt] = preprocess(I, Mc, sz)
    if size(I,3)==3; Ig=rgb2gray(I); else; Ig=I; end
    Ig = im2double(imresize(Ig,sz));
    Ig = imdiffusefilt(Ig,'NumberOfIterations',5);
    Ig = mat2gray(Ig);
    if size(Mc,3)==1; Mc=repmat(Mc,1,1,3); end
    R=Mc(:,:,1); Gc=Mc(:,:,2); B=Mc(:,:,3);
    lesion = (Gc>100 & R<100 & B<100) | (R>100 & Gc<100 & B<100);
    gt = imresize(lesion, sz, 'nearest');
end

function [X9, Af, Ab, y, L] = buildGraphFull(Ig, gt, CFG)
    [L,N] = superpixels(Ig, CFG.nSuperpix);
    glcmOff = [0 1; -1 1; -1 0; -1 -1];
    feat = zeros(N,6); lab = zeros(N,1);
    idxList = label2idx(L); [rr,cc] = size(Ig);
    for k = 1:N
        px = idxList{k}; vals = Ig(px);
        feat(k,1)=mean(vals); feat(k,2)=std(vals);
        [ys,xs]=ind2sub([rr cc],px);
        patch = Ig(min(ys):max(ys), min(xs):max(xs));
        gl = graycomatrix(patch,'Offset',glcmOff,'NumLevels',8,'Symmetric',true);
        st = graycoprops(gl,{'Contrast','Homogeneity','Energy','Correlation'});
        feat(k,3)=mean(st.Contrast); feat(k,4)=mean(st.Homogeneity);
        feat(k,5)=mean(st.Energy);
        cv=mean(st.Correlation); if isnan(cv); cv=0; end; feat(k,6)=cv;
        lab(k)=mean(gt(px))>0.5;
    end
    feat = normalizeCols(feat);
    U = fuzzyCMeans(feat, CFG.fcmC, CFG.fcmM);
    [~,lesCol] = max(corrSafe(U,lab)); uLesion = U(:,lesCol);
    Dfeat = pdist2(feat,feat); Rsim = exp(-(Dfeat.^2)/(2*CFG.sigmaSim^2));
    A = uLesion(:)'; lower=zeros(N,1); upper=zeros(N,1);
    for p=1:N
        lower(p)=min(min(1,1-Rsim(p,:)+A));
        upper(p)=max(max(0,Rsim(p,:)+A-1));
    end
    boundary = upper - lower;
    X9 = [feat, uLesion, lower, boundary];
    Adj = regionAdjacency(L,N);
    Wf = Adj.*Rsim + eye(N); Af = normAdj(Wf);
    Wb = Adj + eye(N);       Ab = normAdj(Wb);
    y = lab(:);
end

function Ah = normAdj(W)
    d = sum(W,2); Dinv = diag(1./sqrt(max(d,eps))); Ah = Dinv*W*Dinv;
end

function U = fuzzyCMeans(X, C, m)
    if exist('fcm','file')==2
        try
            opt = fcmOptions('NumClusters',C,'Exponent',m,'MaxNumIteration',100,'Verbose',false);
            [~,U] = fcm(X,opt);
        catch
            [~,U] = fcm(X,C);
        end
        U = U'; return;
    end
    N=size(X,1); rng(1); V=X(randperm(N,C),:); U=rand(N,C); U=U./sum(U,2);
    for it=1:100
        D=max(pdist2(X,V),1e-9); Uo=U;
        for k=1:C; U(:,k)=1./sum((D(:,k)./D).^(2/(m-1)),2); end
        for k=1:C; w=U(:,k).^m; V(k,:)=(w'*X)/sum(w); end
        if norm(U-Uo,'fro')<1e-5; break; end
    end
end

function A = regionAdjacency(L, N)
    A = false(N,N);
    r = L(:,1:end-1)~=L(:,2:end); [i,j]=find(r);
    p=L(sub2ind(size(L),i,j)); q=L(sub2ind(size(L),i,j+1));
    A(sub2ind([N N],p,q))=true;
    dn = L(1:end-1,:)~=L(2:end,:); [i,j]=find(dn);
    p=L(sub2ind(size(L),i,j)); q=L(sub2ind(size(L),i+1,j));
    A(sub2ind([N N],p,q))=true;
    A = double(A | A');
end

function Y = normalizeCols(X)
    mn=min(X,[],1); mx=max(X,[],1); Y=(X-mn)./max(mx-mn,eps);
end

function c = corrSafe(U, lab)
    if numel(unique(lab))<2; c=mean(U,1);
    else; c=abs(corr(U,lab)); c(isnan(c))=0; end
end
