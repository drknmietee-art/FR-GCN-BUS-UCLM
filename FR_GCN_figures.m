%% ========================================================================
%  FR_GCN_figures.m
%  Figure generator for the FR-GCN BUS-UCLM paper.
%  Produces every remaining manuscript figure from a trained model.
%
%  Figure 1  Pipeline schematic (block diagram).
%  Figure 2  Fuzzy lesion membership map and fuzzy-rough boundary map.
%  Figure 3  Superpixel over-segmentation and the region adjacency graph.
%  Figure 4  Qualitative results: input, ground truth, FR-GCN prediction.
%  Figure 5  Box plots of per-image Dice and Jaccard over a test set.
%  Figure 6  Training loss against epoch (convergence).
%
%  Tested on MATLAB R2024b. Toolboxes: Image Processing, Deep Learning,
%  Fuzzy Logic (optional; internal fallback provided).
%
%  This file is self-contained. It reuses a cached model when available
%  (results/model.mat), otherwise it trains a compact model and caches it.
%  Figures are written to results/figures as PNG (300 dpi) and PDF.
% =========================================================================

function FR_GCN_figures()

clc; close all; rng(42);

%% --------------------------- CONFIG ------------------------------------
CFG = struct();
CFG.dataRoot   = fullfile('BUS-UCLM Breast ultrasound lesion segmentation dataset','BUS-UCLM');
CFG.imgDir     = fullfile(CFG.dataRoot,'images');
CFG.maskDir    = fullfile(CFG.dataRoot,'masks');
CFG.outDir     = fullfile(pwd,'results');
CFG.figDir     = fullfile(CFG.outDir,'figures');
CFG.imResize   = [256 256];
CFG.nSuperpix  = 300;
CFG.fcmC       = 2;
CFG.fcmM       = 2.0;
CFG.sigmaSim   = 0.30;
CFG.hidden     = 64;
CFG.numLayers  = 2;
CFG.lr         = 1e-2;
CFG.epochs     = 200;
CFG.alpha      = 0.5;
CFG.lambda     = 1.0;
CFG.nTrain     = 150;   % images used to (re)train the cached model
CFG.nTest      = 60;    % images used for the Figure 5 box plot
CFG.dpi        = 300;
CFG.keepOpen   = false; % true = leave figures open on screen for reshaping

if ~exist(CFG.outDir,'dir'); mkdir(CFG.outDir); end
if ~exist(CFG.figDir,'dir'); mkdir(CFG.figDir); end

%% --------------------- INDEX LESION IMAGES -----------------------------
imgFiles = dir(fullfile(CFG.imgDir,'*.png'));
assert(~isempty(imgFiles),'No images found. Check CFG.imgDir.');
names = string({imgFiles.name});

cls = strings(numel(names),1); hasL = false(numel(names),1);
for i = 1:numel(names)
    [cls(i), hasL(i)] = maskClass(imread(fullfile(CFG.maskDir,names(i))));
end
lesionNames = names(hasL);
benignNames = names(cls=="benign");
malignNames = names(cls=="malignant");
fprintf('Lesion images: %d (benign %d, malignant %d)\n', ...
    numel(lesionNames), numel(benignNames), numel(malignNames));

%% ------------------ TRAIN OR LOAD THE MODEL ----------------------------
modelFile = fullfile(CFG.outDir,'model.mat');
if exist(modelFile,'file')
    fprintf('Loading cached model from %s\n', modelFile);
    S = load(modelFile); params = S.params; lossHistory = S.lossHistory;
else
    fprintf('No cached model. Training a compact model for the figures ...\n');
    trList = lesionNames(randperm(numel(lesionNames), ...
             min(CFG.nTrain,numel(lesionNames))));
    Gtr = struct('X',{},'Ahat',{},'y',{},'spLabel',{});
    for t = 1:numel(trList)
        [Ig,gt] = preprocess(imread(fullfile(CFG.imgDir,trList(t))), ...
                             imread(fullfile(CFG.maskDir,trList(t))), CFG.imResize);
        [X,Ahat,y,L] = buildGraph(Ig,gt,CFG);
        Gtr(end+1) = struct('X',X,'Ahat',Ahat,'y',y,'spLabel',L); %#ok<AGROW>
        if mod(t,25)==0; fprintf('   %d / %d training graphs\n',t,numel(trList)); end
    end
    [params, lossHistory] = trainGCNlog(Gtr, CFG);
    save(modelFile,'params','lossHistory','CFG');
    fprintf('Model cached to %s\n', modelFile);
end

%% =========================== FIGURE 1 ==================================
makePipelineFigure(CFG);

%% =========================== FIGURE 2 & 3 ==============================
% Use one clear benign and the graph of the same case.
sample = pickSample(benignNames, malignNames, 1);
[Ig,gt] = preprocess(imread(fullfile(CFG.imgDir,sample)), ...
                     imread(fullfile(CFG.maskDir,sample)), CFG.imResize);
[X,Ahat,y,L] = buildGraph(Ig,gt,CFG); %#ok<ASGLU>
makeMembershipFigure(Ig, L, X, CFG);          % Figure 2
makeGraphFigure(Ig, L, CFG);                  % Figure 3

%% =========================== FIGURE 4 ==================================
% Qualitative grid: two benign and two malignant cases.
grid = [pickSample(benignNames, malignNames, 1), ...
        pickSample(benignNames, malignNames, 2), ...
        pickSample(malignNames, benignNames, 1), ...
        pickSample(malignNames, benignNames, 2)];
makeQualitativeFigure(grid, params, CFG);     % Figure 4

%% =========================== FIGURE 5 ==================================
teList = lesionNames(randperm(numel(lesionNames), ...
         min(CFG.nTest,numel(lesionNames))));
M = zeros(numel(teList),5);
for t = 1:numel(teList)
    [Ig,gt] = preprocess(imread(fullfile(CFG.imgDir,teList(t))), ...
                         imread(fullfile(CFG.maskDir,teList(t))), CFG.imResize);
    [X,Ahat,y,L] = buildGraph(Ig,gt,CFG);
    yhat = predictGCN(params, X, Ahat);
    M(t,:) = imageMetrics(y, yhat, L, CFG.imResize);
end
makeBoxplotFigure(M, CFG);                     % Figure 5

%% =========================== FIGURE 6 ==================================
makeLossFigure(lossHistory, CFG);              % Figure 6

fprintf('\nAll figures written to: %s\n', CFG.figDir);
end % ===================== end main function ============================


%% ======================================================================
%  FIGURE BUILDERS
%  ======================================================================

function makePipelineFigure(CFG)
    f = figure('Color','w','Position',[80 80 1180 430]);
    ax = axes('Parent',f,'Position',[0.02 0.02 0.96 0.96]); hold(ax,'on');
    axis(ax,[0 11 0 4]); axis(ax,'off');
    labels = {'Input US image','Preprocess','SLIC superpixels','FCM membership', ...
              'Fuzzy-rough boundary','Node features','GCN layers','Lesion mask'};
    cxTop = [1.3 4.0 6.7 9.4]; cxBot = [9.4 6.7 4.0 1.3];
    yTop = 2.7; yBot = 0.7; w = 2.2; h = 1.0;
    for k = 1:4; drawBox(ax, cxTop(k), yTop, w, h, labels{k}); end
    for k = 1:4; drawBox(ax, cxBot(k), yBot, w, h, labels{k+4}); end
    for k = 1:3; drawArrowH(ax, cxTop(k)+w/2, cxTop(k+1)-w/2, yTop); end
    drawArrowV(ax, cxTop(4), yTop-h/2, yBot+h/2);                 % 4 -> 5
    for k = 1:3; drawArrowH(ax, cxBot(k)-w/2, cxBot(k+1)+w/2, yBot); end
    title(ax,'Figure 1. FR-GCN pipeline','FontWeight','bold','FontSize',11);
    saveFig(f, 'Figure1_pipeline', CFG);
end

function makeMembershipFigure(Ig, L, X, CFG)
    uLesion = X(:,7); boundary = X(:,9);       % see node feature layout
    memMap = uLesion(L);                        % broadcast node -> pixel
    bndMap = boundary(L);
    f = figure('Color','w','Position',[80 80 1150 380]);
    tiledlayout(f,1,3,'Padding','compact','TileSpacing','compact');
    nexttile; imshow(Ig,[]); title('Input (preprocessed)');
    nexttile; imagesc(memMap); axis image off; colormap(gca,parula); colorbar;
        title('Fuzzy lesion membership');
    nexttile; imagesc(bndMap); axis image off; colormap(gca,hot); colorbar;
        title('Fuzzy-rough boundary uncertainty');
    sgtitle('Figure 2. Membership and uncertainty maps','FontWeight','bold');
    saveFig(f, 'Figure2_membership_boundary', CFG);
end

function makeGraphFigure(Ig, L, CFG)
    N = max(L(:));
    Adj = regionAdjacency(L, N);
    s = regionprops(L,'Centroid'); C = cat(1, s.Centroid);
    f = figure('Color','w','Position',[80 80 900 430]);
    tiledlayout(f,1,2,'Padding','compact','TileSpacing','compact');
    B = boundarymask(L);                       % superpixel borders
    RGB = repmat(Ig,1,1,3);
    Rc=RGB(:,:,1); Gc=RGB(:,:,2); Bc=RGB(:,:,3);
    Rc(B)=1; Gc(B)=1; Bc(B)=0;                 % paint borders yellow
    RGB = cat(3,Rc,Gc,Bc);
    nexttile; imshow(RGB); title('Superpixels');
    nexttile; imshow(Ig,[]); hold on;
        [p,q] = find(triu(Adj,1));
        for e = 1:numel(p)
            plot([C(p(e),1) C(q(e),1)], [C(p(e),2) C(q(e),2)], ...
                 '-','Color',[0.1 0.6 1 0.5],'LineWidth',0.4);
        end
        plot(C(:,1), C(:,2), '.', 'Color',[1 0.2 0.2], 'MarkerSize',6);
        title('Region adjacency graph');
    sgtitle('Figure 3. Superpixel graph construction','FontWeight','bold');
    saveFig(f, 'Figure3_graph', CFG);
end

function makeQualitativeFigure(nameList, params, CFG)
    n = numel(nameList);
    f = figure('Color','w','Position',[60 60 780 240*n]);
    tiledlayout(f,n,3,'Padding','compact','TileSpacing','compact');
    for i = 1:n
        [Ig,gt] = preprocess(imread(fullfile(CFG.imgDir,nameList(i))), ...
                             imread(fullfile(CFG.maskDir,nameList(i))), CFG.imResize);
        [X,Ahat,y,L] = buildGraph(Ig,gt,CFG); %#ok<ASGLU>
        yhat = predictGCN(params, X, Ahat);
        predMask = (yhat > 0.5); predMask = predMask(L);
        gtMask   = (y > 0.5);    gtMask   = gtMask(L);
        nexttile; imshow(Ig,[]); title(shortName(nameList(i)));
        nexttile; imshow(Ig,[]); hold on; visboundaries(gtMask,'Color','g','LineWidth',1);
            if i==1; title('Ground truth (green)'); end
        nexttile; imshow(Ig,[]); hold on; visboundaries(predMask,'Color','r','LineWidth',1);
            if i==1; title('FR-GCN prediction (red)'); end
    end
    sgtitle('Figure 4. Qualitative segmentation results','FontWeight','bold');
    saveFig(f, 'Figure4_qualitative', CFG);
end

function makeBoxplotFigure(M, CFG)
    f = figure('Color','w','Position',[100 100 560 430]);
    boxplot(M(:,1:2), 'Labels', {'Dice','Jaccard'});
    ylabel('Score'); ylim([0 1]); grid on;
    title('Figure 5. Per-image score distribution','FontWeight','bold');
    saveFig(f, 'Figure5_boxplot', CFG);
end

function makeLossFigure(lossHistory, CFG)
    f = figure('Color','w','Position',[100 100 620 430]);
    plot(1:numel(lossHistory), lossHistory, '-','LineWidth',1.6,'Color',[0.1 0.3 0.8]);
    xlabel('Epoch'); ylabel('Mean training loss'); grid on;
    title('Figure 6. FR-GCN training convergence','FontWeight','bold');
    saveFig(f, 'Figure6_convergence', CFG);
end

%% ======================================================================
%  DRAWING / SAVE HELPERS
%  ======================================================================

function drawBox(ax, cx, cy, w, h, str)
    rectangle('Parent',ax,'Position',[cx-w/2 cy-h/2 w h], ...
        'Curvature',0.15,'FaceColor',[0.90 0.94 1.0],'EdgeColor',[0.2 0.3 0.6],'LineWidth',1.2);
    text(ax, cx, cy, str, 'HorizontalAlignment','center', ...
        'VerticalAlignment','middle','FontSize',9);
end

function drawArrowH(ax, x1, x2, y)
    plot(ax,[x1 x2],[y y],'k-','LineWidth',1.1);
    mk = '>'; if x2 < x1; mk = '<'; end
    plot(ax, x2, y, 'k', 'Marker', mk, 'MarkerFaceColor','k','MarkerSize',6);
end

function drawArrowV(ax, x, y1, y2)
    plot(ax,[x x],[y1 y2],'k-','LineWidth',1.1);
    plot(ax, x, y2, 'k', 'Marker','v', 'MarkerFaceColor','k','MarkerSize',6);
end

function saveFig(f, base, CFG)
    png = fullfile(CFG.figDir, [base '.png']);
    pdf = fullfile(CFG.figDir, [base '.pdf']);
    fig = fullfile(CFG.figDir, [base '.fig']);
    % Editable MATLAB figure (reopen with openfig to reshape / restyle).
    savefig(f, fig);
    % Raster and vector exports for the manuscript.
    try
        exportgraphics(f, png, 'Resolution', CFG.dpi);
        exportgraphics(f, pdf, 'ContentType','vector');
    catch
        print(f, png, '-dpng', sprintf('-r%d',CFG.dpi));
    end
    fprintf('   saved %s (+ .pdf, .fig)\n', png);
    if CFG.keepOpen
        return;                 % leave figure open for interactive reshaping
    end
    close(f);
end

function nm = shortName(fullname)
    [~, nm] = fileparts(fullname); nm = strrep(nm,'_','\_');
end

function s = pickSample(primary, backup, k)
    if numel(primary) >= k; s = primary(k); else; s = backup(k); end
end

%% ======================================================================
%  PIPELINE FUNCTIONS  (kept in sync with FR_GCN_BUS_UCLM.m)
%  ======================================================================

function [cls, hasL] = maskClass(M)
    if size(M,3)==1; M = repmat(M,1,1,3); end
    R = M(:,:,1); Gc = M(:,:,2); B = M(:,:,3);
    green = nnz(Gc>100 & R<100 & B<100);
    red   = nnz(R>100 & Gc<100 & B<100);
    if green==0 && red==0; cls="normal"; hasL=false;
    elseif green>=red;     cls="benign"; hasL=true;
    else;                  cls="malignant"; hasL=true; end
end

function [Ig, gt] = preprocess(I, Mc, sz)
    if size(I,3)==3; Ig = rgb2gray(I); else; Ig = I; end
    Ig = im2double(imresize(Ig, sz));
    Ig = imdiffusefilt(Ig, 'NumberOfIterations', 5);
    Ig = mat2gray(Ig);
    if size(Mc,3)==1; Mc = repmat(Mc,1,1,3); end
    R = Mc(:,:,1); Gc = Mc(:,:,2); B = Mc(:,:,3);
    lesion = (Gc>100 & R<100 & B<100) | (R>100 & Gc<100 & B<100);
    gt = imresize(lesion, sz, 'nearest');
end

function [X, Ahat, y, L] = buildGraph(Ig, gt, CFG)
    [L, N] = superpixels(Ig, CFG.nSuperpix);
    glcmOff = [0 1; -1 1; -1 0; -1 -1];
    feat = zeros(N,6); lab = zeros(N,1);
    idxList = label2idx(L); [rr,cc] = size(Ig);
    for k = 1:N
        px = idxList{k}; vals = Ig(px);
        feat(k,1) = mean(vals); feat(k,2) = std(vals);
        [ys,xs] = ind2sub([rr cc], px);
        patch = Ig(min(ys):max(ys), min(xs):max(xs));
        gl = graycomatrix(patch,'Offset',glcmOff,'NumLevels',8,'Symmetric',true);
        st = graycoprops(gl,{'Contrast','Homogeneity','Energy','Correlation'});
        feat(k,3)=mean(st.Contrast); feat(k,4)=mean(st.Homogeneity);
        feat(k,5)=mean(st.Energy);
        cval=mean(st.Correlation); if isnan(cval); cval=0; end
        feat(k,6)=cval;
        lab(k)=mean(gt(px))>0.5;
    end
    feat = normalizeCols(feat);
    U = fuzzyCMeans(feat, CFG.fcmC, CFG.fcmM);
    [~, lesCol] = max(corrSafe(U, lab));
    uLesion = U(:,lesCol);
    Dfeat = pdist2(feat,feat);
    Rsim  = exp(-(Dfeat.^2)/(2*CFG.sigmaSim^2));
    A = uLesion(:)'; lower = zeros(N,1); upper = zeros(N,1);
    for p = 1:N
        lower(p) = min(min(1, 1 - Rsim(p,:) + A));
        upper(p) = max(max(0, Rsim(p,:) + A - 1));
    end
    boundary = upper - lower;
    X = [feat, uLesion, lower, boundary];
    Adj = regionAdjacency(L, N);
    W = Adj .* Rsim + eye(N);
    d = sum(W,2); Dinv = diag(1./sqrt(max(d,eps)));
    Ahat = Dinv * W * Dinv;
    y = lab(:);
end

function U = fuzzyCMeans(X, C, m)
    if exist('fcm','file')==2
        try
            opt = fcmOptions('NumClusters',C,'Exponent',m,'MaxNumIteration',100,'Verbose',false);
            [~,U] = fcm(X, opt);
        catch
            [~,U] = fcm(X, C);
        end
        U = U'; return;
    end
    N = size(X,1); rng(1); V = X(randperm(N,C),:);
    U = rand(N,C); U = U./sum(U,2);
    for it = 1:100
        D = max(pdist2(X,V),1e-9); Uo = U;
        for k=1:C; U(:,k)=1./sum((D(:,k)./D).^(2/(m-1)),2); end
        for k=1:C; w=U(:,k).^m; V(k,:)=(w'*X)/sum(w); end
        if norm(U-Uo,'fro')<1e-5; break; end
    end
end

function A = regionAdjacency(L, N)
    A = false(N,N);
    r = L(:,1:end-1) ~= L(:,2:end); [i,j] = find(r);
    p = L(sub2ind(size(L),i,j)); q = L(sub2ind(size(L),i,j+1));
    A(sub2ind([N N],p,q)) = true;
    dn = L(1:end-1,:) ~= L(2:end,:); [i,j] = find(dn);
    p = L(sub2ind(size(L),i,j)); q = L(sub2ind(size(L),i+1,j));
    A(sub2ind([N N],p,q)) = true;
    A = double(A | A');
end

function [params, lossHist] = trainGCNlog(Gtr, CFG)
    fin = size(Gtr(1).X,2);
    params = initParams(fin, CFG.hidden, 2, CFG.numLayers);
    avgG=[]; avgSq=[]; iter=0; lossHist = zeros(CFG.epochs,1);
    for ep = 1:CFG.epochs
        order = randperm(numel(Gtr)); epLoss = 0;
        for gi = order
            iter = iter + 1;
            Xg = dlarray(single(Gtr(gi).X));
            Ah = dlarray(single(Gtr(gi).Ahat));
            yv = single(Gtr(gi).y); bnd = single(Gtr(gi).X(:,end));
            [loss,grad] = dlfeval(@modelLoss, params, Xg, Ah, yv, bnd, CFG);
            [params,avgG,avgSq] = adamupdate(params,grad,avgG,avgSq,iter,CFG.lr);
            epLoss = epLoss + double(gather(extractdata(loss)));
        end
        lossHist(ep) = epLoss/numel(Gtr);
        if mod(ep,50)==0; fprintf('   epoch %3d | loss %.4f\n',ep,lossHist(ep)); end
    end
end

function p = initParams(fin, h, nc, nl)
    p = struct(); dims = [fin, repmat(h,1,nl-1), nc];
    for l = 1:nl; p.(sprintf('W%d',l)) = dlarray(glorot(dims(l),dims(l+1))); end
end

function W = glorot(fanIn, fanOut)
    r = sqrt(6/(fanIn+fanOut)); W = single(-r + 2*r*rand(fanIn,fanOut));
end

function [loss, grad] = modelLoss(params, X, Ahat, y, bnd, CFG)
    Z = forwardGCN(params, X, Ahat); P = softmaxRows(Z);
    y = y(:); bnd = bnd(:); yhot = [1-y, y];
    w = 1 + CFG.lambda*bnd;
    ce = -sum(w .* sum(yhot .* log(P+1e-8),2))/numel(y);
    pl = P(:,2); dice = 1 - (2*sum(pl.*y)+1)/(sum(pl)+sum(y)+1);
    loss = CFG.alpha*ce + (1-CFG.alpha)*dice;
    grad = dlgradient(loss, params);
end

function Z = forwardGCN(params, X, Ahat)
    nl = sum(startsWith(fieldnames(params),'W'));
    H = X;
    for l = 1:nl
        H = Ahat * H * params.(sprintf('W%d',l));
        if l < nl; H = relu(H); end
    end
    Z = H;
end

function P = softmaxRows(Z)
    Zs = Z - max(Z,[],2); E = exp(Zs); P = E ./ sum(E,2);
end

function yhat = predictGCN(params, X, Ahat)
    Z = forwardGCN(params, dlarray(single(X)), dlarray(single(Ahat)));
    P = softmaxRows(Z);
    yhat = double(gather(extractdata(P(:,2))));
end

function m = imageMetrics(y, yhat, L, sz) %#ok<INUSD>
    predNode = yhat > 0.5; predMask = predNode(L);
    gtNode = y > 0.5;       gtMask = gtNode(L);
    P = predMask(:); G = gtMask(:);
    TP=nnz(P&G); FP=nnz(P&~G); FN=nnz(~P&G); TN=nnz(~P&~G);
    dice=2*TP/max(2*TP+FP+FN,eps); jac=TP/max(TP+FP+FN,eps);
    sens=TP/max(TP+FN,eps); spec=TN/max(TN+FP,eps); prec=TP/max(TP+FP,eps);
    m = [dice jac sens spec prec];
end

function Y = normalizeCols(X)
    mn=min(X,[],1); mx=max(X,[],1); Y=(X-mn)./max(mx-mn,eps);
end

function c = corrSafe(U, lab)
    if numel(unique(lab))<2; c = mean(U,1);
    else; c = abs(corr(U,lab)); c(isnan(c))=0; end
end
