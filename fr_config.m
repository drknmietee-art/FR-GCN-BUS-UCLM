function cfg = fr_config()
%FR_CONFIG  Every setting for the FR-GCN experiments in one place.
%   Values marked "validation" were selected by grid search on the
%   patient-disjoint validation split, never on a test fold (Section 7).
%   Values marked "fixed" were set in advance and never tuned.

    % --- data ------------------------------------------------------------
    % The dataset folder is located automatically by searching upward from
    % this file, so the code works whether the MATLAB folder sits beside the
    % dataset or is nested a level or two below it.
    cfg.dataRoot = fr_finddata();
    cfg.imgDir   = fullfile(cfg.dataRoot, 'images');
    cfg.maskDir  = fullfile(cfg.dataRoot, 'masks');

    % Outputs live beside this file, not in whatever directory MATLAB
    % happened to start in.
    here          = fileparts(mfilename('fullpath'));
    cfg.baseDir   = here;
    cfg.outDir    = fullfile(here, 'results');
    cfg.cacheFile = fullfile(here, 'cache_graphs.mat');

    % --- preprocessing (fixed) -------------------------------------------
    cfg.res         = 256;
    cfg.diffIters   = 20;
    cfg.diffKappa   = 15/255;
    cfg.diffGamma   = 0.14;

    % --- superpixels ------------------------------------------------------
    cfg.nSuperpix   = 300;      % fixed
    cfg.compactness = 0.30;     % oracle reconstruction, see FRGCN_Diagnostics
    cfg.slicIters   = 10;
    cfg.glcmLevels  = 8;

    % --- fuzzy front end ---------------------------------------------------
    cfg.fcmC        = 2;        % fixed
    cfg.fcmM        = 2.0;      % fixed
    cfg.fcmSeed     = 0;        % fixed and reused everywhere, see Lemma 2
    cfg.sigmaE      = 0.30;     % validation: edge gating scale
    cfg.sigmaB      = 0.50;     % validation: fuzzy-rough approximation scale

    % --- network -----------------------------------------------------------
    cfg.layers      = 2;        % validation
    cfg.hidden      = 64;       % validation
    cfg.lr          = 0.01;     % fixed
    cfg.maxEpochs   = 150;      % validation-based early stopping selects
    cfg.valEvery    = 10;
    cfg.alpha       = 0.5;      % fixed
    cfg.lambda      = 1.0;      % fixed
    cfg.tau         = 0.5;      % fixed decision threshold
    cfg.mcDropout   = 0.1;      % MC-dropout comparator only

    % --- protocol ----------------------------------------------------------
    cfg.kFolds      = 5;
    cfg.seeds       = 0:4;      % five random initialisations
    cfg.valFraction = 0.2;      % patient-disjoint, carved from training fold
    cfg.nBoot       = 10000;

    % --- dense baselines (Deep Learning Toolbox) ---------------------------
    cfg.unetRes      = 128;     % CPU concession; predictions upsampled to 256
    cfg.unetEpochs   = 40;
    cfg.unetPatience = 8;
    cfg.unetBatch    = 8;
end
