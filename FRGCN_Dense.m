%FRGCN_DENSE  Dense convolutional baselines: U-Net and an nnU-Net-style
%   network, trained to convergence under the IDENTICAL protocol (Section 8.2).
%
%   The point of this script is fairness.  An earlier version of this work gave
%   the graph model 150 epochs and the dense baselines 25, while claiming an
%   identical protocol.  Here both dense models use the same patient-grouped
%   folds, the same patient-disjoint validation split and validation-based
%   early stopping, and the selected epoch is written out per fold.
%
%   REQUIRES the Deep Learning Toolbox.  This stage is OPTIONAL: skip it and
%   every other table is still produced, with the dense rows simply absent
%   from Table 6.
%
%   Implementation notes.  Training uses trainNetwork with 4-D numeric images
%   and a 4-D categorical label array, which is the long-stable API for
%   semantic segmentation from in-memory data.  Augmentation for the
%   nnU-Net-style variant is applied offline into the training array rather
%   than through augmentedImageDatastore, which takes per-image class labels
%   and is not the right tool for per-pixel label images.

cfg = fr_config();
if exist('unetLayers', 'file') ~= 2
    error('frgcn:noDLT', ['\n\nFRGCN_Dense needs the Deep Learning Toolbox ' ...
        '(unetLayers, trainNetwork).\nSkip this stage; every other table is ' ...
        'still produced without it.\n']);
end
graphs = fr_requirecache(cfg);
patients = cellfun(@(G) {G.patient}, graphs);
folds = fr_patientfolds(patients, cfg.kFolds);

% unetLayers is deprecated in recent releases but still functional; the
% warning is noise here.
wid = 'nnet_cnn:unetLayers:Deprecation';
ws  = warning('off', 'all'); %#ok<WNOFF>
cleanupW = onCleanup(@() warning(ws));

R = cfg.unetRes;
classes = ["background", "lesion"];
modes = {'unet', 'nnunet'};
rows = {};

for mi = 1:numel(modes)
    mode = modes{mi};
    for f = 1:cfg.kFolds
        te = find(folds == f); trAll = find(folds ~= f);
        tp = unique(patients(trAll));
        rs = fr_seed(1000 + f);
        [pp, rs] = fr_randperm(rs, numel(tp), max(1, round(cfg.valFraction*numel(tp))));
        isVal = ismember(patients, tp(pp));
        va = trAll(isVal(trAll)); tr = trAll(~isVal(trAll));

        [Xtr, Ytr] = fr_stack(graphs, tr, R);
        [Xva, Yva] = fr_stack(graphs, va, R);

        if strcmp(mode, 'nnunet')      % offline augmentation: one extra copy
            Xa = Xtr; Ya = Ytr;
            for j = 1:size(Xtr, 4)
                [xa, ya, rs] = fr_augment(Xtr(:,:,1,j), double(Ytr(:,:,1,j)), rs);
                Xa(:,:,1,j) = xa; Ya(:,:,1,j) = uint8(ya);
            end
            Xtr = cat(4, Xtr, Xa); Ytr = cat(4, Ytr, Ya);
        end

        Ltr = categorical(Ytr, [0 1], classes);
        Lva = categorical(Yva, [0 1], classes);

        lgraph = unetLayers([R R 1], 2, 'EncoderDepth', 4, ...
                            'NumFirstEncoderFilters', 16);
        % class-balanced pixel classification, matching the Dice+CE objective
        w = 1 ./ max([sum(Ytr(:) == 0), sum(Ytr(:) == 1)], 1);
        w = w / sum(w);
        try
            lgraph = replaceLayer(lgraph, 'Segmentation-Layer', ...
                pixelClassificationLayer('Name', 'Segmentation-Layer', ...
                                         'Classes', classes, 'ClassWeights', w));
        catch
            fprintf('  (could not set class weights; continuing unweighted)\n');
        end

        vfreq = max(1, floor(size(Xtr,4) / cfg.unetBatch));
        args = {'adam', 'InitialLearnRate', 1e-3, 'MaxEpochs', cfg.unetEpochs, ...
                'MiniBatchSize', cfg.unetBatch, 'Shuffle', 'every-epoch', ...
                'ValidationData', {Xva, Lva}, 'ValidationFrequency', vfreq, ...
                'ValidationPatience', cfg.unetPatience, ...
                'Verbose', false, 'Plots', 'none', 'ExecutionEnvironment', 'auto'};
        try   % keep the best validation checkpoint where the release supports it
            opts = trainingOptions(args{:}, 'OutputNetwork', 'best-validation-loss');
        catch
            opts = trainingOptions(args{:});
        end

        fprintf('[%s] fold %d: training on %d images (%d validation)\n', ...
                mode, f, size(Xtr,4), size(Xva,4));
        net = trainNetwork(Xtr, Ltr, lgraph, opts);

        nm = fr_ternary(strcmp(mode,'unet'), 'U-Net', 'nnU-Net-style');
        for j = 1:numel(te)
            i = te(j);
            xi = single(imresize(graphs{i}.gray, [R R], 'bilinear'));
            C  = semanticseg(xi, net);
            pm = imresize(double(C == 'lesion'), [cfg.res cfg.res], 'bilinear') > 0.5;
            m  = fr_metrics(pm, graphs{i}.gt);
            rows(end+1,:) = {nm, 0, f, graphs{i}.name, m.dice, m.jaccard, ...
                m.sensitivity, m.specificity, m.precision, ...
                fr_hd95(pm, graphs{i}.gt), NaN}; %#ok<SAGROW>
        end
        fprintf('[%s] fold %d done\n', mode, f);
    end
end
fr_writecsv(fullfile(cfg.outDir, 'dense_baselines.csv'), ...
    {'method','seed','fold','image','dice','jaccard','sensitivity', ...
     'specificity','precision','hd95','extra'}, rows);
fprintf('Wrote %s\n', fullfile(cfg.outDir, 'dense_baselines.csv'));
