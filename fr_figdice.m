function fr_figdice(cfg)
%FR_FIGDICE  Figure 3: distribution of per-image Dice across all methods.
    if nargin < 1, cfg = fr_config(); end
    f = fullfile(cfg.outDir, 'perimage_all.csv');
    if ~fr_need(f, 'Figure 3 (Dice distribution)', 'FRGCN_RunAll'), return; end

    [h, D] = fr_readcsv(f);
    dense = fullfile(cfg.outDir, 'dense_baselines.csv');
    if exist(dense, 'file'), [~, D2] = fr_readcsv(dense); D = [D; D2]; end
    mth = D(:, find(strcmp(h,'method'),1));
    dc  = cell2mat(D(:, find(strcmp(h,'dice'),1)));
    keepAll = {'FCM + threshold','SLIC + SVM','Plain GCN','FR-GCN', ...
               'FR-GCN-lean','MLP (no propagation)','U-Net','nnU-Net-style'};
    keep = keepAll(ismember(keepAll, unique(mth)));
    data = cell(1, numel(keep));
    for i = 1:numel(keep), data{i} = dc(strcmp(mth, keep{i})); end

    fh = figure('Position', [100 100 950 420], 'Color', 'w');
    fr_boxplot(data, keep);
    ylabel('Per-image Dice'); ylim([-0.02 1.02]);
    title('Distribution of per-image Dice');
    fr_savefig(fh, cfg, 'Figure3_dice_distribution.png');
end
