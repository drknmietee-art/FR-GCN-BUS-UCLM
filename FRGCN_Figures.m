%FRGCN_FIGURES  Every manuscript figure, from the same results the tables use.
%
%   Figure numbering matches the paper.  Each figure is also callable on its
%   own, e.g.  fr_figablation  or  fr_figuncertainty.
%
%   WHAT EACH FIGURE NEEDS
%     Figure 1  pipeline schematic        nothing
%     Figure 2  superpixels + RAG         cache_graphs.mat   (FRGCN_Extract)
%     Figure 3  Dice distribution         perimage_all.csv   (FRGCN_RunAll)
%     Figure 4  component ablation        T8_ablation.csv    (FRGCN_Tables)
%     Figure 5  uncertainty quality       U2_/U3_*.csv       (FRGCN_Uncertainty)
%     Figure 6  training curve            cache_graphs.mat   (retrains fold 1)
%     Figure 7  threshold sweep           U5_threshold.csv   (FRGCN_Uncertainty)
%
%   So after training, the order that fills everything in is:
%       FRGCN_Uncertainty   ->  Figures 5 and 6
%       FRGCN_Tables        ->  Figure 4
%       FRGCN_Figures       ->  all seven
%
%   A figure whose input is missing is skipped with a message naming the
%   script that produces it, so the rest still render.

cfg = fr_config();
fprintf('Writing figures to %s\n', fr_figdir(cfg));

figs = {@fr_figpipeline,    'Figure 1 (pipeline)'; ...
        @fr_figsuperpixels, 'Figure 2 (superpixels)'; ...
        @fr_figdice,        'Figure 3 (Dice distribution)'; ...
        @fr_figablation,    'Figure 4 (ablation)'; ...
        @fr_figuncertainty, 'Figure 5 (uncertainty quality)'; ...
        @fr_figtraining,    'Figure 6 (training curve)'; ...
        @fr_figthreshold,   'Figure 7 (threshold sweep)'};

for i = 1:size(figs, 1)
    try
        figs{i,1}(cfg);
    catch err
        fprintf('  SKIP %s -- %s\n', figs{i,2}, err.message);
    end
end
fprintf('Done. Figures in %s\n', fr_figdir(cfg));
