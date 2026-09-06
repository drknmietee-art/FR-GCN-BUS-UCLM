function fr_figthreshold(cfg)
%FR_FIGTHRESHOLD  Figure 7: decision-threshold sweep.
%   Confirms the pre-registered tau = 0.5 sits close to the Dice optimum, so
%   the reported numbers do not depend on a fortunate threshold.
    if nargin < 1, cfg = fr_config(); end
    f = fullfile(cfg.outDir, 'U5_threshold.csv');
    if ~fr_need(f, 'Figure 7 (threshold sweep)', 'FRGCN_Uncertainty'), return; end

    [h, T] = fr_readcsv(f);
    g = @(nm) cell2mat(T(:, find(strcmp(h,nm),1)));
    tau = g('threshold');

    fh = figure('Position', [100 100 640 400], 'Color', 'w');
    hold on
    plot(tau, g('Dice'),        '-o', 'Color', [0.12 0.36 0.55], 'LineWidth', 1.7);
    plot(tau, g('Sensitivity'), '--s', 'Color', [0.70 0.42 0.00], 'LineWidth', 1.4);
    plot(tau, g('Precision'),   '--^', 'Color', [0.24 0.56 0.43], 'LineWidth', 1.4);
    yl = ylim;
    plot([cfg.tau cfg.tau], yl, 'k:', 'LineWidth', 1.2);
    text(cfg.tau, yl(2), sprintf('  \\tau = %.1f (fixed in advance)', cfg.tau), ...
         'VerticalAlignment', 'top', 'FontSize', 8);
    legend({'Dice','Sensitivity','Precision'}, 'Location', 'best');
    xlabel('Decision threshold \tau'); ylabel('Score');
    title('Operating point sweep'); hold off
    fr_savefig(fh, cfg, 'Figure7_threshold_sweep.png');
end
