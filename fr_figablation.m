function fr_figablation(cfg)
%FR_FIGABLATION  Figure 4: component ablation against seed-to-seed variation.
%   Only the edge-gating effect should clearly clear the shaded band; the
%   no-propagation control in Table 9 governs how that is read.
    if nargin < 1, cfg = fr_config(); end
    f = fullfile(cfg.outDir, 'T8_ablation.csv');
    if ~fr_need(f, 'Figure 4 (ablation)', 'FRGCN_Tables'), return; end

    [h, T] = fr_readcsv(f);
    name = T(:, find(strcmp(h,'Configuration'),1));
    dd   = cell2mat(T(:, find(strcmp(h,'Delta_Dice'),1)));
    sd   = cell2mat(T(:, find(strcmp(h,'Seed_SD'),1)));
    sel  = ~strcmp(name, 'FR-GCN');
    v = dd(sel); nm = strrep(name(sel), 'abl: ', '');
    band = median(sd(~isnan(sd)));

    fh = figure('Position', [100 100 900 400], 'Color', 'w');
    hold on
    n = numel(v);
    fill([0.3 n+0.7 n+0.7 0.3], [-band -band band band], [0.87 0.87 0.87], ...
         'EdgeColor', 'none');
    % Explicit rectangles rather than bar() in a loop: the width argument of
    % bar() is not honoured consistently across MATLAB and Octave, which
    % silently merges adjacent bars into one band.
    w = 0.3;
    for i = 1:n
        if v(i) > 0, c = [0.24 0.56 0.43]; else, c = [0.69 0.25 0.25]; end
        fill([i-w i+w i+w i-w], [0 0 v(i) v(i)], c, 'EdgeColor', 'none');
        if v(i) >= 0, va = 'bottom'; else, va = 'top'; end
        text(i, v(i), sprintf('%+.3f', v(i)), 'HorizontalAlignment', 'center', ...
             'VerticalAlignment', va, 'FontSize', 8);
    end
    plot([0.3 n+0.7], [0 0], 'k-', 'LineWidth', 0.8);
    set(gca, 'XTick', 1:n, 'XTickLabel', nm, 'XLim', [0.3 n+0.7]);
    ylabel('\Delta Dice vs full FR-GCN');
    title(sprintf('Component ablation (shaded: seed variation +/- %.3f)', band));
    hold off
    fr_savefig(fh, cfg, 'Figure4_ablation.png');
end
