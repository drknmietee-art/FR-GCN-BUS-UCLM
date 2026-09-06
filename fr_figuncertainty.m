function fr_figuncertainty(cfg)
%FR_FIGUNCERTAINTY  Figure 5: is the closed-form descriptor a usable
%   confidence signal?  (a) error-detection AUROC, (b) selective prediction.
    if nargin < 1, cfg = fr_config(); end
    f2 = fullfile(cfg.outDir, 'U2_error_detection.csv');
    f3 = fullfile(cfg.outDir, 'U3_referral.csv');
    if ~fr_need(f2, 'Figure 5 (uncertainty)', 'FRGCN_Uncertainty'), return; end
    if ~fr_need(f3, 'Figure 5 (uncertainty)', 'FRGCN_Uncertainty'), return; end

    [h2, U2] = fr_readcsv(f2);
    [h3, U3] = fr_readcsv(f3);
    auc = cell2mat(U2(:, find(strcmp(h2,'AUROC'),1)));
    lab = {'B(p)', 'Entropy', 'MC-dropout'};

    fh = figure('Position', [100 100 1100 400], 'Color', 'w');

    subplot(1, 3, 1); hold on
    % Explicit rectangles rather than barh() in a loop, whose width argument
    % is not honoured consistently across MATLAB and Octave.
    x0 = 0.4; xmax = max(0.95, max(auc) + 0.06); hh = 0.28;
    cols = [0.12 0.36 0.55; 0.48 0.37 0.66; 0.63 0.19 0.31];
    for i = 1:numel(auc)
        fill([x0 auc(i) auc(i) x0], [i-hh i-hh i+hh i+hh], cols(min(i,3),:), ...
             'EdgeColor', 'none');
        text(auc(i), i, sprintf(' %.3f', auc(i)), 'FontSize', 8, ...
             'VerticalAlignment', 'middle');
    end
    yl = [0.4 numel(auc)+0.6];
    plot([0.5 0.5], yl, 'k--');
    set(gca, 'YTick', 1:numel(auc), 'YTickLabel', lab(1:numel(auc)), ...
             'YLim', yl, 'XLim', [x0 xmax], 'YDir', 'reverse');
    xlabel('AUROC (error detection)'); title('(a) Ranking error'); hold off

    subplot(1, 3, [2 3]); hold on
    fx = zeros(size(U3,1), 1);
    for i = 1:size(U3,1)
        s = U3{i,1};
        if ischar(s), fx(i) = str2double(strrep(s,'%','')) / 100; else, fx(i) = s; end
    end
    cols = [0.12 0.36 0.55; 0.48 0.37 0.66; 0.63 0.19 0.31; 0.53 0.53 0.53];
    for j = 2:min(5, size(U3,2))
        y = cell2mat(U3(:, j));
        plot(fx, y, '-o', 'Color', cols(j-1,:), 'LineWidth', 1.6, 'MarkerSize', 4);
    end
    legend(h3(2:min(5,numel(h3))), 'Location', 'southeast');
    xlabel('Fraction of image area referred to a reader');
    ylabel('Dice after referral'); title('(b) Selective prediction'); hold off

    fr_savefig(fh, cfg, 'Figure5_uncertainty_quality.png');
end
