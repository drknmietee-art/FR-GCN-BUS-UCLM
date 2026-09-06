function fr_boxplot(data, labels)
%FR_BOXPLOT  Box plot without the Statistics Toolbox.
    hold on
    for i = 1:numel(data)
        v = sort(data{i}(:)); v = v(~isnan(v));
        if isempty(v), continue; end
        q1 = fr_prctile(v,25); q2 = fr_prctile(v,50); q3 = fr_prctile(v,75);
        iqr_ = q3 - q1;
        loW = min(v(v >= q1 - 1.5*iqr_)); hiW = max(v(v <= q3 + 1.5*iqr_));
        w = 0.32;
        fill([i-w i+w i+w i-w], [q1 q1 q3 q3], [.81 .89 .95], 'EdgeColor', [.19 .41 .55]);
        plot([i-w i+w], [q2 q2], '-', 'Color', [.69 .19 .19], 'LineWidth', 1.5);
        plot([i i], [q3 hiW], 'k-'); plot([i i], [loW q1], 'k-');
        plot([i-w/2 i+w/2], [hiW hiW], 'k-'); plot([i-w/2 i+w/2], [loW loW], 'k-');
        plot(i, mean(v), 'kd', 'MarkerFaceColor', 'w', 'MarkerSize', 4);
    end
    set(gca, 'XTick', 1:numel(data), 'XTickLabel', labels, 'XLim', [.4 numel(data)+.6]);
    try, xtickangle(18); catch, end
    box off; hold off
end
