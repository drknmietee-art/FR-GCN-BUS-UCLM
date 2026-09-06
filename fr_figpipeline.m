function fr_figpipeline(cfg)
%FR_FIGPIPELINE  Figure 1: schematic of the FR-GCN pipeline.
%   The fuzzy-rough front end is computed in closed form before any learning
%   and enters the model in three places, each ablated separately in Sec. 8.4.
    if nargin < 1, cfg = fr_config(); end
    steps = {'Ultrasound', 'Anisotropic', 'SLIC superpixels', 'Fuzzy c-means', ...
             'Fuzzy-rough', 'Graph convolution', 'Pixel'};
    sub   = {'image', 'diffusion', '+ adjacency graph', 'membership A', ...
             'B(p)', 'with fuzzy gating', 'reconstruction'};
    sec   = {'', '(Sec. 4.2)', '(Sec. 4.3)', '(Sec. 4.4)', '(Sec. 4.5)', ...
             '(Sec. 4.7)', '(Sec. 4.9)'};
    n = numel(steps);

    fh = figure('Position', [100 100 1250 320], 'Color', 'w');
    hold on
    w = 14; h = 11; gap = 2.6; y0 = 12;
    for i = 1:n
        x = 1 + (i-1)*(w+gap);
        if i == 5, fc = [1.00 0.90 0.70]; else, fc = [0.81 0.89 0.95]; end
        rectangle('Position', [x y0 w h], 'Curvature', 0.2, ...
                  'FaceColor', fc, 'EdgeColor', [0.19 0.41 0.55]);
        text(x+w/2, y0+h*0.70, steps{i}, 'HorizontalAlignment','center','FontSize',8);
        text(x+w/2, y0+h*0.45, sub{i},   'HorizontalAlignment','center','FontSize',8);
        text(x+w/2, y0+h*0.20, sec{i},   'HorizontalAlignment','center','FontSize',7, ...
             'Color', [0.4 0.4 0.4]);
        if i > 1
            plot([x-gap x], [y0+h/2 y0+h/2], '-', 'Color', [0.19 0.41 0.55], 'LineWidth', 1.2);
            plot(x-0.9, y0+h/2, '>', 'Color', [0.19 0.41 0.55], 'MarkerSize', 4, ...
                 'MarkerFaceColor', [0.19 0.41 0.55]);
        end
    end
    bx = 1 + 4*(w+gap);
    rectangle('Position', [bx-1 2 2*w+gap+2 7], 'Curvature', 0.15, ...
              'FaceColor', [1.00 0.96 0.87], 'EdgeColor', [0.79 0.55 0.11], ...
              'LineStyle', '--');
    text(bx + w + gap/2, 5.5, ...
         'B(p) used three times: node feature (Eq. 7)  .  edge weight (Eq. 8)  .  loss weight (Eq. 11)', ...
         'HorizontalAlignment','center','FontSize',8,'Color',[0.54 0.36 0.00]);
    plot([bx+w/2 bx+w/2], [9.2 y0-0.2], '-', 'Color', [0.79 0.55 0.11], 'LineWidth', 1.2);
    axis([0 1+n*(w+gap) 0 27]); axis off
    hold off
    fr_savefig(fh, cfg, 'Figure1_pipeline.png');
end
