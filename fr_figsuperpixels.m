function fr_figsuperpixels(cfg)
%FR_FIGSUPERPIXELS  Figure 2: preprocessed image, SLIC over-segmentation and
%   the induced region adjacency graph.
    if nargin < 1, cfg = fr_config(); end
    graphs = fr_requirecache(cfg);
    sz = cellfun(@(G) sum(G.gt(:)), graphs);
    [~, bi] = max(sz);
    G = graphs{bi};

    fh = figure('Position', [100 100 1150 400], 'Color', 'w');
    subplot(1,3,1); imagesc(G.gray); colormap(gray); axis image off;
    title('(a) Preprocessed image');

    subplot(1,3,2); image(fr_boundaryoverlay(G.gray, G.L)); axis image off;
    title(sprintf('(b) SLIC superpixels (n = %d)', G.n));

    subplot(1,3,3); imagesc(G.gray); colormap(gray); axis image off; hold on
    [YY, XX] = ndgrid(1:size(G.L,1), 1:size(G.L,2));
    cy = zeros(G.n,1); cx = zeros(G.n,1);
    for k = 1:G.n, s = (G.L==k); cy(k) = mean(YY(s)); cx(k) = mean(XX(s)); end
    [a, b] = find(triu(G.adj));
    for e = 1:numel(a)
        plot([cx(a(e)) cx(b(e))], [cy(a(e)) cy(b(e))], '-', ...
             'Color', [0.30 0.65 1.00], 'LineWidth', 0.3);
    end
    plot(cx, cy, '.', 'Color', [1 0.30 0.30], 'MarkerSize', 5);
    title('(c) Region adjacency graph'); hold off
    fr_savefig(fh, cfg, 'Figure2_superpixel_graph.png');
end
