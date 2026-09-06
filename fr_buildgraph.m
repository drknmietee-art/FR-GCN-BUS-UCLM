function G = fr_buildgraph(g, lesion, cfg)
%FR_BUILDGRAPH  Fold-independent stage: superpixels, base descriptors,
%   adjacency and node labels (Sections 4.2-4.3, 4.6).
    gd = fr_anisodiff(g, cfg.diffIters, cfg.diffKappa, cfg.diffGamma);
    gd = (gd - min(gd(:))) / max(max(gd(:)) - min(gd(:)), 1e-9);

    L = fr_slic(gd, cfg.nSuperpix, cfg.compactness, cfg.slicIters);
    n = max(L(:));
    res = size(g, 1);

    q = min(max(floor(gd * cfg.glcmLevels), 0), cfg.glcmLevels - 1);
    [YY, XX] = ndgrid(1:res, 1:res);

    feat = zeros(n, 6); y = zeros(n, 1); frac = zeros(n, 1);
    for k = 1:n
        sel  = (L == k);
        vals = gd(sel);
        feat(k,1) = mean(vals);
        feat(k,2) = std(vals, 1);
        ys = YY(sel); xs = XX(sel);
        y0 = min(ys); y1 = max(ys); x0 = min(xs); x1 = max(xs);
        [c, h, e, co] = fr_maskedglcm(q(y0:y1, x0:x1), sel(y0:y1, x0:x1), cfg.glcmLevels);
        feat(k,3:6) = [c h e co];
        frac(k) = mean(lesion(sel));
        y(k) = double(frac(k) > 0.5);          % majority label
    end

    G.L = L; G.n = n; G.feat = feat; G.y = y; G.frac = frac;
    G.adj = fr_regionadj(L, n);
    G.gray = gd; G.gt = lesion;
end
