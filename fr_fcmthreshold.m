function pred = fr_fcmthreshold(G, cfg)
%FR_FCMTHRESHOLD  Classical baseline: fuzzy c-means on pixel intensities,
%   darker cluster taken as lesion.  Unsupervised throughout -- the same
%   darker-centroid rule the fuzzy front end uses to pick its lesion cluster.
    v = G.gray(:);
    sub = v(1:7:end);                       % subsample for speed
    [~, V] = fr_fcm(sub, cfg.fcmC, cfg.fcmM, 100, cfg.fcmSeed);
    [~, dark] = min(V(:,1));
    d = abs(v - V(:,1)');
    [~, lab] = min(d, [], 2);
    pred = reshape(lab == dark, size(G.gray));
end
