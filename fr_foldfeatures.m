function [X, Af, Ab, Ai] = fr_foldfeatures(G, lo, hi, cfg)
%FR_FOLDFEATURES  Fold-dependent stage: scale, cluster, approximate, build
%   the two propagation operators (Sections 4.5-4.7).
%
%   TWO SCALES.  sigmaE gates the edges and must SEPARATE adjacent regions,
%   so it wants to be small; sigmaB enters the approximations and needs a
%   neighbourhood wide enough for the two bounds to differ.  With one shared
%   scale, B(p) collapses to zero in a high-dimensional descriptor space and
%   the descriptor is silently crippled.
    Xs = fr_applyscaler(G.featAll, lo, hi);
    D2 = fr_pdist2(Xs, Xs) .^ 2;

    [U, V] = fr_fcm(Xs, cfg.fcmC, cfg.fcmM, 100, cfg.fcmSeed);
    [A, lower, ~, B] = fr_fuzzyrough(Xs, U, V, cfg.sigmaB, D2);
    X = [Xs, A, lower, B];

    R   = exp(-D2 ./ (2 * cfg.sigmaE^2));
    adj = double(G.adj);
    N   = size(Xs, 1);
    Af  = fr_normadj(adj .* R + eye(N));    % fuzzy-gated  (Eq. 7-8)
    Ab  = fr_normadj(adj + eye(N));         % plain binary
    Ai  = speye(N);                         % no propagation (control)
end
