function [U, V] = fr_fcm(X, C, m, iters, seed)
%FR_FCM  Fuzzy c-means (Dunn 1973; Bezdek 1981).  Equations (1)-(3).
%   [U,V] = FR_FCM(X,C,m,iters,seed) returns memberships U (N-by-C) and
%   centroids V (C-by-d) for data X (N-by-d).
%
%   Lemma 2 of the manuscript: the alternating updates do not increase the
%   objective, so the objective sequence converges.  Convergence is to a
%   local optimum that depends on the initialisation, which is why the seed
%   is fixed and reused across every fold, baseline and ablation.
    if nargin < 2 || isempty(C),     C     = 2;    end
    if nargin < 3 || isempty(m),     m     = 2.0;  end
    if nargin < 4 || isempty(iters), iters = 100;  end
    if nargin < 5 || isempty(seed),  seed  = 0;    end

    N  = size(X, 1);
    rs = fr_seed(seed);
    [pidx, rs] = fr_randperm(rs, N, C);
    V = X(pidx, :);
    [U, ~] = fr_rand(rs, N, C);
    U = U ./ sum(U, 2);

    for it = 1:iters
        D = max(fr_pdist2(X, V), 1e-9);              % N-by-C distances
        % Equation (2): u_ik = [ sum_j (d_ik/d_ij)^(2/(m-1)) ]^-1
        Un = zeros(N, C);
        for k = 1:C
            Un(:, k) = 1 ./ sum((D(:, k) ./ D) .^ (2 / (m - 1)), 2);
        end
        % Equation (3): centroid update
        W = Un .^ m;
        V = (W' * X) ./ max(sum(W, 1)', 1e-9);
        if max(abs(Un(:) - U(:))) < 1e-5
            U = Un; break;
        end
        U = Un;
    end
end
