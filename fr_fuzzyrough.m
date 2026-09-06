function [A, lower, upper, B] = fr_fuzzyrough(X, U, V, sigma, D2)
%FR_FUZZYROUGH  Fuzzy-rough approximations and boundary uncertainty.
%   Equations (4)-(6) with the Lukasiewicz implicator and its dual t-norm.
%
%   LESION-CLUSTER SELECTION (leakage fix).  The cluster treated as "lesion"
%   is the one with the LOWER MEAN-INTENSITY CENTROID -- the correct
%   unsupervised prior for hypoechoic breast lesions.  Earlier versions of
%   this code selected it by correlating the membership columns against the
%   ground-truth node labels, which leaked held-out labels into node
%   features because this function runs on test images too.
    if nargin < 4 || isempty(sigma), sigma = 0.30; end
    if nargin < 5 || isempty(D2)
        D2 = fr_pdist2(X, X) .^ 2;
    end

    [~, les] = min(V(:, 1));            % darker centroid = lesion
    A = U(:, les);

    R = exp(-D2 ./ (2 * sigma^2));      % Eq. (4): reflexive, symmetric

    N = size(X, 1);
    Arow = A(:)';                       % 1-by-N
    % Eq. (5a): lower(p) = inf_q min(1, 1 - R(p,q) + A(q))
    lower = min(min(1, 1 - R + repmat(Arow, N, 1)), [], 2);
    % Eq. (5b): upper(p) = sup_q max(0, R(p,q) + A(q) - 1)
    upper = max(max(0, R + repmat(Arow, N, 1) - 1), [], 2);
    % Eq. (6); Proposition 1 guarantees B in [0,1]
    B = min(max(upper - lower, 0), 1);
end
