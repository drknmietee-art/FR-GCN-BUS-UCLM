function [loss, dZ] = fr_loss(P, y, bnd, alpha, lambda, use_unc)
%FR_LOSS  Uncertainty-weighted cross-entropy plus soft Dice, Eqs (11)-(13).
%   Returns the objective and its gradient with respect to the softmax input.
    if nargin < 4 || isempty(alpha),   alpha   = 0.5;  end
    if nargin < 5 || isempty(lambda),  lambda  = 1.0;  end
    if nargin < 6 || isempty(use_unc), use_unc = true; end

    y = y(:); bnd = bnd(:);
    N = numel(y);
    Y = [1 - y, y];
    if use_unc, w = 1 + lambda * bnd; else, w = ones(N, 1); end

    ce  = -sum(w .* sum(Y .* log(P + 1e-8), 2)) / N;   % Eq. (11)
    dce = (w .* (P - Y)) / N;

    pl  = P(:, 2);                                      % Eq. (12)
    num = 2 * sum(pl .* y) + 1;
    den = sum(pl) + sum(y) + 1;
    dice_loss = 1 - num / den;
    dpl = -(2 * y * den - num) / (den^2);
    dsm = [-dpl .* pl .* P(:,1), dpl .* pl .* (1 - pl)];

    loss = alpha * ce + (1 - alpha) * dice_loss;        % Eq. (13)
    dZ   = alpha * dce + (1 - alpha) * dsm;
end
