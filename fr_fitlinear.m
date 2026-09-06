function mdl = fr_fitlinear(X, y, iters, lr)
%FR_FITLINEAR  Class-balanced linear classifier (logistic, Adam) used as the
%   SLIC+SVM-style baseline.  Balanced weights keep it from being a straw man.
%   Self-contained so no Statistics Toolbox licence is required; if you have
%   one, fitclinear with 'Weights' gives an equivalent baseline.
    if nargin < 3 || isempty(iters), iters = 400;  end
    if nargin < 4 || isempty(lr),    lr    = 0.05; end
    y = double(y(:) > 0.5);
    mu = mean(X, 1); sd = max(std(X, 0, 1), 1e-9);
    Z  = [(X - mu) ./ sd, ones(size(X,1), 1)];
    npos = max(sum(y), 1); nneg = max(sum(1-y), 1);
    w = y * (nneg/npos) + (1-y);              % class-balanced weights
    th = zeros(size(Z,2), 1);
    m = zeros(size(th)); v = zeros(size(th));
    for t = 1:iters
        p = 1 ./ (1 + exp(-Z * th));
        g = Z' * (w .* (p - y)) / sum(w);
        m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*(g.^2);
        th = th - lr * (m/(1-0.9^t)) ./ (sqrt(v/(1-0.999^t)) + 1e-8);
    end
    mdl.theta = th; mdl.mu = mu; mdl.sd = sd;
end
