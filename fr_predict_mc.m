function [mu, sd] = fr_predict_mc(net, Gf, cols, adjName, T)
%FR_PREDICT_MC  Monte-Carlo dropout comparator for the uncertainty study.
%   Costs T stochastic forward passes, against zero for the closed-form
%   descriptor and for predictive entropy.
    if nargin < 5 || isempty(T), T = 20; end
    S = zeros(size(Gf.X, 1), T);
    for t = 1:T
        [P, ~, net] = fr_gcn_forward(net, Gf.X(:, cols), Gf.(adjName), true);
        S(:, t) = P(:, 2);
    end
    mu = mean(S, 2); sd = std(S, 0, 2);
end
