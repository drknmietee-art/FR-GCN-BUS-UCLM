function ci = fr_bca(x, B, alpha, seed)
%FR_BCA  Bias-corrected and accelerated bootstrap CI for the mean.
%   Efron & Tibshirani (1993).  Self-contained; no Statistics Toolbox.
    if nargin < 2 || isempty(B),     B     = 10000;     end
    if nargin < 3 || isempty(alpha), alpha = 0.05;      end
    if nargin < 4 || isempty(seed),  seed  = 20260905;  end
    x = x(:); n = numel(x);
    if n < 2, ci = [NaN NaN]; return; end
    th = mean(x);
    s  = fr_seed(seed);
    bs = zeros(B, 1);
    for b = 1:B
        [u, s] = fr_rand(s, n, 1);
        bs(b) = mean(x(min(floor(u * n) + 1, n)));
    end
    z0 = fr_norminv(min(max(mean(bs < th), 1e-9), 1 - 1e-9));
    jk = (sum(x) - x) / (n - 1);
    d  = mean(jk) - jk;
    a  = sum(d.^3) / (6 * (sum(d.^2))^1.5 + 1e-12);
    ci = zeros(1, 2); q = [alpha/2, 1 - alpha/2];
    for i = 1:2
        z = fr_norminv(q(i));
        ci(i) = fr_prctile(bs, 100 * fr_normcdf(z0 + (z0 + z) / (1 - a * (z0 + z))));
    end
end
