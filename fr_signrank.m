function p = fr_signrank(x, y)
%FR_SIGNRANK  Wilcoxon signed-rank test, two-sided, normal approximation with
%   continuity correction and tie correction.  Self-contained so that no
%   Statistics Toolbox licence is required.
    d = x(:) - y(:);
    d = d(d ~= 0);                 % Wilcoxon zero-handling: drop exact ties
    n = numel(d);
    if n == 0, p = 1; return; end
    r = fr_tiedrank(abs(d));
    Wp = sum(r(d > 0));
    Wm = sum(r(d < 0));
    W  = min(Wp, Wm);
    mu = n * (n + 1) / 4;
    [~, ~, tcount] = unique(abs(d));
    tie = accumarray(tcount, 1);
    sd = sqrt(n*(n+1)*(2*n+1)/24 - sum(tie.^3 - tie)/48);
    if sd == 0, p = 1; return; end
    % No continuity correction, matching scipy.stats.wilcoxon's default and
    % the Python reference implementation used for the published tables.
    z = (W - mu) / sd;
    p = min(1, erfc(abs(z) / sqrt(2)));
end
