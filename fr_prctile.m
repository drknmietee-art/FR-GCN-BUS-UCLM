function q = fr_prctile(x, p)
%FR_PRCTILE  Linear-interpolation percentile (matches numpy.percentile).
    x = sort(x(:));
    n = numel(x);
    if n == 0, q = NaN; return; end
    if n == 1, q = x(1); return; end
    pos = (p / 100) * (n - 1) + 1;
    lo  = floor(pos); hi = ceil(pos); f = pos - lo;
    q   = x(lo) * (1 - f) + x(hi) * f;
end
