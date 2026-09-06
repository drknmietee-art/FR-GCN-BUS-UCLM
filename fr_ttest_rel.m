function p = fr_ttest_rel(x, y)
%FR_TTEST_REL  Two-sided paired t-test p-value (no toolbox required).
    d = x(:) - y(:); n = numel(d);
    if n < 2, p = 1; return; end
    s = std(d);
    if s == 0, p = 1; return; end
    t  = mean(d) / (s / sqrt(n));
    df = n - 1;
    p  = betainc(df / (df + t^2), df/2, 0.5);   % 2-sided Student-t tail
end
