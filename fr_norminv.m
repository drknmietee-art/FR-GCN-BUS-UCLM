function z = fr_norminv(p)
%FR_NORMINV  Standard normal quantile function.
    z = -sqrt(2) * erfcinv(2 * min(max(p, 1e-12), 1 - 1e-12));
end
