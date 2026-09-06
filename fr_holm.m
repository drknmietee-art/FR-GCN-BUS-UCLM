function adj = fr_holm(p)
%FR_HOLM  Holm step-down multiplicity adjustment (Holm 1979).
    p = p(:); m = numel(p);
    [~, ord] = sort(p);
    adj = zeros(m, 1); run = 0;
    for i = 1:m
        k = ord(i);
        run = max(run, (m - i + 1) * p(k));
        adj(k) = min(run, 1);
    end
end
