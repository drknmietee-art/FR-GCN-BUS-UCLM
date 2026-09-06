function r = fr_tiedrank(x)
%FR_TIEDRANK  Ranks with averaged ties.
    x = x(:); n = numel(x);
    [xs, ord] = sort(x);
    r = zeros(n, 1); i = 1;
    while i <= n
        j = i;
        while j < n && xs(j+1) == xs(i), j = j + 1; end
        r(ord(i:j)) = (i + j) / 2;
        i = j + 1;
    end
end
