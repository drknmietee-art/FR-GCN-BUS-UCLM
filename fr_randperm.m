function [p, s] = fr_randperm(s, n, k)
%FR_RANDPERM  Fisher-Yates permutation from the deterministic stream.
    if nargin < 3 || isempty(k), k = n; end
    p = 1:n;
    for i = n:-1:2
        [u, s] = fr_rand(s, 1, 1);
        j = 1 + floor(u * i);
        if j > i, j = i; end
        tmp = p(i); p(i) = p(j); p(j) = tmp;
    end
    p = p(1:k);
end
