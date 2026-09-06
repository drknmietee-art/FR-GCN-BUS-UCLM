function A = fr_regionadj(L, n)
%FR_REGIONADJ  Region adjacency graph: nodes share at least one border pixel.
    A = false(n, n);
    a = L(:, 1:end-1); b = L(:, 2:end); d = a ~= b;
    A(sub2ind([n n], a(d), b(d))) = true;
    a = L(1:end-1, :); b = L(2:end, :); d = a ~= b;
    A(sub2ind([n n], a(d), b(d))) = true;
    A = A | A';
    A(logical(eye(n))) = false;
end
