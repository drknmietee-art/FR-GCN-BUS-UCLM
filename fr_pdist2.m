function D = fr_pdist2(A, B)
%FR_PDIST2  Euclidean distances between rows of A and rows of B.
%   Dependency-free replacement for pdist2 (Statistics Toolbox).
    D = sqrt(max(sum(A.^2, 2) + sum(B.^2, 2)' - 2 * (A * B'), 0));
end
