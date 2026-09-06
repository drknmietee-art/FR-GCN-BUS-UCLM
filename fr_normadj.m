function Ahat = fr_normadj(W)
%FR_NORMADJ  Symmetric normalisation D^-1/2 W D^-1/2 (Equation 8).
%   Returned sparse: the region adjacency graph has mean degree ~6, so the
%   sparse operator is two orders of magnitude cheaper than the dense one.
%   Lemma 1: the spectrum of the result lies in (-1, 1].
    d  = sum(W, 2);
    di = 1 ./ sqrt(max(d, 1e-12));
    Ahat = sparse(W .* (di * di'));
end
