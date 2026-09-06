function P = fr_softmax(Z)
%FR_SOFTMAX  Row-wise softmax, shifted for numerical stability.
    Z = Z - max(Z, [], 2);
    E = exp(Z);
    P = E ./ max(sum(E, 2), 1e-12);
end
