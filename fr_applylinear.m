function p = fr_applylinear(mdl, X)
%FR_APPLYLINEAR  Probability from FR_FITLINEAR.
    Z = [(X - mdl.mu) ./ mdl.sd, ones(size(X,1), 1)];
    p = 1 ./ (1 + exp(-Z * mdl.theta));
end
