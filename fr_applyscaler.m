function Y = fr_applyscaler(F, lo, hi)
%FR_APPLYSCALER  Apply the training-fold scaler, clipped to [0,1].
    Y = (F - lo) ./ max(hi - lo, 1e-9);
    Y = min(max(Y, 0), 1);
end
