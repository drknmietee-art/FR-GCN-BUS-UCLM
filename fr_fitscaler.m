function [lo, hi] = fr_fitscaler(featCell)
%FR_FITSCALER  Min-max statistics from the TRAINING FOLD ONLY (Section 4.6).
%   The statistics are then applied unchanged to the held-out fold, so no
%   test information enters the normalisation.
    X = vertcat(featCell{:});
    lo = min(X, [], 1);
    hi = max(X, [], 1);
end
