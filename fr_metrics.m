function m = fr_metrics(pred, gt)
%FR_METRICS  Overlap metrics of Equations (15)-(19).
%   An empty prediction scores zero on Dice, Jaccard, sensitivity and
%   precision rather than being discarded, so the model's worst failures stay
%   in the average (Section 5.1).
    pred = logical(pred); gt = logical(gt);
    tp = sum(pred(:) &  gt(:));
    fp = sum(pred(:) & ~gt(:));
    fn = sum(~pred(:) &  gt(:));
    tn = sum(~pred(:) & ~gt(:));
    m.dice        = safediv(2*tp, 2*tp + fp + fn);
    m.jaccard     = safediv(tp,   tp + fp + fn);
    m.sensitivity = safediv(tp,   tp + fn);
    m.specificity = safediv(tn,   tn + fp);
    m.precision   = safediv(tp,   tp + fp);
end
function v = safediv(a, b)
    if b > 0, v = a / b; else, v = 0; end
end
