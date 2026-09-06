function r = fr_record(method, seed, fold, G, pred)
%FR_RECORD  One per-image result row.
    m = fr_metrics(pred, G.gt);
    r = {method, seed, fold, G.name, m.dice, m.jaccard, m.sensitivity, ...
         m.specificity, m.precision, fr_hd95(pred, G.gt)};
end
