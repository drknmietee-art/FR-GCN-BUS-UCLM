function [auc, ap] = fr_auroc(labels, scores, w)
%FR_AUROC  Weighted ROC area and average precision, positives = label 1.
    labels = labels(:) > 0.5; scores = scores(:);
    if nargin < 3 || isempty(w), w = ones(size(scores)); end
    w = w(:);
    [~, ord] = sort(scores, 'descend');
    l = labels(ord); ww = w(ord);
    P = sum(ww(l)); N = sum(ww(~l));
    if P == 0 || N == 0, auc = NaN; ap = NaN; return; end
    tp = cumsum(ww .*  l);
    fp = cumsum(ww .* ~l);
    tpr = tp / P; fpr = fp / N;
    auc = trapz([0; fpr], [0; tpr]);
    prec = tp ./ max(tp + fp, 1e-12);
    rec  = tpr;
    ap   = sum(diff([0; rec]) .* prec);
end
