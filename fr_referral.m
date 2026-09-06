function d = fr_referral(rec, score, fracs)
%FR_REFERRAL  Selective prediction: replace the predictions of the most
%   uncertain fraction of image AREA with the reference label, which is the
%   idealised outcome of referring that area to a reader, and return Dice.
    npix = double(rec.npix(:)); y = rec.y(:);
    pred = double(rec.post(:) > 0.5);
    [~, ord] = sort(score(:), 'descend');
    cum = cumsum(npix(ord)) / sum(npix);
    d = zeros(numel(fracs), 1);
    for i = 1:numel(fracs)
        k = sum(cum < fracs(i));
        p = pred; p(ord(1:k)) = y(ord(1:k));
        tp = sum(npix .* p .* y);
        fp = sum(npix .* p .* (1 - y));
        fn = sum(npix .* (1 - p) .* y);
        d(i) = 2*tp / max(2*tp + fp + fn, 1e-9);
    end
end
