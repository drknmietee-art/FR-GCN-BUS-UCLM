function e = fr_ece(prob, y, w, bins)
%FR_ECE  Pixel-weighted expected calibration error (Guo et al., 2017).
    if nargin < 4 || isempty(bins), bins = 15; end
    prob = prob(:); y = y(:) > 0.5; w = w(:);
    conf = max(prob, 1 - prob);
    corr = double((prob >= 0.5) == y);
    edges = linspace(0, 1, bins + 1);
    W = sum(w); e = 0;
    for i = 1:bins
        sel = conf > edges(i) & conf <= edges(i+1);
        if ~any(sel), continue; end
        ww = sum(w(sel));
        e = e + ww / W * abs(sum(corr(sel).*w(sel))/ww - sum(conf(sel).*w(sel))/ww);
    end
end
