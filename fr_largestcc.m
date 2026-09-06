function out = fr_largestcc(mask)
%FR_LARGESTCC  Keep the largest connected component (Section 8.8 variant).
    mask = logical(mask);
    if ~any(mask(:)), out = mask; return; end
    L = bwlabel(mask, 8);
    counts = accumarray(L(L > 0), 1);
    [~, k] = max(counts);
    out = (L == k);
end
