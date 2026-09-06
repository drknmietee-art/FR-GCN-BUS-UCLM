function d = fr_hd95(pred, gt)
%FR_HD95  Symmetric 95th-percentile Hausdorff distance in pixels, Eq. (20).
%   NaN when either boundary set is empty (Section 5.1).
    pred = logical(pred); gt = logical(gt);
    if ~any(pred(:)) || ~any(gt(:)), d = NaN; return; end
    bp = pred & ~imerode_local(pred);
    bg = gt   & ~imerode_local(gt);
    if ~any(bp(:)) || ~any(bg(:)), d = NaN; return; end
    Dg = bwdist(bg);            % distance to the reference boundary
    Dp = bwdist(bp);            % distance to the predicted boundary
    d  = max(fr_prctile(Dg(bp), 95), fr_prctile(Dp(bg), 95));
end
function e = imerode_local(M)
%   4-connected erosion, so the function needs no morphological toolbox.
    e = M;
    e(2:end,:)   = e(2:end,:)   & M(1:end-1,:);
    e(1:end-1,:) = e(1:end-1,:) & M(2:end,:);
    e(:,2:end)   = e(:,2:end)   & M(:,1:end-1);
    e(:,1:end-1) = e(:,1:end-1) & M(:,2:end);
    e(1,:) = false; e(end,:) = false; e(:,1) = false; e(:,end) = false;
end
