function [contrast, homogeneity, energy, correlation] = fr_maskedglcm(patchq, region, levels)
%FR_MASKEDGLCM  Grey-level co-occurrence statistics over a REGION only.
%   A pixel pair contributes only when BOTH pixels belong to the superpixel.
%   Computing the statistics over the axis-aligned bounding box instead, as is
%   sometimes done for convenience, mixes in neighbouring regions' pixels and,
%   at a lesion margin, pixels of the other class.
%   Offsets: unit distance, four orientations, symmetric, averaged.
    if nargin < 3 || isempty(levels), levels = 8; end
    offs = [0 1; -1 1; -1 0; -1 -1];
    P = zeros(levels, levels);
    [H, W] = size(patchq);
    for o = 1:size(offs, 1)
        dy = offs(o,1); dx = offs(o,2);
        ys0 = max(1, 1-dy); ys1 = min(H, H-dy);
        xs0 = max(1, 1-dx); xs1 = min(W, W-dx);
        if ys1 < ys0 || xs1 < xs0, continue; end
        a  = patchq(ys0:ys1,       xs0:xs1);
        b  = patchq(ys0+dy:ys1+dy, xs0+dx:xs1+dx);
        ma = region(ys0:ys1,       xs0:xs1);
        mb = region(ys0+dy:ys1+dy, xs0+dx:xs1+dx);
        ok = ma & mb;
        if ~any(ok(:)), continue; end
        ai = a(ok) + 1; bi = b(ok) + 1;                 % 1-based levels
        P = P + accumarray([ai bi], 1, [levels levels]) ...
              + accumarray([bi ai], 1, [levels levels]); % symmetric
    end
    s = sum(P(:));
    if s <= 0
        contrast = 0; homogeneity = 1; energy = 1; correlation = 0; return;
    end
    P = P / s;
    [i, j] = ndgrid(0:levels-1, 0:levels-1);
    contrast    = sum(sum(P .* (i - j).^2));
    homogeneity = sum(sum(P ./ (1 + (i - j).^2)));
    energy      = sum(sum(P.^2));
    px = sum(P, 2);
    mu = sum((0:levels-1)' .* px);
    sd = sqrt(sum(((0:levels-1)' - mu).^2 .* px));
    if sd < 1e-9
        correlation = 0;
    else
        correlation = sum(sum(P .* (i - mu) .* (j - mu))) / (sd * sd);
    end
end
