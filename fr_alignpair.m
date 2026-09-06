function [a, b] = fr_alignpair(ia, va, ib, vb)
%FR_ALIGNPAIR  Align two per-image score vectors on their shared images, so
%   every comparison in the paper is paired at the image level.
    [tf, loc] = ismember(ia, ib);
    a = va(tf); b = vb(loc(tf));
    ok = ~isnan(a) & ~isnan(b);
    a = a(ok); b = b(ok);
end
