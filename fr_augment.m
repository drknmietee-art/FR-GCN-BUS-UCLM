function [xa, ya, s] = fr_augment(x, y, s)
%FR_AUGMENT  One random flip / rotation / intensity jitter of an image-label
%   pair, using the deterministic stream so augmentation is reproducible.
%   X is H-by-W single, Y is H-by-W numeric (0/1).
    [u, s] = fr_rand(s, 5, 1);
    xa = x; ya = y;
    if u(1) < 0.5, xa = fliplr(xa); ya = fliplr(ya); end
    if u(2) < 0.5, xa = flipud(xa); ya = flipud(ya); end
    if u(3) < 0.5
        k = 1 + floor(u(4) * 3); if k > 3, k = 3; end
        xa = rot90(xa, k); ya = rot90(ya, k);
    end
    if u(5) < 0.3
        [v, s] = fr_rand(s, 2, 1);
        xa = min(max(xa * (0.85 + 0.30*v(1)) + (v(2) - 0.5) * 0.12, 0), 1);
    end
end
