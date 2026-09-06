function Lout = fr_enforce_connectivity(L, minsize)
%FR_ENFORCE_CONNECTIVITY  Split disconnected labels and absorb tiny regions.
%   Flood fill (4-connected) without any toolbox dependency.
    [H, W] = size(L);
    Lout = zeros(H, W);
    next = 0;
    stack = zeros(H*W, 1);
    for s = 1:H*W
        if Lout(s) ~= 0, continue; end
        next = next + 1;
        lab = L(s);
        top = 1; stack(1) = s; Lout(s) = next; n = 0;
        comp = zeros(H*W, 1);
        while top > 0
            p = stack(top); top = top - 1;
            n = n + 1; comp(n) = p;
            [r, c] = ind2sub([H W], p);
            nb = [];
            if r > 1, nb(end+1) = p - 1;  end %#ok<AGROW>
            if r < H, nb(end+1) = p + 1;  end %#ok<AGROW>
            if c > 1, nb(end+1) = p - H;  end %#ok<AGROW>
            if c < W, nb(end+1) = p + H;  end %#ok<AGROW>
            for q = nb
                if Lout(q) == 0 && L(q) == lab
                    Lout(q) = next; top = top + 1; stack(top) = q;
                end
            end
        end
        % absorb a too-small component into an already-labelled neighbour
        if n < minsize && next > 1
            adopt = 0;
            for i = 1:n
                p = comp(i); [r, c] = ind2sub([H W], p);
                cand = [];
                if r > 1, cand(end+1) = p - 1; end %#ok<AGROW>
                if r < H, cand(end+1) = p + 1; end %#ok<AGROW>
                if c > 1, cand(end+1) = p - H; end %#ok<AGROW>
                if c < W, cand(end+1) = p + H; end %#ok<AGROW>
                for q = cand
                    if Lout(q) ~= 0 && Lout(q) ~= next, adopt = Lout(q); break; end
                end
                if adopt, break; end
            end
            if adopt
                Lout(comp(1:n)) = adopt;
                next = next - 1;
            end
        end
    end
    % compact the label set to 1..K
    u = unique(Lout);
    remap = zeros(max(u), 1);
    remap(u) = 1:numel(u);
    Lout = remap(Lout);
end
