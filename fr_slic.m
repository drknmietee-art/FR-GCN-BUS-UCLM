function L = fr_slic(img, nseg, compactness, iters)
%FR_SLIC  SLIC superpixels for a grayscale image (Achanta et al., 2012).
%   Implemented directly so the pipeline does not require the Image
%   Processing Toolbox and so the result is identical in MATLAB and Octave.
%   IMG must be in [0,1].  Labels are 1..K, spatially connected.
    if nargin < 2 || isempty(nseg),        nseg        = 300;  end
    if nargin < 3 || isempty(compactness), compactness = 0.30; end
    if nargin < 4 || isempty(iters),       iters       = 10;   end

    [H, W] = size(img);
    S = max(round(sqrt(H * W / nseg)), 2);           % grid step
    [gy, gx] = meshgrid(1:W, 1:H);

    % --- cluster centres on a regular grid, nudged off local gradient maxima
    cy = round(S/2):S:H;  cx = round(S/2):S:W;
    [CY, CX] = ndgrid(cy, cx);
    CY = CY(:); CX = CX(:);
    Gm = fr_gradmag(img);
    for k = 1:numel(CY)
        y0 = max(CY(k)-1,1):min(CY(k)+1,H);
        x0 = max(CX(k)-1,1):min(CX(k)+1,W);
        sub = Gm(y0, x0);
        [~, mi] = min(sub(:));
        [a, b] = ind2sub(size(sub), mi);
        CY(k) = y0(a); CX(k) = x0(b);
    end
    C = [img(sub2ind([H W], CY, CX)), CY, CX];       % [intensity y x]
    K = size(C, 1);

    L = zeros(H, W); D = inf(H, W);
    m = compactness;
    for it = 1:iters
        L(:) = 0; D(:) = inf;
        for k = 1:K
            y0 = max(round(C(k,2)-S),1):min(round(C(k,2)+S),H);
            x0 = max(round(C(k,3)-S),1):min(round(C(k,3)+S),W);
            di = img(y0,x0) - C(k,1);
            dy = gx(y0,x0) - C(k,2);
            dx = gy(y0,x0) - C(k,3);
            % SLIC distance: intensity + (m/S) * spatial
            dist = di.^2 + (m/S)^2 * (dy.^2 + dx.^2);
            sub  = D(y0,x0);
            upd  = dist < sub;
            sub(upd) = dist(upd);
            D(y0,x0) = sub;
            lab = L(y0,x0); lab(upd) = k; L(y0,x0) = lab;
        end
        L(L == 0) = 1;
        % --- recompute centres
        cnt = accumarray(L(:), 1, [K 1]);
        si  = accumarray(L(:), img(:), [K 1]);
        sy  = accumarray(L(:), gx(:),  [K 1]);
        sx  = accumarray(L(:), gy(:),  [K 1]);
        ok  = cnt > 0;
        C(ok,1) = si(ok)./cnt(ok);
        C(ok,2) = sy(ok)./cnt(ok);
        C(ok,3) = sx(ok)./cnt(ok);
    end
    L = fr_enforce_connectivity(L, round(0.25 * S * S));
end

function G = fr_gradmag(im)
    gx = [im(:,2:end), im(:,end)] - [im(:,1), im(:,1:end-1)];
    gy = [im(2:end,:); im(end,:)] - [im(1,:); im(1:end-1,:)];
    G  = sqrt(gx.^2 + gy.^2);
end
