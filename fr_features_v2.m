function F = fr_features_v2(G)
%FR_FEATURES_V2  The fourteen additional node descriptors (Section 4.6).
%   Motivated by the descriptor-ceiling diagnostic: with only the six
%   intensity and texture statistics, a gradient-boosted forest matches the
%   graph network and neither generalises, so the descriptor is the binding
%   constraint.  The additions follow how a lesion is actually read on
%   ultrasound: echogenicity is relative rather than absolute, and posterior
%   acoustic shadowing or enhancement is among the strongest clinical cues.
    gray = G.gray; L = G.L; n = G.n;
    res  = size(gray, 1);
    med  = median(gray(:));
    iqr_ = max(fr_prctile(gray(:), 75) - fr_prctile(gray(:), 25), 1e-6);
    g4   = fr_gauss(gray, 4);
    g12  = fr_gauss(gray, 12);
    gx = [gray(:,2:end), gray(:,end)] - [gray(:,1), gray(:,1:end-1)];
    gy = [gray(2:end,:); gray(end,:)] - [gray(1,:); gray(1:end-1,:)];
    gm = sqrt(gx.^2 + gy.^2);
    [YY, XX] = ndgrid(1:res, 1:res);

    F = zeros(n, 14);
    means = zeros(n, 1); yb = zeros(n, 4);
    for k = 1:n
        sel = (L == k);
        means(k) = mean(gray(sel));
        ys = YY(sel); xs = XX(sel);
        yb(k,:) = [min(ys) max(ys) min(xs) max(xs)];
        F(k,3) = mean(ys) / res;                       % depth
        F(k,4) = mean(xs) / res;                       % lateral position
        F(k,5) = sum(sel(:)) / (res*res) * 1000;       % relative area
        F(k,6) = sum(sel(:)) / max((yb(k,2)-yb(k,1)+1)*(yb(k,4)-yb(k,3)+1), 1);
        F(k,7) = mean(gm(sel));                        % edge strength
        F(k,8) = mean(g4(sel))  - means(k);            % local multiscale contrast
        F(k,9) = mean(g12(sel)) - means(k);            % regional multiscale contrast
    end
    [~, ordm] = sort(means); rank_ = zeros(n,1); rank_(ordm) = (0:n-1) / max(n-1,1);
    F(:,1) = (means - med) / iqr_;                     % relative echogenicity
    F(:,2) = rank_;                                    % intensity percentile rank

    for k = 1:n
        h = yb(k,2) - yb(k,1) + 1;
        py0 = yb(k,2) + 1; py1 = min(yb(k,2) + h, res);
        if py1 >= py0
            strip = gray(py0:py1, yb(k,3):yb(k,4));
            F(k,10) = mean(strip(:)) - means(k);       % posterior behaviour
            F(k,11) = std(strip(:), 1);
        end
        ay0 = max(yb(k,1) - h, 1); ay1 = yb(k,1) - 1;
        if ay1 >= ay0
            strip = gray(ay0:ay1, yb(k,3):yb(k,4));
            F(k,12) = mean(strip(:)) - means(k);       % anterior control
        end
    end

    A  = G.adj;
    A2 = (double(A) * double(A)) > 0 | A;
    A2(logical(eye(n))) = false;
    nb1 = means; nb2 = means;
    for k = 1:n
        if any(A(k,:)),  nb1(k) = mean(means(A(k,:)));  end
        if any(A2(k,:)), nb2(k) = mean(means(A2(k,:))); end
    end
    F(:,13) = means - nb1;                             % 1-hop contrast
    F(:,14) = means - nb2;                             % 2-hop contrast
end

function out = fr_gauss(im, sigma)
%   Separable Gaussian blur; no toolbox dependency.
    r = max(1, ceil(3 * sigma));
    x = -r:r;
    k = exp(-(x.^2) / (2 * sigma^2)); k = k / sum(k);
    p = [repmat(im(:,1), 1, r), im, repmat(im(:,end), 1, r)];
    p = conv2(p, k, 'same'); p = p(:, r+1:end-r);
    p = [repmat(p(1,:), r, 1); p; repmat(p(end,:), r, 1)];
    p = conv2(p, k', 'same');
    out = p(r+1:end-r, :);
end
