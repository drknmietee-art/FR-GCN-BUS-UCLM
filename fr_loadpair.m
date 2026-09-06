function [g, lesion, kind] = fr_loadpair(imgPath, maskPath, res)
%FR_LOADPAIR  Read an image/mask pair from BUS-UCLM.
%   Masks are RGB coded: green = benign, red = malignant, black = background.
%   Both classes are merged into a binary lesion label (Section 3).
    if nargin < 3 || isempty(res), res = 256; end
    I = imread(imgPath);
    if ndims(I) == 3, G = double(rgb2gray(I)); else, G = double(I); end
    if max(G(:)) <= 1, G = G * 255; end
    g = imresize(G / 255, [res res], 'bilinear');
    g = min(max(g, 0), 1);

    M = imread(maskPath);
    if ndims(M) == 2, M = repmat(M, [1 1 3]); end
    M = double(M);
    % Masks are normally uint8 RGB, but a two-colour PNG can be read back as
    % logical or as a 0..1 image; rescale so the thresholds below always apply.
    if max(M(:)) <= 1, M = M * 255; end
    green = M(:,:,2) > 127 & M(:,:,1) < 128 & M(:,:,3) < 128;
    red   = M(:,:,1) > 127 & M(:,:,2) < 128 & M(:,:,3) < 128;
    les   = green | red;
    lesion = imresize(double(les), [res res], 'nearest') > 0.5;

    if sum(red(:)) > sum(green(:))
        kind = 'malignant';
    elseif sum(green(:)) > 0
        kind = 'benign';
    else
        kind = 'none';
    end
end
