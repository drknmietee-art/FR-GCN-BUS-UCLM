function [X, Y] = fr_stack(graphs, idx, R)
%FR_STACK  Build 4-D image and label arrays for the dense baselines.
%   X is R-by-R-by-1-by-N single in [0,1]; Y is the matching uint8 0/1 mask.
    N = numel(idx);
    X = zeros(R, R, 1, N, 'single');
    Y = zeros(R, R, 1, N, 'uint8');
    for j = 1:N
        G = graphs{idx(j)};
        X(:,:,1,j) = single(imresize(G.gray, [R R], 'bilinear'));
        Y(:,:,1,j) = uint8(imresize(double(G.gt), [R R], 'nearest') > 0.5);
    end
    X = min(max(X, 0), 1);
end
