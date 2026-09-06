function R = fr_boundaryoverlay(gray, L)
%FR_BOUNDARYOVERLAY  Draw superpixel borders on a grayscale image.
    b = false(size(L));
    b(:,1:end-1) = b(:,1:end-1) | (L(:,1:end-1) ~= L(:,2:end));
    b(1:end-1,:) = b(1:end-1,:) | (L(1:end-1,:) ~= L(2:end,:));
    R = repmat(gray, [1 1 3]);
    R(:,:,1) = min(R(:,:,1) + b, 1);
    R(:,:,2) = min(R(:,:,2) + 0.85*b, 1);
    R(:,:,3) = R(:,:,3) .* ~b;
end
