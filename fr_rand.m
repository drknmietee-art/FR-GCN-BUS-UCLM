function [x, s] = fr_rand(s, m, n)
%FR_RAND  Uniform variates in (0,1) from the xorshift32 stream.
    if nargin < 3, n = 1; end
    if nargin < 2, m = 1; end
    k = m * n;
    x = zeros(k, 1);
    for i = 1:k
        s = bitxor(s, bitshift(s,  13));
        s = bitxor(s, bitshift(s, -17));
        s = bitxor(s, bitshift(s,   5));
        x(i) = (double(s) + 0.5) / 4294967296;
    end
    x = reshape(x, m, n);
end
