function [x, s] = fr_randn(s, m, n)
%FR_RANDN  Standard normal variates by the Box-Muller transform.
    if nargin < 3, n = 1; end
    if nargin < 2, m = 1; end
    k  = m * n;
    k2 = 2 * ceil(k / 2);
    [u, s] = fr_rand(s, k2, 1);
    u1 = u(1:2:end);  u2 = u(2:2:end);
    r  = sqrt(-2 * log(max(u1, 1e-12)));
    z  = [r .* cos(2*pi*u2); r .* sin(2*pi*u2)];
    x  = reshape(z(1:k), m, n);
end
