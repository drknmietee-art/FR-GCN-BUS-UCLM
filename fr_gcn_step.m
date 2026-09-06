function net = fr_gcn_step(net, grads, lr, b1, b2, eps0)
%FR_GCN_STEP  Adam update (Kingma & Ba, 2015).
    if nargin < 3 || isempty(lr),   lr   = 0.01;  end
    if nargin < 4 || isempty(b1),   b1   = 0.9;   end
    if nargin < 5 || isempty(b2),   b2   = 0.999; end
    if nargin < 6 || isempty(eps0), eps0 = 1e-8;  end
    net.t = net.t + 1;
    for i = 1:numel(net.W)
        net.m{i} = b1 * net.m{i} + (1 - b1) * grads{i};
        net.v{i} = b2 * net.v{i} + (1 - b2) * (grads{i} .^ 2);
        mh = net.m{i} / (1 - b1 ^ net.t);
        vh = net.v{i} / (1 - b2 ^ net.t);
        net.W{i} = net.W{i} - lr * mh ./ (sqrt(vh) + eps0);
    end
end
