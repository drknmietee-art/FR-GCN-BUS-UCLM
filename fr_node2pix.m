function mask = fr_node2pix(post, G, tau)
%FR_NODE2PIX  Broadcast node posteriors to pixels and threshold (Eq. 14).
    if nargin < 3 || isempty(tau), tau = 0.5; end
    mask = post(G.L) > tau;
end
