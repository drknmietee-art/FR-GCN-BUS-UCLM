function post = fr_predict(net, Gf, cols, adjName)
%FR_PREDICT  Lesion posterior per node.
    P = fr_gcn_forward(net, Gf.X(:, cols), Gf.(adjName), false);
    post = P(:, 2);
end
