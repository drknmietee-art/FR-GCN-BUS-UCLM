function im = fr_anisodiff(img, niter, kappa, gamma)
%FR_ANISODIFF  Perona-Malik anisotropic diffusion (Section 4.2).
%   Edge-preserving speckle reduction.  An isotropic filter would blur the
%   lesion margin that the superpixel step is driven by.
    if nargin < 2 || isempty(niter), niter = 20;       end
    if nargin < 3 || isempty(kappa), kappa = 15/255;   end
    if nargin < 4 || isempty(gamma), gamma = 0.14;     end
    im = double(img);
    for t = 1:niter
        dN = [im(1,:); im(1:end-1,:)] - im;
        dS = [im(2:end,:); im(end,:)] - im;
        dW = [im(:,1), im(:,1:end-1)] - im;
        dE = [im(:,2:end), im(:,end)] - im;
        cN = exp(-(dN/kappa).^2);  cS = exp(-(dS/kappa).^2);
        cW = exp(-(dW/kappa).^2);  cE = exp(-(dE/kappa).^2);
        im = im + gamma * (cN.*dN + cS.*dS + cW.*dW + cE.*dE);
    end
end
