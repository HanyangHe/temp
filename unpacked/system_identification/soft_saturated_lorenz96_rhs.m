function dX = soft_saturated_lorenz96_rhs(X,p)
%SOFT_SATURATED_LORENZ96_RHS Vectorized soft-saturated Lorenz--96 map.
%
% X is N-by-K and dX has the same size. Cyclic indices implement
% z_i=x_{i-1}(x_{i+1}-x_{i-2}) and
% xdot_i=z_i/sqrt(1+(z_i/kappa)^2)-x_i+F.

    if nargin < 2 || isempty(p)
        error('The parameter structure p must be supplied explicitly.');
    end
    if ~isnumeric(X) || size(X,2) ~= p.K
        error('soft_saturated_lorenz96_rhs expects an N-by-%d input.',p.K);
    end

    dX = zeros(size(X));
    for i = 1:p.K
        im2 = mod(i-3,p.K)+1;
        im1 = mod(i-2,p.K)+1;
        ip1 = mod(i,p.K)+1;
        z = X(:,im1).*(X(:,ip1)-X(:,im2));
        dX(:,i) = z./sqrt(1+(z./p.kappa).^2)-X(:,i)+p.F;
    end
end
