function p = soft_saturated_lorenz96_parameters(K,F,kappa)
%SOFT_SATURATED_LORENZ96_PARAMETERS Parameters for a K-state SI benchmark.
%
% The cyclic system is
%   z_i    = x_{i-1}(x_{i+1}-x_{i-2}),
%   xdot_i = z_i/sqrt(1+(z_i/kappa)^2) - x_i + F.
%
% Indices are interpreted periodically. K need not be a multiple of four.
% K>=4 keeps the four local roles distinct. F and kappa are explicit case
% parameters supplied by the demo. They are required explicitly so the true
% model cannot silently fall back to a second parameter source.

    if nargin < 3 || isempty(K) || isempty(F) || isempty(kappa)
        error('K, F, and kappa must all be supplied explicitly.');
    end

    K = round(double(K));
    F = double(F);
    kappa = double(kappa);
    assert(isfinite(K) && K>=4, 'Lorenz--96 requires an integer K>=4.');
    assert(isscalar(F) && isfinite(F), 'Lorenz--96 forcing F must be a finite scalar.');
    assert(isscalar(kappa) && isfinite(kappa) && kappa>0, ...
        'The saturation parameter kappa must be a positive finite scalar.');

    p = struct();
    p.K = K;
    p.F = F;
    p.kappa = kappa;
    p.modelVariant = 'soft_saturated_lorenz96';
    p.stateNames = arrayfun(@(i) sprintf('x%d',i),1:p.K,'UniformOutput',false);
    p.derivativeNames = arrayfun(@(i) sprintf('x%d_dot',i),1:p.K,'UniformOutput',false);
    p.xEquilibrium = p.F*ones(1,p.K);
end
