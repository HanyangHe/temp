function [noiseKey,info] = soft_saturated_lorenz96_noise_key(noiseLevel)
%SOFT_SATURATED_LORENZ96_NOISE_KEY Per-mille rho-folder naming convention.
%
% The low-noise robustness sweep uses per-mille units in the folder suffix:
%   rho = 0       = 0.0% =  0 per mille -> rho_000_permil
%   rho = 0.001   = 0.1% =  1 per mille -> rho_001_permil
%   rho = 0.005   = 0.5% =  5 per mille -> rho_005_permil
%   rho = 0.01    = 1.0% = 10 per mille -> rho_010_permil
%
% This avoids confusing the historical *_pct convention, where the numeric
% field denotes integer percent (for example rho_010_pct means 10%).

    rho = double(noiseLevel);
    if ~isscalar(rho) || ~isfinite(rho) || rho < 0
        error('noiseLevel must be one finite nonnegative scalar.');
    end

    perMille = 1000*rho;
    roundedPerMille = round(perMille);
    tolPerMille = 1e-10*max(1,abs(perMille));
    if abs(perMille-roundedPerMille) > tolPerMille
        error(['Noise level %.16g is not exactly representable by the ', ...
            'integer per-mille storage convention.'],rho);
    end

    noiseKey = sprintf('rho_%03d_permil',roundedPerMille);

    info = struct();
    info.noiseLevel = rho;
    info.noisePercent = 100*rho;
    info.perMille = roundedPerMille;
    info.scheme = 'integer_permille';
end
