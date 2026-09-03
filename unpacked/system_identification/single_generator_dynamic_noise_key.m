function [noiseKey,info] = single_generator_dynamic_noise_key(noiseLevel)
%SINGLE_GENERATOR_DYNAMIC_NOISE_KEY Canonical generator noise-storage key.
% New low-noise manuscript grid uses per-mille naming:
%   0      -> rho_000_permil
%   0.001  -> rho_001_permil   (0.1%)
%   0.005  -> rho_005_permil   (0.5%)
%   0.01   -> rho_010_permil   (1.0%)
% Historical larger integer-percent cases retain the old percent convention:
%   0.05   -> rho_005_pct      (5%)
%   0.10   -> rho_010_pct      (10%)

    rho = double(noiseLevel);
    tol = 1e-12;
    if ~isscalar(rho) || ~isfinite(rho) || rho < 0
        error('noiseLevel must be a finite nonnegative scalar.');
    end

    permilleValue = 1000*rho;
    if abs(permilleValue-round(permilleValue)) <= tol && permilleValue <= 10+tol
        integerPermille = round(permilleValue);
        noiseKey = sprintf('rho_%03d_permil',integerPermille);
        info = struct('unit','permil','encodedValue',integerPermille, ...
            'noiseLevel',rho,'noisePercent',100*rho,'noisePermille',permilleValue);
        return;
    end

    percentValue = 100*rho;
    if abs(percentValue-round(percentValue)) <= tol
        integerPercent = round(percentValue);
        noiseKey = sprintf('rho_%03d_pct',integerPercent);
        info = struct('unit','pct','encodedValue',integerPercent, ...
            'noiseLevel',rho,'noisePercent',percentValue,'noisePermille',1000*rho);
        return;
    end

    error(['Unsupported noiseLevel %.15g. Use an integer per-mille level up to ', ...
        '10 permille, or an integer-percent historical level.'],rho);
end
