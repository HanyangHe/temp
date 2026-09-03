function audit_lorenz96_low_noise_storage_keys()
%AUDIT_LORENZ96_LOW_NOISE_STORAGE_KEYS Verify the low-noise per-mille keys.
levels = [0,0.001,0.005,0.01];
expected = {'rho_000_permil','rho_001_permil','rho_005_permil','rho_010_permil'};
actual = arrayfun(@(rho) soft_saturated_lorenz96_noise_key(rho),levels,'UniformOutput',false);
assert(isequal(actual,expected),'Unexpected Lorenz--96 low-noise storage-key mapping.');
fprintf('Lorenz--96 low-noise storage keys verified:\n');
for k = 1:numel(levels)
    fprintf('  rho=%.6g (%.3g%%, %.3g per mille) -> %s\n', ...
        levels(k),100*levels(k),1000*levels(k),actual{k});
end
end
