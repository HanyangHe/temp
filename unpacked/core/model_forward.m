function [Y, cache] = model_forward(X, Coef_M, arch, normOpt)
%MODEL_FORWARD Forward propagation with cache.
%
% Flexible-width PhDN convention:
%   dims = [nx, d2, ..., dL, ny].
% Each layer output h^{ell+1} has dimension dims(ell+1).

	if nargin < 4 || isempty(normOpt)
		normOpt = default_norm_options();
	end
	if isfield(normOpt, 'useLayerNorm') && normOpt.useLayerNorm
		error('Current implementation does not support hidden-layer normalization.');
	end

	dims = get_arch_dims(arch);

	Nsp = size(X, 1);
	Xn = apply_input_norm(X, normOpt);
	Xn(~isfinite(Xn)) = 0;

	if size(Xn, 2) ~= dims(1)
		error('Input dimension mismatch: X has %d columns, but dims(1) = %d.', size(Xn, 2), dims(1));
	end

	h = cell(1, arch.layer + 1);
	h{1} = Xn.';

	cache = struct();
	cache.h = cell(1, arch.layer + 1);
	cache.h{1} = h{1};
	cache.branch = cell(arch.layer, arch.layer);
	cache.tmpRaw = cell(1, arch.layer);
	cache.tmpClipMask = cell(1, arch.layer);
	cache.normOpt = normOpt;
	cache.dims = dims;

	for ell = 1:arch.layer
		rowDim = dims(ell + 1);
		tmpRaw = zeros(rowDim, Nsp);

		for src = 1:ell
			k = ell - src + 1;

			branch = build_branch_cache(h{k}, arch, ell, h, src);
			cache.branch{src, ell} = branch;

			A = Coef_M{src, ell};

			if size(A, 1) ~= rowDim
				error('Coef_M{%d,%d} row mismatch: got %d, expected %d.', ...
					src, ell, size(A, 1), rowDim);
			end
			if size(A, 2) ~= size(branch.Phi, 1)
				error('Coef_M{%d,%d} column mismatch: got %d, expected %d.', ...
					src, ell, size(A, 2), size(branch.Phi, 1));
			end

			tmpRaw = tmpRaw + A * branch.Phi;
		end

		tmpRaw(~isfinite(tmpRaw)) = 0;

		if ell == arch.layer
			clipBound = get_clip_bound(arch.safety, 'finalOutputClip', Inf);
		else
			clipBound = get_clip_bound(arch.safety, 'hiddenLayerOutputClip', ...
				get_clip_bound(arch.safety, 'layerOutputClip', Inf));
		end

		if isinf(clipBound)
			tmp = tmpRaw;
			clipMask = true(size(tmpRaw));
		else
			tmp = min(max(tmpRaw, -clipBound), clipBound);
			clipMask = abs(tmpRaw) < clipBound;
		end

		tmp(~isfinite(tmp)) = 0;

		h{ell + 1} = tmp;
		cache.h{ell + 1} = tmp;
		cache.tmpRaw{ell} = tmpRaw;
		cache.tmpClipMask{ell} = clipMask;
	end

	Ynorm = h{arch.layer + 1}.';
	Y = reverse_output_norm(Ynorm, normOpt);
	cache.Ynorm = Ynorm;
	cache.outputDenormDerivative = output_denorm_derivative(normOpt, dims(end));
end

function val = get_clip_bound(safety, fieldName, defaultVal)
	if isfield(safety, fieldName) && ~isempty(safety.(fieldName))
		val = safety.(fieldName);
	else
		val = defaultVal;
	end
end
