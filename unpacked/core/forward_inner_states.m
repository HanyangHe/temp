function h = forward_inner_states(X, Coef_M, arch, normOpt)
%FORWARD_INNER_STATES Forward propagate up to h{layer}; final layer skipped.
%
% Updated for layer-wise hidden dimensions. This is used by PhDN model evaluation. It returns:
%   h{1}, h{2}, ..., h{layer}
% but does not compute h{layer+1}.

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

	h = cell(1, arch.layer);
	h{1} = Xn.';

	for ell = 1:(arch.layer - 1)
		rowDim = dims(ell + 1);
		tmpRaw = zeros(rowDim, Nsp);

		for src = 1:ell
			k = ell - src + 1;
			branch = build_branch_cache(h{k}, arch, ell, h, src);
			A = Coef_M{src, ell};

			if size(A, 1) ~= rowDim
				error('Coef_M{%d,%d} row mismatch: got %d, expected %d.', ...
					src, ell, size(A, 1), rowDim);
			end

			tmpRaw = tmpRaw + A * branch.Phi;
		end

		tmpRaw(~isfinite(tmpRaw)) = 0;

		clipBound = get_clip_bound(arch.safety, 'hiddenLayerOutputClip', ...
			get_clip_bound(arch.safety, 'layerOutputClip', Inf));

		if isinf(clipBound)
			tmp = tmpRaw;
		else
			tmp = min(max(tmpRaw, -clipBound), clipBound);
		end

		tmp(~isfinite(tmp)) = 0;
		h{ell + 1} = tmp;
	end
end

function val = get_clip_bound(safety, fieldName, defaultVal)
	if isfield(safety, fieldName) && ~isempty(safety.(fieldName))
		val = safety.(fieldName);
	else
		val = defaultVal;
	end
end
