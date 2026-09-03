function [r, J] = residual_jacobian_masked_phdn(theta, mask, coefZero, X, Y, arch, normOpt, cfgObjective, scaleY)
%RESIDUAL_JACOBIAN_MASKED_PHDN Residual and analytic Jacobian for masked PhDN LSQ.
%
% This function is the previous-version analytic BP/Jacobian implementation for the
% cross-only, flexible-width PhDN model. It supports:
%   - layer-wise hidden widths;
%   - dense skip branches h^(k) -> h^(ell+1);
%   - within-branch dictionary cross terms;
%   - masked coefficient packing;
%   - no hidden-hidden Phi_mut terms;
%   - no PFLI/profied-readout regularization term.
%
% Residual layout is exactly consistent with the old residual function:
%   r = vec((Ypred - Y) ./ scaleY), where Ypred and Y are N-by-ny.
%
% If two outputs are requested, J has size numel(Y)-by-numel(theta), with
% optional regularization rows appended to both r and J.

	if nargin < 9 || isempty(scaleY)
		scaleY = make_residual_scale_local(Y, cfgObjective);
	end
	if nargin < 8 || isempty(cfgObjective)
		cfgObjective = struct();
	end
	if nargin < 7 || isempty(normOpt)
		normOpt = default_norm_options();
	end

	theta = theta(:);
	nVars = numel(theta);
	baseLen = numel(Y);
	fullLen = residual_length_local(Y, theta, cfgObjective);
	invalidPenalty = getfield_default_local(cfgObjective, 'invalidPenalty', 1e6);

	try
		Coef = unpack_Coef_M_by_mask(theta, mask, coefZero);
		[Ypred, cache] = model_forward(X, Coef, arch, normOpt);

		if ~isreal(Ypred) || any(~isfinite(Ypred(:)))
			r = invalidPenalty * ones(fullLen, 1);
			if nargout > 1
				J = zeros(fullLen, nVars);
			end
			return;
		end

		err = (Ypred - Y) ./ scaleY;
		if ~isreal(err) || any(~isfinite(err(:)))
			r = invalidPenalty * ones(fullLen, 1);
			if nargout > 1
				J = zeros(fullLen, nVars);
			end
			return;
		end

		r = err(:);

		if nargout > 1
			J = compute_residual_jacobian_local(Coef, mask, cache, scaleY, nVars);
			if size(J, 1) ~= baseLen || size(J, 2) ~= nVars
				error('Internal Jacobian size mismatch: got [%s], expected [%d %d].', ...
					num2str(size(J)), baseLen, nVars);
			end
		end

		% Optional Tikhonov residual rows.
		if isfield(cfgObjective, 'lambda2') && cfgObjective.lambda2 > 0
			lam2s = sqrt(cfgObjective.lambda2);
			r = [r; lam2s * theta(:)];
			if nargout > 1
				J = [J; lam2s * eye(nVars)];
			end
		end

		% Optional smooth-L1 residual rows.
		if isfield(cfgObjective, 'lambda1') && cfgObjective.lambda1 > 0
			epsSmoothL1 = getfield_default_local(cfgObjective, 'epsSmoothL1', 1e-8);
			lam1s = sqrt(cfgObjective.lambda1);
			q = theta(:).^2 + epsSmoothL1;
			rL1 = lam1s * q.^(1/4);
			r = [r; rL1];
			if nargout > 1
				dr = lam1s * theta(:) ./ (2 * q.^(3/4));
				J = [J; diag(dr)];
			end
		end

		if any(~isfinite(r)) || ~isreal(r)
			r = invalidPenalty * ones(fullLen, 1);
			if nargout > 1
				J = zeros(fullLen, nVars);
			end
		end
		if nargout > 1
			J(~isfinite(J)) = 0;
		end

	catch ME
		if isfield(cfgObjective, 'rethrowOnError') && logical(cfgObjective.rethrowOnError)
			rethrow(ME);
		end
		if isfield(cfgObjective, 'verboseError') && logical(cfgObjective.verboseError)
			warning('residual_jacobian_masked_phdn failed and returned invalid penalty: %s', ME.message);
		end
		r = invalidPenalty * ones(fullLen, 1);
		if nargout > 1
			J = zeros(fullLen, nVars);
		end
	end
end

function J = compute_residual_jacobian_local(Coef, mask, cache, scaleY, nVars)
	L = size(mask, 2);
	dims = cache.dims;
	N = size(cache.h{1}, 2);
	ny = dims(L + 1);

	[paramCell, paramRows, paramCols] = build_param_index_local(mask, nVars);

	% dH{k} has size dims(k)-by-N-by-nVars and stores d h^(k)/d theta.
	dH = cell(1, L + 1);
	for k = 1:(L + 1)
		dH{k} = zeros(dims(k), N, nVars);
	end

	for ell = 1:L
		rowDim = dims(ell + 1);
		dRaw = zeros(rowDim, N, nVars);

		for src = 1:ell
			k = ell - src + 1;
			branch = cache.branch{src, ell};
			A = Coef{src, ell};

			% Indirect part through the source state h^(k).
			if nVars > 0 && ~isempty(A) && any(dH{k}(:) ~= 0)
				Jphi = branch_phi_jacobian_local(branch);
				for s = 1:N
					Dsrc = reshape(dH{k}(:, s, :), dims(k), nVars);
					if any(Dsrc(:) ~= 0)
						B = A * Jphi(:, :, s);
						dRaw(:, s, :) = dRaw(:, s, :) + reshape(B * Dsrc, rowDim, 1, nVars);
					end
				end
			end

			% Direct part for coefficients in this branch.
			idx = paramCell{src, ell};
			if ~isempty(idx)
				rows = paramRows{src, ell};
				cols = paramCols{src, ell};
				Phi = branch.Phi;
				for a = 1:numel(idx)
					p = idx(a);
					dRaw(rows(a), :, p) = dRaw(rows(a), :, p) + Phi(cols(a), :);
				end
			end
		end

		% Chain derivative through output clipping.
		if isfield(cache, 'tmpClipMask') && numel(cache.tmpClipMask) >= ell && ~isempty(cache.tmpClipMask{ell})
			clipMask = cache.tmpClipMask{ell};
			for p = 1:nVars
				dRaw(:, :, p) = dRaw(:, :, p) .* clipMask;
			end
		end

		dH{ell + 1} = dRaw;
	end

	J = zeros(N * ny, nVars);
	outScale = ones(1, ny);
	if isfield(cache, 'outputDenormDerivative') && ~isempty(cache.outputDenormDerivative)
		outScale = cache.outputDenormDerivative;
	end
	for p = 1:nVars
		dY = dH{L + 1}(:, :, p).';  % N-by-ny in normalized output coordinates
		dY = dY .* outScale;        % chain through output denormalization
		dY = dY ./ scaleY;
		J(:, p) = dY(:);
	end
end

function Jphi = branch_phi_jacobian_local(branch)
%BRANCH_PHI_JACOBIAN_LOCAL Return d Phi / d H for one branch.
%
% Jphi has size nPhi-by-nVar-by-N and follows the same row order as
% branch.Phi in build_branch_cache.

	if isfield(branch, 'Jphi') && ~isempty(branch.Jphi)
		Jphi = branch.Jphi;
		Jphi(~isfinite(Jphi)) = 0;
		return;
	end

	nPhi = size(branch.Phi, 1);
	nVar = size(branch.H, 1);
	N = size(branch.H, 2);
	Jphi = zeros(nPhi, nVar, N);

	if isfield(branch.idx, 'baseU') && ~isempty(branch.idx.baseU)
		Jphi(branch.idx.baseU, :, :) = branch.Ju;
	end

	if isfield(branch.idx, 'baseG') && ~isempty(branch.idx.baseG)
		nQ = size(branch.Q, 1);
		nOp = numel(branch.cfg.opNames);
		for op = 1:nOp
			idxG = (op - 1) * nQ + (1:nQ);
			rows = branch.idx.baseG(idxG);
			for q = 1:nQ
				gRow = idxG(q);
				for s = 1:N
					Jphi(rows(q), :, s) = branch.dOp{op}(q, s) * branch.Jq(q, :, s);
				end
			end
		end
	end

	if isfield(branch, 'cross') && isfield(branch.cross, 'rows') && ~isempty(branch.cross.rows)
		leftIndex = branch.cross.leftIndex;
		rightIndex = branch.cross.rightIndex;
		crossRows = branch.cross.rows;
		nQ = size(branch.Q, 1);

		for r = 1:numel(crossRows)
			row = crossRows(r);
			a = leftIndex(r);
			b = rightIndex(r);
			op = branch.GopIndex(b);
			q = branch.GqIndex(b);

			for s = 1:N
				dLeft = reshape(branch.Jleft(a, :, s), 1, nVar);
				dG = branch.dOp{op}(q, s) * reshape(branch.Jq(q, :, s), 1, nVar);
				Jphi(row, :, s) = branch.G(b, s) * dLeft + branch.Uleft(a, s) * dG;
			end
		end
	end

	Jphi(~isfinite(Jphi)) = 0;
end

function [paramCell, paramRows, paramCols] = build_param_index_local(mask, nVars)
	L = size(mask, 2);
	paramCell = cell(size(mask));
	paramRows = cell(size(mask));
	paramCols = cell(size(mask));
	ptr = 1;

	for ell = 1:L
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M)
				paramCell{src, ell} = [];
				paramRows{src, ell} = [];
				paramCols{src, ell} = [];
				continue;
			end

			active = find(M);
			n = numel(active);
			[row, col] = ind2sub(size(M), active);
			paramCell{src, ell} = ptr:(ptr + n - 1);
			paramRows{src, ell} = row(:).';
			paramCols{src, ell} = col(:).';
			ptr = ptr + n;
		end
	end

	if ptr - 1 ~= nVars
		error('Parameter map mismatch: mask contains %d entries but nVars=%d.', ptr - 1, nVars);
	end
end

function scaleY = make_residual_scale_local(Y, cfgObjective)
	if ~isfield(cfgObjective, 'normalizeResidual') || ~cfgObjective.normalizeResidual
		scaleY = 1;
		return;
	end

	mode = 'std';
	if isfield(cfgObjective, 'residualScale') && ~isempty(cfgObjective.residualScale)
		mode = lower(string(cfgObjective.residualScale));
	end

	switch mode
		case "std"
			scaleY = std(Y, 0, 1);
		case "range"
			scaleY = max(Y, [], 1) - min(Y, [], 1);
		otherwise
			scaleY = ones(1, size(Y, 2));
	end

	bad = ~isfinite(scaleY) | scaleY < 1e-12;
	scaleY(bad) = 1;
end

function n = residual_length_local(Y, theta, cfgObjective)
	n = numel(Y);
	if isfield(cfgObjective, 'lambda2') && cfgObjective.lambda2 > 0
		n = n + numel(theta);
	end
	if isfield(cfgObjective, 'lambda1') && cfgObjective.lambda1 > 0
		n = n + numel(theta);
	end
end

function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
