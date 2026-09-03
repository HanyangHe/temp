function [J, gradCoef] = obj_phdn_core_withgrad(Coef_M, X, Y, arch, lambda1, lambda2, epsSmoothL1, normOpt)
%OBJ_PHDN_CORE_WITHGRAD Shared objective/backprop core.
%
% Updated with active-operator feasibility penalty.
%
% Loss:
%   mean((Ypred - Y).^2)
% Regularization:
%   lambda1 * sum(sqrt(theta.^2 + epsSmoothL1))
%   lambda2 * sum(theta.^2)
% Operator feasibility penalty:
%   Penalizes active use of operator arguments that are illegal in ordinary
%   symbolic math, e.g., inv(0), log(0), log(negative), sqrt(negative),
%   invsqrt(0), asin/acos outside [-1,1], tan near singularity, or overly
%   large exp arguments. The penalty is weighted by the coefficients that
%   use the offending operator features, so unused library entries are not
%   penalized.

	if nargin < 9 || isempty(normOpt)
		normOpt = default_norm_options();
	end

	[Ypred, cache] = model_forward(X, Coef_M, arch, normOpt);

	err = Ypred - Y;
	Nall = numel(err);
	J_fit = mean(err(:).^2);

	theta = pack_Coef_M_full(Coef_M);
	J_sparse = lambda1 * sum(sqrt(theta.^2 + epsSmoothL1));
	J_ridge = lambda2 * sum(theta.^2);
	J = J_fit + J_sparse + J_ridge;

	L = arch.layer;
	dims = cache.dims;
	Nsp = size(X, 1);

	gradCoef = cell(size(Coef_M));
	gradCoefFeas = cell(size(Coef_M));
	for ell = 1:L
		for src = 1:ell
			gradCoef{src, ell} = zeros(size(Coef_M{src, ell}));
			gradCoefFeas{src, ell} = zeros(size(Coef_M{src, ell}));
		end
	end

	Delta = cell(1, L + 1);
	for k = 1:(L + 1)
		Delta{k} = zeros(dims(k), Nsp);
	end

	Delta{L + 1} = (2 / Nall) * err.';

	% Add active operator-domain feasibility penalty directly into the
	% optimization objective. This pushes the optimizer away from x/0, log(0),
	% log(negative), and related invalid symbolic paths instead of only
	% rejecting them after training.
	[J_feas, gradCoefFeas, DeltaFeas] = operator_feasibility_penalty_withgrad_local( ...
		Coef_M, cache, arch, size(Y, 1));
	J = J + J_feas;

	for k = 1:min(numel(Delta), numel(DeltaFeas))
		if ~isempty(DeltaFeas{k})
			Delta{k} = Delta{k} + DeltaFeas{k};
		end
	end

	for ell = L:-1:1
		deltaRaw = Delta{ell + 1} .* cache.tmpClipMask{ell};

		for src = 1:ell
			k = ell - src + 1;
			branch = cache.branch{src, ell};
			A = Coef_M{src, ell};
			Z = branch.Phi;

			gradA = deltaRaw * Z.';

			if lambda1 ~= 0
				gradA = gradA + lambda1 * A ./ sqrt(A.^2 + epsSmoothL1);
			end
			if lambda2 ~= 0
				gradA = gradA + 2 * lambda2 * A;
			end

			gradCoef{src, ell} = gradA + gradCoefFeas{src, ell};

			[dH, dContext] = backprop_branch_local(branch, A, deltaRaw, arch);
			Delta{k} = Delta{k} + dH;
			for kk = 1:min(numel(Delta), numel(dContext))
				if ~isempty(dContext{kk})
					Delta{kk} = Delta{kk} + dContext{kk};
				end
			end
		end
	end
end

function [J_feas, gradCoefFeas, DeltaFeas] = operator_feasibility_penalty_withgrad_local(Coef_M, cache, arch, nSamples)
%OPERATOR_FEASIBILITY_PENALTY_WITHGRAD_LOCAL Active operator-domain penalty.

	L = arch.layer;
	dims = cache.dims;

	J_feas = 0;
	gradCoefFeas = cell(size(Coef_M));
	for ell = 1:L
		for src = 1:ell
			gradCoefFeas{src, ell} = zeros(size(Coef_M{src, ell}));
		end
	end

	DeltaFeas = cell(1, L + 1);
	for k = 1:(L + 1)
		DeltaFeas{k} = zeros(dims(k), nSamples);
	end

	if ~isfield(arch, 'feasibility') || isempty(arch.feasibility)
		return;
	end

	feas = arch.feasibility;
	if ~get_bool_field_local(feas, 'useOperatorFeasibilityPenalty', false)
		return;
	end

	weight = get_field_local(feas, 'operatorFeasibilityPenaltyWeight', 1e-2);
	if weight <= 0
		return;
	end

	for ell = 1:L
		for src = 1:ell
			branch = cache.branch{src, ell};
			A = Coef_M{src, ell};

			[Jb, gradA_b, dH_b] = branch_operator_feasibility_penalty_local(branch, A, feas, arch.safety);

			if Jb ~= 0
				J_feas = J_feas + weight * Jb;
				gradCoefFeas{src, ell} = gradCoefFeas{src, ell} + weight * gradA_b;

				k = ell - src + 1;
				DeltaFeas{k} = DeltaFeas{k} + weight * dH_b;
			end
		end
	end
end

function [Jb, gradA, dH] = branch_operator_feasibility_penalty_local(branch, A, feas, safety)
%BRANCH_OPERATOR_FEASIBILITY_PENALTY_LOCAL Penalty for one branch.

	Jb = 0;
	gradA = zeros(size(A));
	dH = zeros(size(branch.H));

	if isempty(branch.Q) || isempty(branch.G) || isempty(branch.cfg.opNames)
		return;
	end

	Q = branch.Q;
	Jq = branch.Jq;
	nQ = size(Q, 1);
	Nsp = size(Q, 2);
	nOp = numel(branch.cfg.opNames);
	gQ = zeros(size(Q));

	zeroMargin = get_field_local(feas, 'operatorZeroMargin', 1e-4);
	posMargin = get_field_local(feas, 'operatorPositiveMargin', 1e-4);
	domainMargin = get_field_local(feas, 'operatorDomainMargin', 1e-6);
	expLimit = get_field_local(feas, 'operatorExpSoftLimit', min(get_field_local(safety, 'expClip', 300), 50));
	tanCosMargin = get_field_local(feas, 'operatorTanCosMargin', 1e-3);
	usageEps = get_field_local(feas, 'operatorUsageEps', 1e-10);

	if expLimit <= 0
		expLimit = 50;
	end

	for op = 1:nOp
		opName = lower(strtrim(char(branch.cfg.opNames{op})));
		idxG = (op - 1) * nQ + (1:nQ);

		for qIdx = 1:nQ
			gRow = idxG(qIdx);
			[phiRows, dArows] = feature_rows_using_g_local(branch, gRow);

			if isempty(phiRows)
				continue;
			end

			usage = 0;
			usageGrad = cell(numel(phiRows), 1);
			for r = 1:numel(phiRows)
				phiRow = phiRows(r);
				Avec = A(:, phiRow);
				den = sqrt(Avec.^2 + usageEps);
				usage = usage + sum(den);
				usageGrad{r} = Avec ./ den;
			end

			if usage <= 0
				continue;
			end

			q = Q(qIdx, :);
			[pMean, dpdq] = invalid_argument_penalty_1d_local(q, opName, ...
				zeroMargin, posMargin, domainMargin, expLimit, tanCosMargin);

			if pMean == 0
				continue;
			end

			Jb = Jb + usage * pMean;
			gQ(qIdx, :) = gQ(qIdx, :) + usage * dpdq;

			for r = 1:numel(phiRows)
				phiRow = phiRows(r);
				gradA(:, phiRow) = gradA(:, phiRow) + pMean * usageGrad{r};
			end
		end
	end

	for s = 1:Nsp
		Jq_s = Jq(:, :, s);
		dH(:, s) = dH(:, s) + Jq_s.' * gQ(:, s);
	end

	dH(~isfinite(dH)) = 0;
	gradA(~isfinite(gradA)) = 0;
	if ~isfinite(Jb)
		Jb = 1e20;
	end
end

function [pMean, dpdq] = invalid_argument_penalty_1d_local(q, opName, zeroMargin, posMargin, domainMargin, expLimit, tanCosMargin)
%INVALID_ARGUMENT_PENALTY_1D_LOCAL Normalized hinge-square penalty.

	q = reshape(q, 1, []);
	N = max(1, numel(q));
	dpdq = zeros(size(q));
	p = zeros(size(q));

	finiteMask = isfinite(q);
	qSafe = q;
	qSafe(~finiteMask) = 0;

	switch opName
		case {'inv', 'inverse'}
			m = max(zeroMargin, eps);
			viol = max(0, m - abs(qSafe));
			active = viol > 0;
			p(active) = (viol(active) ./ m).^2;
			dpdq(active) = -(2 / N) .* viol(active) ./ (m.^2) .* sign(qSafe(active));

		case {'log', 'sqrt', 'invsqrt'}
			m = max(posMargin, eps);
			viol = max(0, m - qSafe);
			active = viol > 0;
			p(active) = (viol(active) ./ m).^2;
			dpdq(active) = -(2 / N) .* viol(active) ./ (m.^2);

		case {'asin', 'arcsin', 'acos', 'arccos'}
			upper = max(0, 1 - domainMargin);
			m = max(domainMargin, eps);
			viol = max(0, abs(qSafe) - upper);
			active = viol > 0;
			p(active) = (viol(active) ./ m).^2;
			dpdq(active) = (2 / N) .* viol(active) ./ (m.^2) .* sign(qSafe(active));

		case 'tan'
			m = max(tanCosMargin, eps);
			c = cos(qSafe);
			absC = abs(c);
			viol = max(0, m - absC);
			active = viol > 0;
			p(active) = (viol(active) ./ m).^2;
			% d(m-|cos(q)|)/dq = sin(q)*sign(cos(q))
			dpdq(active) = (2 / N) .* viol(active) ./ (m.^2) .* sin(qSafe(active)) .* sign(c(active));

		case {'exp', 'sinh', 'cosh'}
			m = max(expLimit, eps);
			viol = max(0, abs(qSafe) - m);
			active = viol > 0;
			p(active) = (viol(active) ./ m).^2;
			dpdq(active) = (2 / N) .* viol(active) ./ (m.^2) .* sign(qSafe(active));

		otherwise
			% sin, cos, atan, tanh, abs, cbrt, identity are globally defined
			% or already handled safely enough for symbolic feasibility.
	end

	if any(~finiteMask)
		p(~finiteMask) = p(~finiteMask) + 1e6;
	end

	pMean = mean(p);
	dpdq(~finiteMask) = 0;
	dpdq(~isfinite(dpdq)) = 0;
	if ~isfinite(pMean)
		pMean = 1e20;
	end
end

function [phiRows, localIndex] = feature_rows_using_g_local(branch, gRow)
%FEATURE_ROWS_USING_G_LOCAL Return Phi rows that use a G row.

	phiRows = [];
	localIndex = [];

	if isfield(branch.idx, 'baseG') && ~isempty(branch.idx.baseG)
		if gRow <= numel(branch.idx.baseG)
			phiRows(end + 1, 1) = branch.idx.baseG(gRow); %#ok<AGROW>
			localIndex(end + 1, 1) = gRow; %#ok<AGROW>
		end
	end

end

function val = get_field_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end

function tf = get_bool_field_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		tf = logical(s.(name));
	else
		tf = logical(defaultVal);
	end
end
