function report = check_masked_phdn_jacobian(theta, mask, coefZero, X, Y, arch, normOpt, cfgObjective, scaleY, opts)
%CHECK_MASKED_PHDN_JACOBIAN Finite-difference diagnostic for dictionary-PhDN LSQ Jacobian.
%
% This diagnostic is intended for the exact dictionary-PhDN route, not for the
% Stage-I branch-MLP surrogate. It compares the analytic Jacobian returned by
% residual_jacobian_masked_phdn against central finite differences of the same
% residual function under the same active-mask pack/unpack convention.

	if nargin < 10 || isempty(opts)
		opts = struct();
	end
	if nargin < 9
		scaleY = [];
	end
	if nargin < 8 || isempty(cfgObjective)
		cfgObjective = struct();
	end

	theta = theta(:);
	nVars = numel(theta);
	report = struct();
	report.checked = false;
	report.ok = true;
	report.reason = '';
	report.nVars = nVars;
	report.columns = [];
	report.maxAbsError = NaN;
	report.maxRelError = NaN;
	report.medianRelError = NaN;
	report.badColumnCount = 0;
	report.details = struct([]);

	if nVars == 0
		report.reason = 'empty theta';
		return;
	end

	nSample = max(1, round(getfield_default_local(opts, 'numSamples', min(40, size(X,1)))));
	nSample = min(nSample, size(X, 1));
	Xc = X(1:nSample, :);
	Yc = Y(1:nSample, :);

	nCols = max(1, round(getfield_default_local(opts, 'numColumns', min(12, nVars))));
	nCols = min(nCols, nVars);
	mode = lower(strtrim(char(getfield_default_local(opts, 'columnMode', 'spread'))));
	switch mode
		case {'first','head'}
			cols = 1:nCols;
		otherwise
			cols = unique(round(linspace(1, nVars, nCols)));
	end

	epsFD = getfield_default_local(opts, 'epsilon', 1e-6);
	relTol = getfield_default_local(opts, 'relTolerance', 1e-3);
	absTol = getfield_default_local(opts, 'absTolerance', 1e-6);

	cfgObjectiveCheck = cfgObjective;
	cfgObjectiveCheck.rethrowOnError = true;
	cfgObjectiveCheck.verboseError = false;

	try
		[r0, J] = residual_jacobian_masked_phdn(theta, mask, coefZero, Xc, Yc, arch, normOpt, cfgObjectiveCheck, scaleY);
	catch ME
		report.ok = false;
		report.reason = ['analytic residual/Jacobian threw error: ' ME.message];
		return;
	end
	if isempty(J) || any(~isfinite(J(:))) || size(J, 2) ~= nVars
		report.ok = false;
		report.reason = 'analytic Jacobian is empty, nonfinite, or has wrong column count';
		return;
	end
	if any(~isfinite(r0(:)))
		report.ok = false;
		report.reason = 'base residual is nonfinite';
		return;
	end

	absErr = zeros(numel(cols), 1);
	relErr = zeros(numel(cols), 1);
	details = repmat(struct('paramIndex', [], 'absError', [], 'relError', [], 'analyticNorm', [], 'finiteDiffNorm', []), numel(cols), 1);
	for ii = 1:numel(cols)
		p = cols(ii);
		step = epsFD * max(1, abs(theta(p)));
		tp = theta; tm = theta;
		tp(p) = tp(p) + step;
		tm(p) = tm(p) - step;
		rp = residual_jacobian_masked_phdn(tp, mask, coefZero, Xc, Yc, arch, normOpt, cfgObjectiveCheck, scaleY);
		rm = residual_jacobian_masked_phdn(tm, mask, coefZero, Xc, Yc, arch, normOpt, cfgObjectiveCheck, scaleY);
		jfd = (rp(:) - rm(:)) ./ (2 * step);
		ja = J(:, p);
		m = min(numel(jfd), numel(ja));
		jfd = jfd(1:m);
		ja = ja(1:m);
		ae = norm(ja - jfd, Inf);
		den = max([1, norm(ja, Inf), norm(jfd, Inf)]);
		re = ae / den;
		absErr(ii) = ae;
		relErr(ii) = re;
		details(ii).paramIndex = p;
		details(ii).absError = ae;
		details(ii).relError = re;
		details(ii).analyticNorm = norm(ja, Inf);
		details(ii).finiteDiffNorm = norm(jfd, Inf);
	end

	report.checked = true;
	report.columns = cols(:).';
	report.maxAbsError = max(absErr);
	report.maxRelError = max(relErr);
	report.medianRelError = median(relErr);
	report.badColumnCount = nnz(absErr > absTol & relErr > relTol);
	report.details = details;
	report.ok = report.badColumnCount == 0;
	if report.ok
		report.reason = 'analytic Jacobian matches finite-difference check';
	else
		report.reason = 'analytic Jacobian mismatch against finite differences';
	end
end

function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
