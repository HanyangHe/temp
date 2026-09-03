function theta0Mat = make_scale_aware_seed_points(Coef_template, trainMask, Xtr, Ytr, arch, normOpt, opts)
%MAKE_SCALE_AWARE_SEED_POINTS Mask-aware feature-scale initialization.
%
% This function generates coefficient-vector seed points for GA/random-LSQ.
% It only initializes coefficients selected by trainMask; all inactive
% coefficients remain zero.  For every active dictionary row, the random
% coefficient scale is inversely proportional to the feature standard
% deviation on the training set.
%
% Basic idea for a block row:
%   h_r = sum_j c_j phi_j
%   std(c_j phi_j) should be O(targetStd/sqrt(m))
% hence
%   c_j ~ N(0, alpha^2 targetStd^2/(m std(phi_j)^2)).
%
% The generated Coef is packed by trainMask and returned as theta rows.
% The public seed-count field is opts.numCandidates.  This is a candidate
% count, not the number of LSQ starts.

	if nargin < 8 || isempty(opts)
		opts = struct();
	end
	if nargin < 7 || isempty(normOpt)
		normOpt = default_norm_options();
	end

	nVars = count_active_mask(trainMask);
	if nVars == 0
		theta0Mat = zeros(0, 0);
		return;
	end

	seed = getfield_default_local(opts, 'rngSeed', getfield_default_local(opts, 'seed', 1));
	rng(seed + 7919);

	nSeeds = getfield_default_local(opts, 'numCandidates', getfield_default_local(opts, 'numInitialSeeds', 40));
	nSeeds = max(0, round(nSeeds));
	if nSeeds == 0
		theta0Mat = zeros(0, nVars);
		return;
	end

	alphaList = getfield_default_local(opts, 'scaleAwareAlphaList', []);
	if isempty(alphaList)
		alphaList = getfield_default_local(opts, 'scaleAwareAlpha', 0.5);
	end
	alphaList = alphaList(:).';
	if isempty(alphaList)
		alphaList = 0.5;
	end

	theta0Mat = zeros(nSeeds, nVars);

	for s = 1:nSeeds
		alpha = alphaList(1 + mod(s - 1, numel(alphaList)));
		Coef0 = make_one_scale_aware_coef_local(Coef_template, trainMask, ...
			Xtr, Ytr, arch, normOpt, opts, alpha, s);
		theta0Mat(s, :) = pack_Coef_M_by_mask(Coef0, trainMask).';
	end

	finiteRows = all(isfinite(theta0Mat), 2);
	if ~all(finiteRows)
		theta0Mat = theta0Mat(finiteRows, :);
	end

	% Keep the requested count cleanly.  Do not append generic seeds here;
	% masked_lsq_initialize selects either scale-aware seeds or generic box seeds.
end

function Coef0 = make_one_scale_aware_coef_local(Coef_template, trainMask, Xtr, Ytr, arch, normOpt, opts, alpha, seedIndex)
	Coef0 = Coef_template;
	dims = get_arch_dims(arch);

	Nsp = size(Xtr, 1);
	h = cell(1, arch.layer + 1);
	h{1} = Xtr.';
	h{1}(~isfinite(h{1})) = 0;

	yStd = std(Ytr, 0, 1).';
	yStd(~isfinite(yStd) | yStd < getfield_default_local(opts, 'scaleAwareTargetStdFloor', 1e-8)) = 1;

	for ell = 1:arch.layer
		rowDim = dims(ell + 1);

		% Build branch caches using the already-initialized prefix states.
		branches = cell(1, ell);
		for src = 1:ell
			k = ell - src + 1;
			if isempty(h{k})
				% This should not happen for valid prefix propagation, but keep a
				% safe zero fallback to avoid crashing seed generation.
				inputDim = dims(k);
				h{k} = zeros(inputDim, Nsp);
			end
			branches{src} = build_branch_cache(h{k}, arch, ell, h, src);
		end

		targetStd = make_layer_target_std_local(rowDim, yStd, ell, arch, opts);

		% Count active terms per output row over all source blocks in this layer.
		mRow = zeros(rowDim, 1);
		for src = 1:ell
			M = trainMask{src, ell};
			if isempty(M)
				continue;
			end
			M = logical(M);
			M = remove_invalid_rows_from_mask_local(M, branches{src}, opts);
			mRow = mRow + sum(M, 2);
		end
		mRow(mRow < 1) = 1;

		% Initialize every active coefficient with feature-scale-aware std.
		for src = 1:ell
			M = trainMask{src, ell};
			if isempty(M)
				continue;
			end

			M = logical(M);
			branch = branches{src};
			M = remove_invalid_rows_from_mask_local(M, branch, opts);

			W = zeros(size(Coef0{src, ell}));

			for r = 1:rowDim
				idx = find(M(r, :));
				if isempty(idx)
					continue;
				end

				Zr = branch.Phi(idx, :);
				phiStd = std(Zr, 0, 2);
				phiStd = make_safe_feature_std_local(phiStd, opts);

				coefStd = alpha * targetStd(r) ./ (sqrt(mRow(r)) * phiStd);

				% Random sign and magnitude; this is scale-aware, not LS-fitted.
				W(r, idx) = coefStd(:).' .* randn(1, numel(idx));
			end

			Coef0{src, ell} = W;
		end

		% Row-wise target-scale calibration for the current layer.
		[Coef0, hNext] = calibrate_current_layer_local(Coef0, trainMask, ...
			branches, targetStd, ell, arch, opts);

		h{ell + 1} = hNext;
	end

	% Optional small global jitter makes multiple seeds less correlated after
	% deterministic calibration.
	jitter = getfield_default_local(opts, 'scaleAwarePostJitter', 0);
	if jitter > 0
		for ell = 1:arch.layer
			for src = 1:ell
				M = trainMask{src, ell};
				if isempty(M)
					continue;
				end
				J = 1 + jitter * randn(size(Coef0{src, ell}));
				Coef0{src, ell}(logical(M)) = Coef0{src, ell}(logical(M)) .* J(logical(M));
			end
		end
	end

	% Make sure inactive coefficients are exactly zero.
	for ell = 1:arch.layer
		for src = 1:ell
			M = trainMask{src, ell};
			if isempty(M)
				Coef0{src, ell}(:) = 0;
			else
				Coef0{src, ell}(~logical(M)) = 0;
			end
		end
	end
end

function M = remove_invalid_rows_from_mask_local(M, branch, opts)
	if getfield_default_local(opts, 'scaleAwareRemoveInvalidRows', true)
		if isfield(branch, 'PhiInvalidRows') && ~isempty(branch.PhiInvalidRows)
			bad = logical(branch.PhiInvalidRows(:)).';
			if numel(bad) == size(M, 2)
				M(:, bad) = false;
			end
		end
	end
end

function phiStd = make_safe_feature_std_local(phiStd, opts)
	floorVal = getfield_default_local(opts, 'scaleAwareFeatureStdFloor', 1e-8);
	constFactor = getfield_default_local(opts, 'scaleAwareConstantStdFactor', 1.0);

	bad = ~isfinite(phiStd) | phiStd < floorVal;

	% For constant or nearly constant features, do not divide by a tiny std.
	% Treat them as O(1) features, with an optional conservative factor.
	phiStd(bad) = max(floorVal, constFactor);
	phiStd(~isfinite(phiStd) | phiStd < floorVal) = floorVal;
end

function targetStd = make_layer_target_std_local(rowDim, yStd, ell, arch, opts)
	if ell == arch.layer
		mode = getfield_default_local(opts, 'scaleAwareOutputTargetMode', 'output_std');
	else
		mode = getfield_default_local(opts, 'scaleAwareHiddenTargetMode', 'clipped_output_std');
	end
	mode = lower(strtrim(char(mode)));

	switch mode
		case {'unit', 'one'}
			targetStd = ones(rowDim, 1);

		case {'output_std', 'y_std'}
			targetStd = repeat_to_rowdim_local(yStd, rowDim);

		case {'clipped_output_std', 'clipped-y-std'}
			targetStd = repeat_to_rowdim_local(yStd, rowDim);
			capVal = getfield_default_local(opts, 'scaleAwareHiddenTargetStdCap', 1.0);
			targetStd = min(targetStd, capVal);

		case {'fixed', 'constant'}
			val = getfield_default_local(opts, 'scaleAwareFixedTargetStd', 1.0);
			targetStd = val * ones(rowDim, 1);

		otherwise
			error('Unknown scale-aware target mode: %s', mode);
	end

	floorVal = getfield_default_local(opts, 'scaleAwareTargetStdFloor', 1e-6);
	targetStd(~isfinite(targetStd) | targetStd < floorVal) = floorVal;
end

function v = repeat_to_rowdim_local(vIn, rowDim)
	vIn = vIn(:);
	if numel(vIn) == rowDim
		v = vIn;
	elseif numel(vIn) == 1
		v = vIn * ones(rowDim, 1);
	else
		v = vIn(1 + mod((0:rowDim-1).', numel(vIn)));
	end
end

function [Coef0, hNext] = calibrate_current_layer_local(Coef0, trainMask, branches, targetStd, ell, arch, opts)
	rowDim = numel(targetStd);
	Nsp = size(branches{1}.Phi, 2);
	tmpRaw = zeros(rowDim, Nsp);

	for src = 1:ell
		tmpRaw = tmpRaw + Coef0{src, ell} * branches{src}.Phi;
	end

	doCalibrate = getfield_default_local(opts, 'scaleAwareCalibrateLayerOutput', true);
	if doCalibrate
		hStd = std(tmpRaw, 0, 2);
		alphaCal = getfield_default_local(opts, 'scaleAwareCalibrationFactor', ...
			getfield_default_local(opts, 'scaleAwareAlpha', 0.5));
		desiredStd = alphaCal * targetStd;

		scaleMin = getfield_default_local(opts, 'scaleAwareLayerScaleMin', 0.1);
		scaleMax = getfield_default_local(opts, 'scaleAwareLayerScaleMax', 10);
		stdFloor = getfield_default_local(opts, 'scaleAwareTargetStdFloor', 1e-8);

		rowScale = desiredStd ./ max(hStd, stdFloor);
		rowScale(~isfinite(rowScale)) = 1;
		rowScale = min(max(rowScale, scaleMin), scaleMax);

		for src = 1:ell
			M = trainMask{src, ell};
			if isempty(M)
				continue;
			end
			for r = 1:rowDim
				Coef0{src, ell}(r, :) = rowScale(r) * Coef0{src, ell}(r, :);
			end
			Coef0{src, ell}(~logical(M)) = 0;
		end

		tmpRaw = zeros(rowDim, Nsp);
		for src = 1:ell
			tmpRaw = tmpRaw + Coef0{src, ell} * branches{src}.Phi;
		end
	end

	tmpRaw(~isfinite(tmpRaw)) = 0;

	if ell == arch.layer
		clipBound = get_clip_bound_local(arch.safety, 'finalOutputClip', Inf);
	else
		clipBound = get_clip_bound_local(arch.safety, 'hiddenLayerOutputClip', ...
			get_clip_bound_local(arch.safety, 'layerOutputClip', Inf));
	end

	if isinf(clipBound)
		hNext = tmpRaw;
	else
		hNext = min(max(tmpRaw, -clipBound), clipBound);
	end
	hNext(~isfinite(hNext)) = 0;
end

function val = get_clip_bound_local(safety, fieldName, defaultVal)
	if isfield(safety, fieldName) && ~isempty(safety.(fieldName))
		val = safety.(fieldName);
	else
		val = defaultVal;
	end
end

function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
