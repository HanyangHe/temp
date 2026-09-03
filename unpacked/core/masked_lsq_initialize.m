function [theta_est, initStats, Coef_est, trainMask_est] = masked_lsq_initialize(Coef_template, trainMask, Xtr, Ytr, Xval, Yval, arch, normOpt, opts)
%MASKED_LSQ_INITIALIZE compact-dictionary route compact-mask GA/random/skip/direct BP-LSQ training.
%
% compact-mask route clean route:
%   1) Use the mask supplied by the case-specific compact dictionary support.
%   2) If arch.layer == 1, bypass BP and solve the linear Xi problem by
%      sequential thresholded least squares (STLS/SINDy fast path). Otherwise,
%      generate scale-aware/random/zero seed candidates, one direct BP-LSQ
%      start, or a lightweight multi-start LSQ/BP screening set.
%   3) Optionally run Beam Search Pruning with BP-LSQ fast fitting when mode='bsp_lsq'.
%   4) Optionally run coefficient GA when mode='ga_lsq'.
%      If mode='skip', no Stage-I coefficient optimizer is used; the selected
%      mask is passed directly to the final BP-LSQ refinement from one seed.
%   5) Run nonlinear LSQ/BP refinement.
%   6) Optionally run post-BP coefficient-score pruning iterations:
%      coefficient-score prune -> fast BP refine. Default iteration count is 1.

	if nargin < 8 || isempty(normOpt)
		normOpt = default_norm_options();
	end

	useAdmissibleMask = isfield(opts.training, 'useAdmissibleMask') && logical(opts.training.useAdmissibleMask);
	cfg = normalize_init_cfg_local(opts, useAdmissibleMask);
	isSingleLayerSTLS = is_single_layer_stls_active_local(arch, cfg);

	nVarsOriginal = count_active_mask(trainMask);
	nVars = nVarsOriginal;
	if nVars == 0 && cfg.errorOnEmptyMask
		error('Masked-LSQ training mask has zero active coefficients.');
	end

	if cfg.ga.autoConfigureEffort && strcmpi(cfg.mode, 'ga_lsq')
		opts = configure_ga_effort(opts, nVars);
		cfg = normalize_init_cfg_local(opts, useAdmissibleMask);
		isSingleLayerSTLS = is_single_layer_stls_active_local(arch, cfg);
	end

	isLargeNoMask = nVars > cfg.maxGAParams && ~useAdmissibleMask;
	% Large-no-mask fallback is only a GA safety rule. Direct LSQ-BP and
	% multi-start LSQ-BP intentionally train large smooth dictionaries and must
	% not be silently converted to random_lsq.
	applyLargeNoMaskPolicy = isLargeNoMask && strcmpi(cfg.mode, 'ga_lsq');
	if applyLargeNoMaskPolicy && ~cfg.allowLargeNoMaskProblem
		policy = lower(strtrim(char(cfg.largeNoMaskPolicy)));
		switch policy
			case {'fallback_random_lsq','fallback-random-lsq','random_lsq','random-lsq'}
				cfg.mode = 'random_lsq';
				cfg.ga.enable = false;
				cfg.seed.numCandidates = min(max(cfg.seed.numCandidates, 1), cfg.noMaskSeedCandidates);
				cfg.lsq.numStarts = max(1, min(cfg.lsq.numStarts, cfg.seed.numCandidates));
				if cfg.verbose
					warning(['No-mask problem has %d trainable coefficients, which is too large for GA. ', ...
						'Automatically switching to random_lsq with %d seed candidates and %d LSQ start(s).'], ...
						nVars, cfg.seed.numCandidates, cfg.lsq.numStarts);
				end
			case {'error','stop'}
				error(['No-mask problem has %d parameters, which is too large for GA. ', ...
					'Use an admissible/compact mask, set opts.init.allowLargeNoMaskProblem=true, ', ...
					'or set opts.init.largeNoMaskPolicy=''fallback_random_lsq''.'], nVars);
			case {'allow_ga','allow-ga','ga'}
				% Continue with GA.
			otherwise
				error('Unknown opts.init.largeNoMaskPolicy: %s', policy);
		end
	end

	if cfg.verbose
		fprintf('\n========================================\n');
		fprintf('Running compact-mask route compact-mask initialization/refinement\n');
		fprintf('========================================\n');
		fprintf('  initialization mode             = %s\n', cfg.mode);
		fprintf('  active masked coefficients       = %d\n', nVars);
		if isSingleLayerSTLS
			fprintf('  single-layer STLS/SINDy path     = on, threshold %.3e, maxIter %d\n', ...
				cfg.singleLayerSTLS.threshold, cfg.singleLayerSTLS.maxIter);
		end
		fprintf('  use admissible mask              = %d\n', useAdmissibleMask);
		if isSingleLayerSTLS
			fprintf('  seed generation mode             = skipped by single-layer STLS/SINDy path\n');
			fprintf('  seed candidates                  = 0\n');
			fprintf('  actual LSQ starts                = 0\n');
			fprintf('  analytic LSQ Jacobian            = not used\n');
		else
			fprintf('  seed generation mode             = %s\n', cfg.seed.mode);
			fprintf('  seed candidates                  = %d\n', cfg.seed.numCandidates);
			fprintf('  actual LSQ starts                = %d\n', cfg.lsq.numStarts);
			fprintf('  analytic LSQ Jacobian            = %d\n', cfg.lsq.useAnalyticJacobian);
			fprintf('  random initialization bounds     = [%g, %g]\n', ...
				min(expand_bound_local(cfg.seed.lowerBound, max(nVars,1))), ...
				max(expand_bound_local(cfg.seed.upperBound, max(nVars,1))));
			fprintf('  optimization coefficient bounds  = [%g, %g]\n', ...
				min(expand_bound_local(cfg.bounds.lower, max(nVars,1))), ...
				max(expand_bound_local(cfg.bounds.upper, max(nVars,1))));
		end
		if isfield(cfg, 'bsp') && cfg.bsp.enable
			fprintf('  HBSP-LSQ pruning initialization  = on, branch beam %d, basis beam %d, parallel %d\n', ...
				cfg.bsp.branchBeamWidth, cfg.bsp.beamWidth, cfg.bsp.useParallel);
			fprintf('  branch BSP rounds/fast BP        = %s / %d/%d, min branches %d\n', ...
				format_count_for_log_local(cfg.bsp.branchMaxRounds), cfg.bsp.branchFastMaxIter, ...
				cfg.bsp.branchFastMaxFunEvals, cfg.bsp.branchMinActiveBranches);
			fprintf('  basis BSP rounds/fast BP         = %s / %d/%d, min block %d\n', ...
				format_count_for_log_local(cfg.bsp.maxRounds), cfg.bsp.fastMaxIter, ...
				cfg.bsp.fastMaxFunEvals, cfg.bsp.minTermsPerXiBlock);
			fprintf('  HBSP validation tolerance        = branch rel %.3e, basis rel %.3e, abs %.3e\n', ...
				cfg.bsp.branchRelImproveTol, cfg.bsp.relImproveTol, cfg.bsp.absImproveTol);
			if useAdmissibleMask && cfg.bsp.skipWhenAdmissibleMask
				fprintf('  HBSP-LSQ pruning initialization  = skipped for admissible/strong-prior mask\n');
			end
		end
		if strcmpi(cfg.mode, 'ga_lsq')
			fprintf('  GA effort                        = %s\n', cfg.ga.effort);
			fprintf('  GA pop/gen/stall                 = %d / %d / %d\n', ...
				cfg.ga.populationSize, cfg.ga.maxGenerations, cfg.ga.maxStallGenerations);
			fprintf('  GA rough eval budget             = %d\n', cfg.ga.populationSize * cfg.ga.maxGenerations);
			if ~isempty(cfg.ga.targetETPD), fprintf('  GA target ETPD                   = %.2f\n', cfg.ga.targetETPD); end
			if ~isempty(cfg.ga.effectiveETPD), fprintf('  GA effective ETPD                = %.2f\n', cfg.ga.effectiveETPD); end
			if ~isempty(cfg.ga.capLimited), fprintf('  GA cap limited                   = %d\n', cfg.ga.capLimited); end
		end
		if isSingleLayerSTLS
			fprintf('  direct multi-start screening     = skipped by single-layer STLS/SINDy path\n');
			fprintf('  final LSQ/BP refinement          = skipped by single-layer STLS/SINDy path\n');
			fprintf('  post-BP prune iterations         = skipped by single-layer STLS/SINDy path\n');
		else
			if strcmpi(cfg.mode, 'multistart_lsq_bp') || (strcmpi(cfg.mode, 'skip') && cfg.multiStart.enable)
				fprintf('  direct multi-start screening     = %d starts, fast BP %d/%d\n', ...
					cfg.multiStart.numStarts, cfg.multiStart.screenMaxIter, cfg.multiStart.screenMaxFunEvals);
			end
			if strcmpi(cfg.mode, 'skip')
				if cfg.skip.hasExternalSeed
					fprintf('  Stage-2 initialization           = skipped; using Stage-1 medium-BP coefficient seed when valid\n');
				elseif cfg.multiStart.enable
					fprintf('  Stage-I initialization           = skipped; fixed-mask light BP screening provides final BP seed\n');
				else
					fprintf('  Stage-I initialization           = skipped; direct final BP-LSQ uses one seed under the selected mask\n');
				end
			end
			if cfg.lsq.enable
				fprintf('  final LSQ/BP refinement          = maxIter %d, maxFunEvals %d, analyticJac %d\n', ...
					cfg.lsq.maxIter, cfg.lsq.maxFunEvals, cfg.lsq.useAnalyticJacobian);
			else
				fprintf('  final LSQ/BP refinement          = disabled by opts.init.lsq.enable=false\n');
			end
			if cfg.lsq.enable && cfg.lsq.debugJacobianCheck
				fprintf('  exact-PhDN Jacobian check        = on, cols %d, samples %d, fallbackFD %d\n', ...
					cfg.lsq.debugJacobianNumColumns, cfg.lsq.debugJacobianNumSamples, ...
					cfg.lsq.fallbackToFiniteDifferenceOnBadJacobian);
			end
			if cfg.lsq.enable && isfield(cfg.lsq, 'earlyStop') && cfg.lsq.earlyStop.enable
				fprintf('  final LSQ/BP early stop          = chunk %d, minChunks %d, patience %d, rel/abs %.3e / %.3e\n', ...
					cfg.lsq.earlyStop.chunkMaxIter, cfg.lsq.earlyStop.minChunks, cfg.lsq.earlyStop.valPatience, ...
					cfg.lsq.earlyStop.relImproveTol, cfg.lsq.earlyStop.absImproveTol);
			end
			if cfg.postBPPrune.enable
				fprintf('  post-BP prune iterations         = %d, score=%s, abs/rel %.3e / %.3e, fast BP %d/%d\n', ...
					cfg.postBPPrune.numIterations, cfg.postBPPrune.scoreMode, cfg.postBPPrune.absThreshold, ...
					cfg.postBPPrune.relThreshold, cfg.postBPPrune.refineMaxIter, cfg.postBPPrune.refineMaxFunEvals);
			end
		end
		if isLargeNoMask && strcmpi(cfg.mode, 'ga_lsq')
			fprintf('  large no-mask policy             = %s\n', cfg.largeNoMaskPolicy);
		elseif isLargeNoMask
			fprintf('  large no-mask policy             = not applied for mode %s\n', cfg.mode);
		end
	end

	if cfg.debugMaskSummary
		print_mask_summary_local(trainMask, 'Masked-LSQ trainable mask');
	end

	if isSingleLayerSTLS
		[theta_est, initStats, Coef_est, trainMask_est] = single_layer_stls_initialize_local( ...
			Coef_template, trainMask, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, nVarsOriginal);
		return;
	end

	lb = expand_bound_local(cfg.bounds.lower, nVars);
	ub = expand_bound_local(cfg.bounds.upper, nVars);
	if any(lb >= ub)
		error('Invalid optimization coefficient bounds: lower bound must be less than upper bound.');
	end

	% Random/scale-aware initialization bounds are independent of the hard
	% optimization bounds.  They default to the optimization bounds for full
	% backward compatibility with all existing demos and callers.
	seedLb = expand_bound_local(cfg.seed.lowerBound, nVars);
	seedUb = expand_bound_local(cfg.seed.upperBound, nVars);
	if any(seedLb >= seedUb)
		error('Invalid initialization bounds: lower bound must be less than upper bound.');
	end
	if any(seedLb < lb) || any(seedUb > ub)
		error(['Initialization bounds must lie inside the optimization coefficient bounds. ', ...
			'Got initialization [%g, %g] and optimization [%g, %g].'], ...
			min(seedLb), max(seedUb), min(lb), max(ub));
	end

	coefZero = Coef_template;

	% Seed candidate generation.
	if strcmpi(cfg.mode, 'lsq_only')
		theta0Mat = zeros(1, nVars);
	else
		seedCfg = cfg.seed;
		seedCfg.numCandidates = max(1, round(seedCfg.numCandidates));
		seedCfg.lowerBound = cfg.seed.lowerBound;
		seedCfg.upperBound = cfg.seed.upperBound;
		seedCfg.lambda1 = cfg.objective.lambda1;
		seedCfg.lambda2 = cfg.objective.lambda2;
		seedCfg.epsSmoothL1 = cfg.objective.epsSmoothL1;
		seedCfg.invalidPenalty = cfg.objective.invalidPenalty;

		if strcmpi(seedCfg.mode, 'scale_aware') && nVars > 0
			theta0Mat = make_scale_aware_seed_points(Coef_template, trainMask, Xtr, Ytr, arch, normOpt, seedCfg);
		elseif strcmpi(seedCfg.mode, 'zero')
			theta0Mat = zeros(seedCfg.numCandidates, nVars);
		else
			theta0Mat = make_ga_seed_points(nVars, seedLb, seedUb, seedCfg);
		end

		% Enforce the dedicated initialization box for every generated seed mode.
		% External coefficient seeds are handled separately below and are not
		% altered by this operation.
		if ~isempty(theta0Mat)
			theta0Mat = min(max(theta0Mat, seedLb(:).'), seedUb(:).');
		end
	end

	usedExternalTheta = false;
	externalThetaSource = '';
	if isfield(cfg.seed, 'externalTheta') && ~isempty(cfg.seed.externalTheta) && nVars > 0
		thetaExt = cfg.seed.externalTheta(:).';
		if numel(thetaExt) == nVars
			% When a previous screening phase explicitly asks to continue from a
			% selected coefficient vector, do not let a newly generated random seed
			% replace it through the seed-objective ranking.
			if getfield_default_local(cfg.seed, 'forceExternalThetaOnly', false)
				theta0Mat = thetaExt;
			elseif getfield_default_local(cfg.seed, 'prependExternalTheta', true)
				theta0Mat = [thetaExt; theta0Mat];
			else
				theta0Mat = [theta0Mat; thetaExt];
			end
			usedExternalTheta = true;
			externalThetaSource = getfield_default_local(cfg.seed, 'externalThetaSource', 'external_seed');
		elseif cfg.verbose
			warning('External coefficient seed ignored: length %d does not match active parameter count %d.', numel(thetaExt), nVars);
		end
	end

	if isempty(theta0Mat)
		theta0Mat = zeros(1, nVars);
	end
	if size(theta0Mat, 2) ~= nVars
		error('Seed matrix has %d columns, but nVars=%d.', size(theta0Mat, 2), nVars);
	end

	scaleY = make_residual_scale_local(Ytr, cfg.objective);
	objTrain = @(theta) scalar_objective_local(theta(:), trainMask, coefZero, Xtr, Ytr, arch, normOpt, cfg.objective, scaleY);
	[resTrain, resJacTrain, lsqResidualPaddingRows] = make_lsq_residual_handles_local( ...
		trainMask, coefZero, Xtr, Ytr, arch, normOpt, cfg.objective, scaleY);

	initStats = struct();
	initStats.method = 'masked_lsq_compact-mask route_hbsp_lsq';
	initStats.mode = cfg.mode;
	initStats.requestedMode = cfg.requestedMode;
	initStats.nVarsOriginal = nVarsOriginal;
	initStats.nVars = nVars;
	initStats.seedMode = cfg.seed.mode;
	initStats.initializationLowerBound = cfg.seed.lowerBound;
	initStats.initializationUpperBound = cfg.seed.upperBound;
	initStats.optimizationLowerBound = cfg.bounds.lower;
	initStats.optimizationUpperBound = cfg.bounds.upper;
	initStats.numSeedCandidatesRequested = cfg.seed.numCandidates;
	initStats.numSeedCandidatesActual = size(theta0Mat, 1);
	initStats.usedExternalThetaSeed = usedExternalTheta;
	initStats.externalThetaSource = externalThetaSource;
	initStats.numLSQStartsRequested = cfg.lsq.numStarts;
	initStats.lsqResidualPaddingRows = lsqResidualPaddingRows;
	if cfg.verbose && lsqResidualPaddingRows > 0
		fprintf(['  bounded LSQ residual padding    = %d exact-zero row(s); ', ...
			'objective unchanged, bounds preserved\n'], lsqResidualPaddingRows);
	end
	initStats.populationSize = cfg.ga.populationSize;
	initStats.maxGenerations = cfg.ga.maxGenerations;
	initStats.maxStallGenerations = cfg.ga.maxStallGenerations;
	initStats.gaExitFlag = NaN;
	initStats.gaOutput = [];
	initStats.gaBestObjective = NaN;
	initStats.seedBestObjective = NaN;
	initStats.randomBestObjective = NaN;
	initStats.lsqExitFlag = NaN;
	initStats.lsqOutput = [];
	initStats.gaTime = 0;
	initStats.usedGA = false;
	initStats.preLSQObjective = NaN;
	initStats.preLSQTrainMSE = NaN;
	initStats.preLSQValMSE = NaN;
	initStats.seedSearchTime = 0;
	initStats.randomSearchTime = 0;
	initStats.lsqTime = 0;
	initStats.lsqStartTable = struct([]);
	initStats.jacobianDiagnostics = {};   % cell array: reports may have diagnostic-dependent fields
	initStats.bsp = struct('applied', false);
	initStats.multiStart = struct('applied', false);
	initStats.postBPPrune = struct('applied', false);
	initStats.augmentationDiagnostics = struct('enabled',false);

	% Seed ranking.
	tSeed = tic;
	fVals = inf(size(theta0Mat, 1), 1);
	for k = 1:size(theta0Mat, 1)
		fVals(k) = objTrain(theta0Mat(k, :).');
	end
	[bestSeedObj, bestIdx] = min(fVals);
	[~, seedOrder] = sort(fVals, 'ascend');
	thetaSeedBest = theta0Mat(bestIdx, :).';
	initStats.seedSearchTime = toc(tSeed);
	initStats.randomSearchTime = initStats.seedSearchTime;
	initStats.seedBestObjective = bestSeedObj;
	initStats.randomBestObjective = bestSeedObj;
	initStats.bestSeedIndex = bestIdx;

	if cfg.verbose
		fprintf('  seed best objective              = %.6e\n', bestSeedObj);
		fprintf('  best seed index                  = %d / %d\n', bestIdx, size(theta0Mat, 1));
		if usedExternalTheta
			fprintf('  external coefficient seed        = used [%s]\n', externalThetaSource);
		end
	end

	% Direct multi-start LSQ/BP screening. This is only a basin-selection stage.
	% The selected parameter vector is then passed to the normal final
	% heavy LSQ/BP refinement below.
	multiStartStats = struct('applied', false);
	if (strcmpi(cfg.mode, 'multistart_lsq_bp') || (strcmpi(cfg.mode, 'skip') && cfg.multiStart.enable)) && nVars > 0
		multiStartStats = direct_multistart_lsqbp_screen_local(theta0Mat, seedOrder, trainMask, coefZero, ...
			Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, scaleY, lb, ub, resTrain, resJacTrain);
		if isfield(multiStartStats, 'applied') && multiStartStats.applied
			thetaSeedBest = multiStartStats.bestTheta(:);
			bestSeedObj = objTrain(thetaSeedBest);
			bestIdx = multiStartStats.bestStartIndex;
			theta0Mat = thetaSeedBest(:).';
			seedOrder = 1;
			initStats.seedBestObjective = bestSeedObj;
			initStats.randomBestObjective = bestSeedObj;
			initStats.bestSeedIndex = bestIdx;
			initStats.numSeedCandidatesActual = multiStartStats.numStarts;
			if cfg.verbose
				fprintf('  multi-start selected start       = %d / %d\n', ...
					multiStartStats.bestStartIndex, multiStartStats.numStarts);
				fprintf('  multi-start selected train/val   = %.6e / %.6e\n', ...
					multiStartStats.bestTrainMSE, multiStartStats.bestValMSE);
			end
		end
	end
	initStats.multiStart = multiStartStats;

	% compact-mask route Hierarchical BSP-LSQ fast fitting.
	% Branch-level BSP is applied first; dictionary-basis-level BSP then inherits
	% the selected branch structure. Candidate structures are evaluated by
	% lightweight LSQ/BP refinement and validation MSE.
	bspStats = struct('applied', false);
	useBSP = isfield(cfg, 'bsp') && cfg.bsp.enable && nVars > 0;
	if useBSP && useAdmissibleMask && cfg.bsp.skipWhenAdmissibleMask
		bspStats = struct('applied', false, 'reason', 'skipped_admissible_or_strong_prior_mask');
		if cfg.verbose && cfg.bsp.verbose
			fprintf('  HBSP-LSQ skipped because an admissible/strong-prior mask is active.\n');
		end
		useBSP = false;
	end
	if useBSP
		[thetaBSP, trainMaskBSP, CoefBSP, bspStats] = bsp_lsq_init_local( ...
			thetaSeedBest, trainMask, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg);
		if isfield(bspStats, 'applied') && bspStats.applied
			trainMask = trainMaskBSP;
			nVars = count_active_mask(trainMask);
			thetaSeedBest = thetaBSP(:);
			theta0Mat = thetaSeedBest(:).';
			lb = expand_bound_local(cfg.bounds.lower, nVars);
			ub = expand_bound_local(cfg.bounds.upper, nVars);
			coefZero = zero_Coef_like(CoefBSP);
			scaleY = make_residual_scale_local(Ytr, cfg.objective);
			objTrain = @(theta) scalar_objective_local(theta(:), trainMask, coefZero, Xtr, Ytr, arch, normOpt, cfg.objective, scaleY);
			[resTrain, resJacTrain, lsqResidualPaddingRows] = make_lsq_residual_handles_local( ...
				trainMask, coefZero, Xtr, Ytr, arch, normOpt, cfg.objective, scaleY);
			initStats.lsqResidualPaddingRows = lsqResidualPaddingRows;
			bestSeedObj = objTrain(thetaSeedBest);
			bestIdx = 1;
			seedOrder = 1;
			initStats.nVars = nVars;
			initStats.numSeedCandidatesActual = size(theta0Mat, 1);
			initStats.bsp = bspStats;
			if cfg.verbose
				fprintf('  HBSP-LSQ active                  = %d -> %d\n', bspStats.originalActive, bspStats.finalActive);
				fprintf('  post-BSP seed objective          = %.6e\n', bestSeedObj);
			end
		end
	end

	% Optional coefficient GA coarse search.
	thetaGA = thetaSeedBest;
	objGA = bestSeedObj;
	useGA = strcmpi(cfg.mode, 'ga_lsq') && cfg.ga.enable && exist('ga', 'file') == 2 && nVars > 0;
	if useGA
		tGA = tic;
		gaOpts = optimoptions('ga', ...
			'Display', cfg.ga.display, ...
			'PopulationSize', cfg.ga.populationSize, ...
			'MaxGenerations', cfg.ga.maxGenerations, ...
			'MaxStallGenerations', cfg.ga.maxStallGenerations, ...
			'FunctionTolerance', cfg.ga.functionTolerance, ...
			'UseParallel', cfg.ga.useParallel);
		try
			pop = theta0Mat;
			if size(pop, 1) > cfg.ga.populationSize
				pop = pop(1:cfg.ga.populationSize, :);
			end
			gaOpts = optimoptions(gaOpts, 'InitialPopulationMatrix', pop);
			[thetaGArow, objGA, exitflagGA, outputGA] = ga(@(thetaRow) objTrain(thetaRow(:)), ...
				nVars, [], [], [], [], lb, ub, [], gaOpts);
			thetaGA = thetaGArow(:);
			initStats.gaExitFlag = exitflagGA;
			initStats.gaOutput = outputGA;
		catch ME
			warning('GA failed; using the best seed point. Reason: %s', ME.message);
			thetaGA = thetaSeedBest;
			objGA = bestSeedObj;
		end
		initStats.gaTime = toc(tGA);
		initStats.usedGA = true;
	elseif cfg.verbose && strcmpi(cfg.mode, 'ga_lsq')
		if exist('ga', 'file') ~= 2
			fprintf('  GA toolbox not found; using the best seed point.\n');
		else
			fprintf('  GA disabled; using the best seed point.\n');
		end
	end
	if initStats.usedGA
		initStats.gaBestObjective = objGA;
	else
		initStats.gaBestObjective = NaN;
	end

	% Nonlinear LSQ/BP refinement.
	thetaBeforeLSQ = thetaGA;
	initStats.preLSQObjective = objTrain(thetaBeforeLSQ);
	thetaLSQ = thetaBeforeLSQ;
	bestLSQVal = Inf;
	bestLSQTrain = Inf;
	bestLSQExitFlag = NaN;
	bestLSQOutput = [];
	lsqStartMat = thetaGA(:).';
	if ~strcmpi(cfg.mode, 'ga_lsq')
		lsqStartMat = zeros(0, nVars);
	end

	if cfg.lsq.enable && nVars > 0
		nSeedStarts = max(0, round(cfg.lsq.numStarts));
		nSeedStarts = min(nSeedStarts, numel(seedOrder));
		seedStarts = theta0Mat(seedOrder(1:nSeedStarts), :);
		if strcmpi(cfg.mode, 'ga_lsq')
			lsqStartMat = [thetaGA(:).'; seedStarts];
		else
			lsqStartMat = seedStarts;
		end
		lsqStartMat = unique_rows_stable_local(lsqStartMat);
		if isempty(lsqStartMat)
			lsqStartMat = thetaBeforeLSQ(:).';
		end

		tLSQ = tic;
		if exist('lsqnonlin', 'file') == 2
			mainMaxIter = cfg.lsq.maxIter;
			mainMaxFunEvals = cfg.lsq.maxFunEvals;
			lsqOpts = optimoptions('lsqnonlin', ...
				'Display', cfg.lsq.display, ...
				'MaxIterations', mainMaxIter, ...
				'MaxFunctionEvaluations', mainMaxFunEvals, ...
				'StepTolerance', cfg.lsq.stepTolerance, ...
				'OptimalityTolerance', cfg.lsq.optimalityTolerance);
			if cfg.lsq.useAnalyticJacobian
				try
					lsqOpts = optimoptions(lsqOpts, 'SpecifyObjectiveGradient', true);
				catch
					try
						lsqOpts = optimoptions(lsqOpts, 'Jacobian', 'on');
					catch
						warning('This MATLAB version did not accept analytic-Jacobian options; falling back to finite-difference LSQ.');
						cfg.lsq.useAnalyticJacobian = false;
					end
				end
			end

			for q = 1:size(lsqStartMat, 1)
				thetaStart = lsqStartMat(q, :).';
				evalTrainFcn = @(theta) eval_mse_local(theta(:), trainMask, coefZero, Xtr, Ytr, arch, normOpt);
				evalValFcn = @(theta) eval_mse_local(theta(:), trainMask, coefZero, Xval, Yval, arch, normOpt);
				useAnalyticForStart = cfg.lsq.useAnalyticJacobian;
				jacReport = struct('checked', false);
				if cfg.lsq.debugJacobianCheck && useAnalyticForStart
					jacOpts = struct();
					jacOpts.numColumns = cfg.lsq.debugJacobianNumColumns;
					jacOpts.numSamples = cfg.lsq.debugJacobianNumSamples;
					jacOpts.relTolerance = cfg.lsq.debugJacobianRelTolerance;
					jacOpts.absTolerance = cfg.lsq.debugJacobianAbsTolerance;
					jacOpts.epsilon = cfg.lsq.debugJacobianEpsilon;
					jacReport = check_masked_phdn_jacobian(thetaStart, trainMask, coefZero, ...
						Xtr, Ytr, arch, normOpt, cfg.objective, scaleY, jacOpts);
					initStats.jacobianDiagnostics{q} = jacReport;
					if isfield(jacReport, 'checked') && jacReport.checked
						fprintf('  exact-PhDN Jacobian check start %d: ok=%d, maxRel %.3e, maxAbs %.3e, badCols %d/%d\n', ...
							q, jacReport.ok, jacReport.maxRelError, jacReport.maxAbsError, ...
							jacReport.badColumnCount, numel(jacReport.columns));
					else
						fprintf('  exact-PhDN Jacobian check start %d: failed before comparison (%s)\n', q, jacReport.reason);
					end
					if ~jacReport.ok
						if cfg.lsq.errorOnBadJacobian
							error('Exact-PhDN analytic Jacobian diagnostic failed: %s', jacReport.reason);
						elseif cfg.lsq.fallbackToFiniteDifferenceOnBadJacobian
							warning('Exact-PhDN analytic Jacobian diagnostic failed; falling back to finite-difference LSQ for this start. Reason: %s', jacReport.reason);
							useAnalyticForStart = false;
						end
					end
				end
				[thetaCand, trainCand, valCand, exitflagLSQ, outputLSQ, earlyStopStats] = ...
					run_lsq_with_validation_early_stop_local(thetaStart, resTrain, resJacTrain, ...
					useAnalyticForStart, lsqOpts, lb(:), ub(:), cfg.lsq, ...
					evalTrainFcn, evalValFcn, mainMaxIter, mainMaxFunEvals, q);
				initStats.lsqStartTable(q).startIndex = q;
				initStats.lsqStartTable(q).trainMSE = trainCand;
				initStats.lsqStartTable(q).valMSE = valCand;
				initStats.lsqStartTable(q).exitflag = exitflagLSQ;
				initStats.lsqStartTable(q).earlyStop = earlyStopStats;
				initStats.lsqStartTable(q).usedAnalyticJacobian = useAnalyticForStart;
				initStats.lsqStartTable(q).jacobianDiagnostic = jacReport;
				if isfinite(valCand) && valCand < bestLSQVal
					bestLSQVal = valCand;
					bestLSQTrain = trainCand;
					thetaLSQ = thetaCand;
					bestLSQExitFlag = exitflagLSQ;
					bestLSQOutput = outputLSQ;
				end
			end
		else
			warning('lsqnonlin was not found. Keeping the pre-LSQ initialization.');
			thetaLSQ = thetaBeforeLSQ;
			bestLSQTrain = eval_mse_local(thetaLSQ, trainMask, coefZero, Xtr, Ytr, arch, normOpt);
			bestLSQVal = eval_mse_local(thetaLSQ, trainMask, coefZero, Xval, Yval, arch, normOpt);
		end
		initStats.lsqTime = toc(tLSQ);
	else
		bestLSQTrain = eval_mse_local(thetaLSQ, trainMask, coefZero, Xtr, Ytr, arch, normOpt);
		bestLSQVal = eval_mse_local(thetaLSQ, trainMask, coefZero, Xval, Yval, arch, normOpt);
	end

	initStats.lsqExitFlag = bestLSQExitFlag;
	initStats.lsqOutput = bestLSQOutput;
	initStats.numLSQStartsActual = size(lsqStartMat, 1);

	trainMSE_before = eval_mse_local(thetaBeforeLSQ, trainMask, coefZero, Xtr, Ytr, arch, normOpt);
	valMSE_before = eval_mse_local(thetaBeforeLSQ, trainMask, coefZero, Xval, Yval, arch, normOpt);
	trainMSE_LSQ = bestLSQTrain;
	valMSE_LSQ = bestLSQVal;

	acceptLSQ = true;
	if cfg.lsq.acceptByValidation
		acceptLSQ = isfinite(valMSE_LSQ) && valMSE_LSQ <= valMSE_before * (1 + cfg.lsq.maxRelValIncrease);
	end
	if ~isfinite(valMSE_LSQ) || valMSE_LSQ >= cfg.lsq.invalidValThreshold
		acceptLSQ = false;
	end
	if acceptLSQ
		theta_est = thetaLSQ;
	else
		theta_est = thetaBeforeLSQ;
	end

	Coef_est = unpack_Coef_M_by_mask(theta_est, trainMask, coefZero);
	trainMask_est = trainMask;

	% Reporting-only Stage-2 augmentation diagnostics. This block does not
	% modify theta_est, Coef_est, trainMask_est, or any acceptance decision.
	augmentationDiag = struct('enabled',false);
	if cfg.augmentationDiagnostics.enable
		CoefBeforeLSQ = unpack_Coef_M_by_mask(thetaBeforeLSQ,trainMask,coefZero);
		CoefLSQCandidate = unpack_Coef_M_by_mask(thetaLSQ,trainMask,coefZero);
		augmentationDiag = struct();
		augmentationDiag.enabled = true;
		augmentationDiag.threshold = cfg.augmentationDiagnostics.nonzeroThreshold;
		augmentationDiag.acceptedLSQ = acceptLSQ;
		augmentationDiag.lsqExitFlag = bestLSQExitFlag;
		augmentationDiag.firstOrderOptimality = getfield_default_local( ...
			bestLSQOutput,'firstorderopt',NaN);
		augmentationDiag.parameterDeltaL2 = norm(thetaLSQ-thetaBeforeLSQ,2);
		augmentationDiag.parameterDeltaInf = norm(thetaLSQ-thetaBeforeLSQ,Inf);
		augmentationDiag.beforeLSQ = summarize_phdn_augmentation_coefficients( ...
			CoefBeforeLSQ,trainMask,arch,cfg.augmentationDiagnostics.nonzeroThreshold);
		augmentationDiag.lsqCandidate = summarize_phdn_augmentation_coefficients( ...
			CoefLSQCandidate,trainMask,arch,cfg.augmentationDiagnostics.nonzeroThreshold);
		augmentationDiag.selectedBeforePrune = summarize_phdn_augmentation_coefficients( ...
			Coef_est,trainMask_est,arch,cfg.augmentationDiagnostics.nonzeroThreshold);
		augmentationDiag.trainPerOutputBeforeLSQ = per_output_mse_from_coef_local( ...
			CoefBeforeLSQ,Xtr,Ytr,arch,normOpt);
		augmentationDiag.valPerOutputBeforeLSQ = per_output_mse_from_coef_local( ...
			CoefBeforeLSQ,Xval,Yval,arch,normOpt);
		augmentationDiag.trainPerOutputLSQCandidate = per_output_mse_from_coef_local( ...
			CoefLSQCandidate,Xtr,Ytr,arch,normOpt);
		augmentationDiag.valPerOutputLSQCandidate = per_output_mse_from_coef_local( ...
			CoefLSQCandidate,Xval,Yval,arch,normOpt);
	end

	% Post-BP coefficient-score pruning and fast BP refinement.
	pruneStats = struct('applied', false);
	if cfg.postBPPrune.enable && nVars > 0
		[theta_est, Coef_est, trainMask_est, pruneStats] = post_bp_prune_refine_local( ...
			theta_est, trainMask_est, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, scaleY);
	end
	initStats.postBPPrune = pruneStats;
	if cfg.augmentationDiagnostics.enable
		augmentationDiag.final = summarize_phdn_augmentation_coefficients( ...
			Coef_est,trainMask_est,arch,cfg.augmentationDiagnostics.nonzeroThreshold);
		augmentationDiag.trainPerOutputFinal = per_output_mse_from_coef_local( ...
			Coef_est,Xtr,Ytr,arch,normOpt);
		augmentationDiag.valPerOutputFinal = per_output_mse_from_coef_local( ...
			Coef_est,Xval,Yval,arch,normOpt);
		augmentationDiag.postBPPruneApplied = isfield(pruneStats,'applied') && pruneStats.applied;
	end
	initStats.augmentationDiagnostics = augmentationDiag;
	trainMask = trainMask_est;
	initStats.trainMSEBeforeLSQ = trainMSE_before;
	initStats.valMSEBeforeLSQ = valMSE_before;
	initStats.preLSQTrainMSE = trainMSE_before;
	initStats.preLSQValMSE = valMSE_before;
	initStats.trainMSEAfterLSQ = trainMSE_LSQ;
	initStats.valMSEAfterLSQ = valMSE_LSQ;
	initStats.acceptedLSQ = acceptLSQ;
	initStats.selectedTrainMSE = eval_mse_local(theta_est, trainMask, coefZero, Xtr, Ytr, arch, normOpt);
	initStats.selectedValMSE = eval_mse_local(theta_est, trainMask, coefZero, Xval, Yval, arch, normOpt);
	initStats.nActive = count_active_mask(trainMask);
	initStats.elapsedTime = initStats.seedSearchTime + getfield_default_local(initStats.bsp, 'elapsedTime', 0) + ...
		getfield_default_local(initStats.multiStart, 'elapsedTime', 0) + initStats.gaTime + initStats.lsqTime + ...
		getfield_default_local(pruneStats, 'elapsedTime', 0);

	if cfg.verbose
		if initStats.usedGA
			fprintf('  GA pre-LSQ val MSE              = %.6e\n', valMSE_before);
		elseif isfield(initStats.bsp, 'applied') && initStats.bsp.applied
			fprintf('  HBSP-LSQ pre-LSQ val MSE        = %.6e\n', valMSE_before);
		elseif isfield(initStats.multiStart, 'applied') && initStats.multiStart.applied
			fprintf('  multi-start pre-LSQ val MSE     = %.6e\n', valMSE_before);
		else
			fprintf('  seed init pre-LSQ val MSE       = %.6e\n', valMSE_before);
		end
		fprintf('  best LSQ val MSE                = %.6e\n', valMSE_LSQ);
		if isfield(cfg.lsq, 'earlyStop') && isfield(cfg.lsq.earlyStop, 'enable') && cfg.lsq.earlyStop.enable && ~isempty(initStats.lsqStartTable)
			[~, bestStartForPrint] = min([initStats.lsqStartTable.valMSE]);
			esPrint = initStats.lsqStartTable(bestStartForPrint).earlyStop;
			if isstruct(esPrint) && isfield(esPrint, 'enabled') && esPrint.enabled
				fprintf('  LSQ validation early stop       = 1, best start/chunk %d/%d, chunks used %d, patience %d\n', ...
					bestStartForPrint, esPrint.bestChunk, esPrint.numChunks, esPrint.valPatience);
			end
		end
		fprintf('  LSQ accepted                    = %d\n', acceptLSQ);
		fprintf('  selected train/val MSE          = %.6e / %.6e\n', initStats.selectedTrainMSE, initStats.selectedValMSE);
		bspTimeForPrint = getfield_default_local(initStats.bsp, 'elapsedTime', 0);
		if initStats.usedGA
			fprintf('  seed / GA / LSQ time            = %.3f / %.3f / %.3f s\n', ...
				initStats.seedSearchTime, initStats.gaTime, initStats.lsqTime);
		elseif bspTimeForPrint > 0
			fprintf('  seed / HBSP-LSQ / LSQ time      = %.3f / %.3f / %.3f s\n', ...
				initStats.seedSearchTime, bspTimeForPrint, initStats.lsqTime);
		elseif isfield(initStats.multiStart, 'applied') && initStats.multiStart.applied
			fprintf('  seed / multistart / LSQ time    = %.3f / %.3f / %.3f s\n', ...
				initStats.seedSearchTime, initStats.multiStart.elapsedTime, initStats.lsqTime);
		else
			fprintf('  seed / LSQ time                 = %.3f / %.3f s\n', ...
				initStats.seedSearchTime, initStats.lsqTime);
		end
		if isfield(pruneStats, 'applied') && pruneStats.applied
			fprintf('  post-BP prune/refine time       = %.3f s\n', pruneStats.elapsedTime);
		end
		if cfg.augmentationDiagnostics.enable
			print_augmentation_diagnostics_local(augmentationDiag, ...
				cfg.augmentationDiagnostics.printPerOutput);
		end
	end
end

function print_augmentation_diagnostics_local(d,printPerOutput)
	if ~isstruct(d) || ~isfield(d,'enabled') || ~d.enabled
		return;
	end
	fprintf('  augmentation diagnostics        = reporting only; training logic unchanged\n');
	fprintf('  LSQ delta L2/Inf                = %.6e / %.6e\n', ...
		d.parameterDeltaL2,d.parameterDeltaInf);
	fprintf('  LSQ exit/first-order optimality = %g / %.6e\n', ...
		d.lsqExitFlag,d.firstOrderOptimality);
	print_aug_summary_line_local('before LSQ',d.beforeLSQ);
	print_aug_summary_line_local('raw LSQ candidate',d.lsqCandidate);
	print_aug_summary_line_local('selected before prune',d.selectedBeforePrune);
	print_aug_summary_line_local('final after prune',d.final);
	if printPerOutput && isfield(d,'valPerOutputBeforeLSQ')
		n = numel(d.valPerOutputBeforeLSQ);
		fprintf('  per-output validation MSE [before -> LSQ candidate -> final]:\n');
		for r = 1:n
			fprintf('    y%-3d %.6e -> %.6e -> %.6e\n',r, ...
			d.valPerOutputBeforeLSQ(r),d.valPerOutputLSQCandidate(r), ...
			d.valPerOutputFinal(r));
		end
	end
end

function print_aug_summary_line_local(label,a)
	if ~isstruct(a) || ~isfield(a,'available') || ~a.available
		fprintf('  augmentation %-20s = unavailable\n',label);
		return;
	end
	fprintf(['  augmentation %-20s = nonzero %d/%d, neural %d/%d ', ...
		'(L2 %.3e, max %.3e), linear %d/%d, overlap cols %d\n'], ...
		label,a.nonzero,a.trainable,a.neuralNonzero,a.neuralTrainable, ...
		a.neuralL2Norm,a.neuralMaxAbs,a.linearNonzero,a.linearTrainable, ...
		a.overlapColumns);
end

function mse = per_output_mse_from_coef_local(Coef,X,Y,arch,normOpt)
	if isempty(X) || isempty(Y)
		mse = NaN(1,size(Y,2));
		return;
	end
	Yhat = model_forward(X,Coef,arch,normOpt);
	err = Yhat-Y;
	mse = mean(err.^2,1,'omitnan');
	mse(~isfinite(mse)) = Inf;
end

function tf = is_single_layer_stls_active_local(arch, cfg)
%IS_SINGLE_LAYER_STLS_ACTIVE_LOCAL True when the linear single-layer fast path is allowed.
	if ~isfield(cfg, 'singleLayerSTLS') || ~isfield(cfg.singleLayerSTLS, 'enable') || ~cfg.singleLayerSTLS.enable
		tf = false;
		return;
	end
	tf = isfield(arch, 'layer') && isequal(round(arch.layer), 1);
end

function [theta_est, initStats, Coef_est, trainMask_est] = single_layer_stls_initialize_local( ...
	Coef_template, trainMask, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, nVarsOriginal)
%SINGLE_LAYER_STLS_INITIALIZE_LOCAL Fast SINDy-style solve for arch.layer == 1.
%
% For a single-layer PhDN, Y = A{1,1} * Phi(X), so the model is linear in
% the Xi coefficients.  Nonlinear BP/LSQ, seed search, and multi-start basin
% screening are unnecessary.  This routine solves each output row by ordinary
% or ridge least squares and then performs sequential thresholded least
% squares (STLS) on the raw coefficient magnitudes.
	tFast = tic;
	if arch.layer ~= 1
		error('single_layer_stls_initialize_local was called for arch.layer=%d.', arch.layer);
	end
	if isempty(trainMask) || isempty(trainMask{1,1})
		error('Single-layer STLS requires trainMask{1,1}.');
	end
	if nargin < 8 || isempty(normOpt)
		normOpt = default_norm_options();
	end

	mask0 = trainMask{1,1};
	Coef_est = Coef_template;
	A0 = Coef_template{1,1};
	if isempty(A0)
		A0 = zeros(size(mask0));
		Coef_est{1,1} = A0;
	end
	if ~isequal(size(A0), size(mask0))
		error('Single-layer STLS mask and coefficient size mismatch.');
	end

	% Single-layer PhDN has exactly one branch A{1,1}.  The explicit
	% dictionary interface is branch-aware, so we must pass branchIndex=1.
	% Without this argument, branchwise level-3 dictionaries can fall back to
	% a dimension/global dictionary with fewer columns than trainMask{1,1},
	% which causes Phi(:, active) index-out-of-bounds errors.
	branchTr = build_branch_cache(Xtr.', arch, 1, {Xtr.'}, 1);
	PhiTr = branchTr.Phi.';
	if size(PhiTr, 2) ~= size(mask0, 2)
		error('Single-layer STLS dictionary/mask size mismatch for A{1,1}: Phi has %d columns but trainMask has %d columns. Check branchwise explicit dictionary terms.', ...
			size(PhiTr, 2), size(mask0, 2));
	end
	if ~isempty(Xval)
		branchVal = build_branch_cache(Xval.', arch, 1, {Xval.'}, 1);
		PhiVal = branchVal.Phi.';
	else
		PhiVal = zeros(0, size(PhiTr, 2));
	end

	nOut = size(A0, 1);
	if size(Ytr, 2) ~= nOut
		error('Single-layer STLS output mismatch: Ytr has %d columns but A has %d rows.', size(Ytr, 2), nOut);
	end

	stlsCfg = cfg.singleLayerSTLS;
	absThr = stlsCfg.threshold;
	if isempty(absThr) || ~isfinite(absThr)
		absThr = cfg.postBPPrune.absThreshold;
	end
	if isempty(absThr) || ~isfinite(absThr)
		absThr = 1e-10;
	end
	relThr = stlsCfg.relThreshold;
	if isempty(relThr) || ~isfinite(relThr)
		relThr = 0;
	end
	maxIter = max(0, round(stlsCfg.maxIter));
	lambda2 = max(0, stlsCfg.lambda2);
	minTermsPerRow = max(0, round(stlsCfg.minTermsPerRow));
	useValidationSelection = isfield(stlsCfg, 'useValidationSelection') && logical(stlsCfg.useValidationSelection);

	A = zeros(size(A0));
	maskOut = false(size(mask0));
	rowStats = repmat(struct('row', [], 'initialActive', [], 'finalActive', [], 'numPruned', [], ...
		'numIterations', [], 'threshold', [], 'trainMSE', [], 'valMSE', []), nOut, 1);

	for r = 1:nOut
		active = find(mask0(r, :));
		initialActive = numel(active);
		if isempty(active)
			continue;
		end
		y = Ytr(:, r);
		activeBest = active;
		coefBest = zeros(numel(active), 1);
		valBest = Inf;
		trainBest = Inf;
		lastCoef = [];
		lastActive = active;
		for it = 1:maxIter + 1
			coef = solve_linear_lsq_local(PhiTr(:, active), y, lambda2);
			lastCoef = coef;
			lastActive = active;
			trainRow = mean_square_omitnan_local(PhiTr(:, active) * coef - y);
			if ~isempty(PhiVal) && ~isempty(Yval)
				valRow = mean_square_omitnan_local(PhiVal(:, active) * coef - Yval(:, r));
			else
				valRow = trainRow;
			end
			if ~useValidationSelection || valRow < valBest
				activeBest = active;
				coefBest = coef;
				valBest = valRow;
				trainBest = trainRow;
			end
			if it > maxIter
				break;
			end
			score = abs(coef);
			thr = max(absThr, relThr * max(score));
			removeLocal = find(score <= thr);
			if isempty(removeLocal)
				break;
			end
			if minTermsPerRow > 0
				maxRemove = max(0, numel(active) - minTermsPerRow);
				if numel(removeLocal) > maxRemove
					[~, ord] = sort(score(removeLocal), 'ascend');
					removeLocal = removeLocal(ord(1:maxRemove));
				end
			end
			if isempty(removeLocal)
				break;
			end
			active(removeLocal) = [];
			if isempty(active)
				break;
			end
		end
		if ~useValidationSelection
			activeBest = lastActive;
			coefBest = lastCoef;
			trainBest = mean_square_omitnan_local(PhiTr(:, activeBest) * coefBest - y);
			if ~isempty(PhiVal) && ~isempty(Yval)
				valBest = mean_square_omitnan_local(PhiVal(:, activeBest) * coefBest - Yval(:, r));
			else
				valBest = trainBest;
			end
		end
		A(r, activeBest) = coefBest(:).';
		maskOut(r, activeBest) = true;
		rowStats(r).row = r;
		rowStats(r).initialActive = initialActive;
		rowStats(r).finalActive = numel(activeBest);
		rowStats(r).numPruned = initialActive - numel(activeBest);
		rowStats(r).numIterations = maxIter;
		rowStats(r).threshold = absThr;
		rowStats(r).trainMSE = trainBest;
		rowStats(r).valMSE = valBest;
	end

	Coef_est{1,1} = A;
	trainMask_est = trainMask;
	trainMask_est{1,1} = maskOut;
	theta_est = pack_Coef_M_by_mask(Coef_est, trainMask_est);

	trainMSE = eval_mse_from_coef_local(Coef_est, Xtr, Ytr, arch, normOpt);
	valMSE = eval_mse_from_coef_local(Coef_est, Xval, Yval, arch, normOpt);
	activeFinal = count_active_mask(trainMask_est);
	elapsed = toc(tFast);

	initStats = struct();
	initStats.method = 'single_layer_stls_sindy';
	initStats.mode = cfg.mode;
	initStats.requestedMode = cfg.requestedMode;
	initStats.nVarsOriginal = nVarsOriginal;
	initStats.nVars = activeFinal;
	initStats.seedMode = 'none_single_layer_stls';
	initStats.numSeedCandidatesRequested = 0;
	initStats.numSeedCandidatesActual = 0;
	initStats.numLSQStartsRequested = 0;
	initStats.numLSQStartsActual = 0;
	initStats.populationSize = 0;
	initStats.maxGenerations = 0;
	initStats.maxStallGenerations = 0;
	initStats.gaExitFlag = NaN;
	initStats.gaOutput = [];
	initStats.gaBestObjective = NaN;
	initStats.seedBestObjective = NaN;
	initStats.randomBestObjective = NaN;
	initStats.lsqExitFlag = NaN;
	initStats.lsqOutput = [];
	initStats.gaTime = 0;
	initStats.usedGA = false;
	initStats.preLSQObjective = NaN;
	initStats.preLSQTrainMSE = trainMSE;
	initStats.preLSQValMSE = valMSE;
	initStats.seedSearchTime = 0;
	initStats.randomSearchTime = 0;
	initStats.lsqTime = elapsed;
	initStats.lsqStartTable = struct([]);
	initStats.jacobianDiagnostics = {};   % cell array: reports may have diagnostic-dependent fields
	initStats.bsp = struct('applied', false);
	initStats.multiStart = struct('applied', false, 'skippedBySingleLayerSTLS', true);
	initStats.postBPPrune = struct('applied', false, 'skippedBySingleLayerSTLS', true);
	initStats.singleLayerSTLS = struct('applied', true, 'threshold', absThr, 'relThreshold', relThr, ...
		'maxIter', maxIter, 'lambda2', lambda2, 'minTermsPerRow', minTermsPerRow, ...
		'useValidationSelection', useValidationSelection, 'rowStats', rowStats, 'elapsedTime', elapsed, ...
		'originalActive', nVarsOriginal, 'finalActive', activeFinal);
	initStats.trainMSEBeforeLSQ = trainMSE;
	initStats.valMSEBeforeLSQ = valMSE;
	initStats.trainMSEAfterLSQ = trainMSE;
	initStats.valMSEAfterLSQ = valMSE;
	initStats.acceptedLSQ = true;
	initStats.selectedTrainMSE = trainMSE;
	initStats.selectedValMSE = valMSE;
	initStats.nActive = activeFinal;
	initStats.elapsedTime = elapsed;

	if cfg.verbose || stlsCfg.verbose
		fprintf('  single-layer STLS solved Xi      = active %d -> %d, threshold %.3e, maxIter %d, lambda2 %.3e\n', ...
			nVarsOriginal, activeFinal, absThr, maxIter, lambda2);
		fprintf('  single-layer STLS train/val MSE  = %.6e / %.6e\n', trainMSE, valMSE);
		fprintf('  single-layer STLS time           = %.3f s\n', elapsed);
	end
end

function mse = mean_square_omitnan_local(err)
	err = err(:);
	err = err(isfinite(err));
	if isempty(err)
		mse = Inf;
	else
		mse = mean(err.^2);
	end
end

function coef = solve_linear_lsq_local(Phi, y, lambda2)
%SOLVE_LINEAR_LSQ_LOCAL Small robust LSQ/ridge solver for STLS.
	if isempty(Phi)
		coef = zeros(0, 1);
		return;
	end
	Phi(~isfinite(Phi)) = 0;
	y(~isfinite(y)) = 0;
	if lambda2 > 0
		n = size(Phi, 2);
		coef = [Phi; sqrt(lambda2) * eye(n)] \ [y; zeros(n, 1)];
	else
		coef = Phi \ y;
	end
	coef(~isfinite(coef)) = 0;
end

function mse = eval_mse_from_coef_local(Coef, X, Y, arch, normOpt)
	if isempty(X) || isempty(Y)
		mse = NaN;
		return;
	end
	Yp = model_forward(X, Coef, arch, normOpt);
	E = Yp - Y;
	mse = mean_square_omitnan_local(E(:));
end

function stats = direct_multistart_lsqbp_screen_local(theta0Mat, seedOrder, trainMask, coefZero, ...
	Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, scaleY, lb, ub, resTrain, resJacTrain)
%DIRECT_MULTISTART_LSQBP_SCREEN_LOCAL Lightweight basin selection for direct BP-LSQ.
%
% Required flow:
%   multiple simple seeds -> independent light LSQ/BP screening for each seed
%   -> choose the best validation start -> pass this theta to the final heavy
%   LSQ/BP refinement outside this helper.
	stats = struct('applied', false, 'numStarts', 0, 'bestStartIndex', NaN, ...
		'bestTrainMSE', NaN, 'bestValMSE', NaN, 'bestTheta', [], 'elapsedTime', 0, ...
		'usedParallel', false, 'startTable', struct([]));
	if isempty(theta0Mat) || isempty(seedOrder)
		return;
	end
	nVars = size(theta0Mat, 2);
	if nVars == 0
		return;
	end
	nStarts = min(max(1, cfg.multiStart.numStarts), numel(seedOrder));
	startRows = seedOrder(1:nStarts);
	stats.numStarts = nStarts;

	bestVal = Inf;
	bestTrain = Inf;
	bestTheta = theta0Mat(startRows(1), :).';
	bestStartIndex = startRows(1);
	tScreen = tic;
	evalTrainFcn = @(theta) eval_mse_local(theta(:), trainMask, coefZero, Xtr, Ytr, arch, normOpt);
	evalValFcn = @(theta) eval_mse_local(theta(:), trainMask, coefZero, Xval, Yval, arch, normOpt);

	thetaCell = cell(nStarts, 1);
	trainVec = inf(nStarts, 1);
	valVec = inf(nStarts, 1);
	exitVec = nan(nStarts, 1);
	outputCell = cell(nStarts, 1);
	earlyCell = cell(nStarts, 1);

	useParallel = should_use_parallel_multistart_local(cfg.multiStart);
	stats.usedParallel = useParallel;
	if cfg.verbose
		if useParallel
			fprintf('  direct multi-start screening     = evaluating %d starts in parallel ...\n', nStarts);
		else
			fprintf('  direct multi-start screening     = evaluating %d starts serially ...\n', nStarts);
		end
	end

	if exist('lsqnonlin', 'file') == 2 && cfg.multiStart.screenMaxIter > 0
		screenCfg = cfg.lsq;
		screenCfg.maxIter = cfg.multiStart.screenMaxIter;
		screenCfg.maxFunEvals = cfg.multiStart.screenMaxFunEvals;
		% Keep screening cheap. The final heavy LSQ/BP below still uses the
		% normal cfg.lsq settings and runs once from the selected best theta.
		lsqOpts = optimoptions('lsqnonlin', ...
			'Display', cfg.lsq.display, ...
			'MaxIterations', screenCfg.maxIter, ...
			'MaxFunctionEvaluations', screenCfg.maxFunEvals, ...
			'StepTolerance', cfg.lsq.stepTolerance, ...
			'OptimalityTolerance', cfg.lsq.optimalityTolerance);
		if cfg.lsq.useAnalyticJacobian
			try
				lsqOpts = optimoptions(lsqOpts, 'SpecifyObjectiveGradient', true);
			catch
				try
					lsqOpts = optimoptions(lsqOpts, 'Jacobian', 'on');
				catch
					screenCfg.useAnalyticJacobian = false;
				end
			end
		end

		if useParallel
			parfor q = 1:nStarts
				idx = startRows(q);
				thetaStart = theta0Mat(idx, :).';
				[thetaCand, trainCand, valCand, exitflagLSQ, outputLSQ, earlyStopStats] = ...
					screen_lsqbp_start_once_local(thetaStart, trainMask, coefZero, Xtr, Ytr, Xval, Yval, ...
					arch, normOpt, cfg, scaleY, lb(:), ub(:), screenCfg, lsqOpts, q);
				thetaCell{q} = thetaCand(:);
				trainVec(q) = trainCand;
				valVec(q) = valCand;
				exitVec(q) = exitflagLSQ;
				outputCell{q} = outputLSQ;
				earlyCell{q} = earlyStopStats;
			end
		else
			for q = 1:nStarts
				idx = startRows(q);
				thetaStart = theta0Mat(idx, :).';
				[thetaCand, trainCand, valCand, exitflagLSQ, outputLSQ, earlyStopStats] = ...
					screen_lsqbp_start_once_local(thetaStart, trainMask, coefZero, Xtr, Ytr, Xval, Yval, ...
					arch, normOpt, cfg, scaleY, lb(:), ub(:), screenCfg, lsqOpts, q);
				thetaCell{q} = thetaCand(:);
				trainVec(q) = trainCand;
				valVec(q) = valCand;
				exitVec(q) = exitflagLSQ;
				outputCell{q} = outputLSQ;
				earlyCell{q} = earlyStopStats;
			end
		end
	else
		% Fallback: if lsqnonlin is unavailable, select by raw seed validation error.
		for q = 1:nStarts
			idx = startRows(q);
			thetaCand = theta0Mat(idx, :).';
			trainVec(q) = evalTrainFcn(thetaCand);
			valVec(q) = evalValFcn(thetaCand);
			thetaCell{q} = thetaCand(:);
			exitVec(q) = NaN;
			outputCell{q} = [];
			earlyCell{q} = struct('enabled', false, 'numChunks', 0, 'bestChunk', NaN);
		end
	end

	for q = 1:nStarts
		idx = startRows(q);
		stats.startTable(q).seedIndex = idx;
		stats.startTable(q).trainMSE = trainVec(q);
		stats.startTable(q).valMSE = valVec(q);
		stats.startTable(q).exitflag = exitVec(q);
		stats.startTable(q).output = outputCell{q};
		stats.startTable(q).earlyStop = earlyCell{q};
		if isfinite(valVec(q)) && valVec(q) < bestVal
			bestVal = valVec(q);
			bestTrain = trainVec(q);
			bestTheta = thetaCell{q}(:);
			bestStartIndex = idx;
		end
	end

	% If all validation scores are invalid, fall back to the best finite train MSE.
	if ~isfinite(bestVal)
		[bestTrainFallback, qBestTrain] = min(trainVec);
		if isfinite(bestTrainFallback)
			bestTrain = bestTrainFallback;
			bestVal = valVec(qBestTrain);
			bestTheta = thetaCell{qBestTrain}(:);
			bestStartIndex = startRows(qBestTrain);
		end
	end

	stats.applied = true;
	stats.bestStartIndex = bestStartIndex;
	stats.bestTrainMSE = bestTrain;
	stats.bestValMSE = bestVal;
	stats.bestTheta = bestTheta(:);
	stats.elapsedTime = toc(tScreen);
end

function [thetaCand, trainCand, valCand, exitflagLSQ, outputLSQ, earlyStopStats] = screen_lsqbp_start_once_local( ...
	thetaStart, trainMask, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, scaleY, lb, ub, screenCfg, lsqOpts, startIndex)
%SCREEN_LSQBP_START_ONCE_LOCAL Run one independent light LSQ/BP screening start.
%
% This helper is intentionally self-contained so that parallel workers do not
% rely on function handles captured from the parent workspace.
	resTrainLocal = @(theta) residual_vector_local(theta(:), trainMask, coefZero, Xtr, Ytr, arch, normOpt, cfg.objective, scaleY);
	resJacTrainLocal = @(theta) residual_jacobian_masked_phdn(theta(:), trainMask, coefZero, Xtr, Ytr, arch, normOpt, cfg.objective, scaleY);
	evalTrainFcnLocal = @(theta) eval_mse_local(theta(:), trainMask, coefZero, Xtr, Ytr, arch, normOpt);
	evalValFcnLocal = @(theta) eval_mse_local(theta(:), trainMask, coefZero, Xval, Yval, arch, normOpt);
	[thetaCand, trainCand, valCand, exitflagLSQ, outputLSQ, earlyStopStats] = ...
		run_lsq_with_validation_early_stop_local(thetaStart, resTrainLocal, resJacTrainLocal, ...
		screenCfg.useAnalyticJacobian, lsqOpts, lb(:), ub(:), screenCfg, ...
		evalTrainFcnLocal, evalValFcnLocal, screenCfg.maxIter, screenCfg.maxFunEvals, startIndex);
end

function tf = should_use_parallel_multistart_local(msCfg)
	% Parallel screening is optional and mirrors the BSP candidate-evaluation logic.
	tf = false;
	if ~getfield_default_local(msCfg, 'useParallel', false)
		return;
	end
	try
		if exist('gcp', 'file') ~= 2
			return;
		end
		pool = gcp('nocreate');
		if isempty(pool) && getfield_default_local(msCfg, 'autoStartParallelPool', true)
			try
				pool = parpool;
			catch
				pool = [];
			end
		end
		tf = ~isempty(pool);
	catch
		tf = false;
	end
end

function cfg = normalize_init_cfg_local(opts, useAdmissibleMask)
	cfg = struct();
	cfg.mode = getfield_default_local(opts.init, 'mode', 'auto');
	cfg.mode = lower(strtrim(char(cfg.mode)));
	if strcmpi(cfg.mode, 'auto')
		if useAdmissibleMask
			cfg.mode = 'ga_lsq';
		else
			cfg.mode = 'random_lsq';
		end
	end
	cfg.requestedMode = cfg.mode;
	cfg.errorOnEmptyMask = getfield_default_local(opts.init, 'errorOnEmptyMask', true);
	cfg.maxGAParams = getfield_default_local(opts.init, 'maxGAParams', 120);
	cfg.allowLargeNoMaskProblem = getfield_default_local(opts.init, 'allowLargeNoMaskProblem', false);
	cfg.largeNoMaskPolicy = getfield_default_local(opts.init, 'largeNoMaskPolicy', 'fallback_random_lsq');
	cfg.noMaskSeedCandidates = getfield_default_local(opts.init, 'noMaskSeedCandidates', 60);
	cfg.verbose = getfield_default_local(opts.init, 'verbose', true);
	cfg.debugMaskSummary = getfield_default_local(opts.init, 'debugMaskSummary', true);
	cfg.augmentationDiagnostics = getfield_default_local(opts.init,'augmentationDiagnostics',struct());
	cfg.augmentationDiagnostics.enable = logical(getfield_default_local( ...
		cfg.augmentationDiagnostics,'enable',false));
	cfg.augmentationDiagnostics.nonzeroThreshold = max(0,double(getfield_default_local( ...
		cfg.augmentationDiagnostics,'nonzeroThreshold',1e-10)));
	cfg.augmentationDiagnostics.printPerOutput = logical(getfield_default_local( ...
		cfg.augmentationDiagnostics,'printPerOutput',true));
	if isfield(opts.init, 'bounds')
		cfg.bounds = opts.init.bounds;
	else
		cfg.bounds = struct('lower', -3, 'upper', 3);
	end
	cfg.seed = getfield_default_local(opts.init, 'seed', struct());
	cfg.seed.mode = getfield_default_local(cfg.seed, 'mode', 'scale_aware');
	cfg.seed.numCandidates = getfield_default_local(cfg.seed, 'numCandidates', 40);
	cfg.seed.rngSeed = getfield_default_local(cfg.seed, 'rngSeed', 1);
	% Dedicated initial-point range; omitted fields inherit the optimization
	% bounds, preserving the historical single-bound behavior.
	cfg.seed.lowerBound = getfield_default_local(cfg.seed, 'lowerBound', cfg.bounds.lower);
	cfg.seed.upperBound = getfield_default_local(cfg.seed, 'upperBound', cfg.bounds.upper);
	cfg.seed.useScaleAware = getfield_default_local(cfg.seed, 'useScaleAware', strcmpi(cfg.seed.mode, 'scale_aware'));
	if cfg.seed.useScaleAware, cfg.seed.mode = 'scale_aware'; end
	cfg.seed.scaleAwareAlpha = getfield_default_local(cfg.seed, 'scaleAwareAlpha', 0.5);
	cfg.seed.scaleAwareAlphaList = getfield_default_local(cfg.seed, 'scaleAwareAlphaList', cfg.seed.scaleAwareAlpha);
	cfg.seed.scaleAwareHiddenTargetMode = getfield_default_local(cfg.seed, 'scaleAwareHiddenTargetMode', 'clipped_output_std');
	cfg.seed.scaleAwareOutputTargetMode = getfield_default_local(cfg.seed, 'scaleAwareOutputTargetMode', 'output_std');
	cfg.seed.scaleAwareHiddenTargetStdCap = getfield_default_local(cfg.seed, 'scaleAwareHiddenTargetStdCap', 1.0);
	cfg.seed.scaleAwareFixedTargetStd = getfield_default_local(cfg.seed, 'scaleAwareFixedTargetStd', 1.0);
	cfg.seed.scaleAwareTargetStdFloor = getfield_default_local(cfg.seed, 'scaleAwareTargetStdFloor', 1e-6);
	cfg.seed.scaleAwareFeatureStdFloor = getfield_default_local(cfg.seed, 'scaleAwareFeatureStdFloor', 1e-8);
	cfg.seed.scaleAwareConstantStdFactor = getfield_default_local(cfg.seed, 'scaleAwareConstantStdFactor', 1.0);
	cfg.seed.scaleAwareCalibrateLayerOutput = getfield_default_local(cfg.seed, 'scaleAwareCalibrateLayerOutput', true);
	cfg.seed.scaleAwareCalibrationFactor = getfield_default_local(cfg.seed, 'scaleAwareCalibrationFactor', 0.5);
	cfg.seed.scaleAwareLayerScaleMin = getfield_default_local(cfg.seed, 'scaleAwareLayerScaleMin', 0.1);
	cfg.seed.scaleAwareLayerScaleMax = getfield_default_local(cfg.seed, 'scaleAwareLayerScaleMax', 10);
	cfg.seed.scaleAwareRemoveInvalidRows = getfield_default_local(cfg.seed, 'scaleAwareRemoveInvalidRows', true);
	cfg.seed.scaleAwarePostJitter = getfield_default_local(cfg.seed, 'scaleAwarePostJitter', 0.02);
	cfg.seed.externalTheta = getfield_default_local(cfg.seed, 'externalTheta', []);
	cfg.seed.externalThetaSource = getfield_default_local(cfg.seed, 'externalThetaSource', 'external_seed');
	cfg.seed.prependExternalTheta = getfield_default_local(cfg.seed, 'prependExternalTheta', true);
	cfg.ga = getfield_default_local(opts.init, 'ga', struct());
	cfg.ga.enable = getfield_default_local(cfg.ga, 'enable', strcmpi(cfg.mode, 'ga_lsq'));
	cfg.ga.autoConfigureEffort = getfield_default_local(cfg.ga, 'autoConfigureEffort', true);
	cfg.ga.effort = getfield_default_local(cfg.ga, 'effort', 'quick');
	cfg.ga.populationSize = getfield_default_local(cfg.ga, 'populationSize', 40);
	cfg.ga.maxGenerations = getfield_default_local(cfg.ga, 'maxGenerations', 20);
	cfg.ga.maxStallGenerations = getfield_default_local(cfg.ga, 'maxStallGenerations', 8);
	cfg.ga.functionTolerance = getfield_default_local(cfg.ga, 'functionTolerance', 1e-8);
	cfg.ga.display = getfield_default_local(cfg.ga, 'display', 'off');
	cfg.ga.useParallel = getfield_default_local(cfg.ga, 'useParallel', false);
	cfg.ga.respectAutoCaps = getfield_default_local(cfg.ga, 'respectAutoCaps', false);
	cfg.ga.targetETPD = getfield_default_local(cfg.ga, 'targetETPD', []);
	cfg.ga.generationPopulationRatio = getfield_default_local(cfg.ga, 'generationPopulationRatio', 1/3);
	cfg.ga.capLimited = getfield_default_local(cfg.ga, 'capLimited', false);
	cfg.ga.effectiveETPD = getfield_default_local(cfg.ga, 'effectiveETPD', []);
	cfg.lsq = getfield_default_local(opts.init, 'lsq', struct());
	cfg.lsq.enable = getfield_default_local(cfg.lsq, 'enable', true);
	cfg.lsq.numStarts = getfield_default_local(cfg.lsq, 'numStarts', 1);
	cfg.lsq.maxIter = getfield_default_local(cfg.lsq, 'maxIter', 500);
	cfg.lsq.maxFunEvals = getfield_default_local(cfg.lsq, 'maxFunEvals', 5e4);
	cfg.lsq.display = getfield_default_local(cfg.lsq, 'display', 'off');
	cfg.lsq.stepTolerance = getfield_default_local(cfg.lsq, 'stepTolerance', 1e-10);
	cfg.lsq.optimalityTolerance = getfield_default_local(cfg.lsq, 'optimalityTolerance', 1e-8);
	cfg.lsq.acceptByValidation = getfield_default_local(cfg.lsq, 'acceptByValidation', true);
	cfg.lsq.maxRelValIncrease = getfield_default_local(cfg.lsq, 'maxRelValIncrease', 1e-10);
	cfg.lsq.useAnalyticJacobian = getfield_default_local(cfg.lsq, 'useAnalyticJacobian', true);
	cfg.lsq.invalidValThreshold = getfield_default_local(cfg.lsq, 'invalidValThreshold', 1e8);
	% Exact dictionary-PhDN Jacobian diagnostics. These checks are intentionally
	% attached to the Stage-II / true-operator compact-mask route, not to the
	% Stage-I branch-MLP surrogate.
	cfg.lsq.debugJacobianCheck = getfield_default_local(cfg.lsq, 'debugJacobianCheck', false);
	cfg.lsq.debugJacobianNumColumns = getfield_default_local(cfg.lsq, 'debugJacobianNumColumns', 12);
	cfg.lsq.debugJacobianNumSamples = getfield_default_local(cfg.lsq, 'debugJacobianNumSamples', 40);
	cfg.lsq.debugJacobianRelTolerance = getfield_default_local(cfg.lsq, 'debugJacobianRelTolerance', 1e-3);
	cfg.lsq.debugJacobianAbsTolerance = getfield_default_local(cfg.lsq, 'debugJacobianAbsTolerance', 1e-6);
	cfg.lsq.debugJacobianEpsilon = getfield_default_local(cfg.lsq, 'debugJacobianEpsilon', 1e-6);
	cfg.lsq.fallbackToFiniteDifferenceOnBadJacobian = getfield_default_local(cfg.lsq, 'fallbackToFiniteDifferenceOnBadJacobian', false);
	cfg.lsq.errorOnBadJacobian = getfield_default_local(cfg.lsq, 'errorOnBadJacobian', false);
	cfg.lsq.earlyStop = getfield_default_local(cfg.lsq, 'earlyStop', struct());
	cfg.lsq.earlyStop.enable = getfield_default_local(cfg.lsq.earlyStop, 'enable', true);
	cfg.lsq.earlyStop.chunkMaxIter = getfield_default_local(cfg.lsq.earlyStop, 'chunkMaxIter', 50);
	cfg.lsq.earlyStop.chunkMaxFunEvals = getfield_default_local(cfg.lsq.earlyStop, 'chunkMaxFunEvals', []);
	cfg.lsq.earlyStop.maxChunks = getfield_default_local(cfg.lsq.earlyStop, 'maxChunks', []);
	cfg.lsq.earlyStop.minChunks = getfield_default_local(cfg.lsq.earlyStop, 'minChunks', 1);
	cfg.lsq.earlyStop.valPatience = getfield_default_local(cfg.lsq.earlyStop, 'valPatience', 3);
	cfg.lsq.earlyStop.relImproveTol = getfield_default_local(cfg.lsq.earlyStop, 'relImproveTol', 1e-4);
	cfg.lsq.earlyStop.absImproveTol = getfield_default_local(cfg.lsq.earlyStop, 'absImproveTol', 0);
	cfg.lsq.earlyStop.restoreBestValidation = getfield_default_local(cfg.lsq.earlyStop, 'restoreBestValidation', true);
	cfg.lsq.earlyStop.stopOnSolverConvergence = getfield_default_local(cfg.lsq.earlyStop, 'stopOnSolverConvergence', true);
	cfg.lsq.earlyStop.stopOnInvalidVal = getfield_default_local(cfg.lsq.earlyStop, 'stopOnInvalidVal', true);
	cfg.lsq.earlyStop.verbose = getfield_default_local(cfg.lsq.earlyStop, 'verbose', false);
	cfg.bsp = getfield_default_local(opts.init, 'bsp', struct());
	cfg.bsp.enable = getfield_default_local(cfg.bsp, 'enable', false);
	cfg.bsp.beamWidth = max(1, round(getfield_default_local(cfg.bsp, 'beamWidth', 5)));
	cfg.bsp.maxRounds = max(0, round(getfield_default_local(cfg.bsp, 'maxRounds', Inf)));
	cfg.bsp.minTermsPerXiBlock = max(0, round(getfield_default_local(cfg.bsp, 'minTermsPerXiBlock', 0)));
	cfg.bsp.minTotalActive = max(0, round(getfield_default_local(cfg.bsp, 'minTotalActive', 0)));
	cfg.bsp.fastMaxIter = max(0, round(getfield_default_local(cfg.bsp, 'fastMaxIter', 20)));
	cfg.bsp.fastMaxFunEvals = max(100, round(getfield_default_local(cfg.bsp, 'fastMaxFunEvals', 2000)));
	cfg.bsp.fastNumStarts = max(1, round(getfield_default_local(cfg.bsp, 'fastNumStarts', 1)));
	cfg.bsp.sizePenalty = getfield_default_local(cfg.bsp, 'sizePenalty', 0);
	cfg.bsp.relImproveTol = max(0, getfield_default_local(cfg.bsp, 'relImproveTol', 1e-4));
	cfg.bsp.absImproveTol = max(0, getfield_default_local(cfg.bsp, 'absImproveTol', 0));
	cfg.bsp.maxRelValIncrease = getfield_default_local(cfg.bsp, 'maxRelValIncrease', Inf);
	cfg.bsp.patience = max(0, round(getfield_default_local(cfg.bsp, 'patience', 0)));
	cfg.bsp.keepParentsInBeam = getfield_default_local(cfg.bsp, 'keepParentsInBeam', true);
	cfg.bsp.skipWhenAdmissibleMask = getfield_default_local(cfg.bsp, 'skipWhenAdmissibleMask', true);
	cfg.bsp.printBranchMasks = getfield_default_local(cfg.bsp, 'printBranchMasks', true);
	cfg.bsp.verbose = getfield_default_local(cfg.bsp, 'verbose', true);
	cfg.bsp.useParallel = getfield_default_local(cfg.bsp, 'useParallel', true);
	cfg.bsp.autoStartParallelPool = getfield_default_local(cfg.bsp, 'autoStartParallelPool', true);
	cfg.bsp.parallelVerbose = getfield_default_local(cfg.bsp, 'parallelVerbose', true);
	cfg.bsp.branchEnable = getfield_default_local(cfg.bsp, 'branchEnable', true);
	cfg.bsp.branchBeamWidth = max(1, round(getfield_default_local(cfg.bsp, 'branchBeamWidth', cfg.bsp.beamWidth)));
	cfg.bsp.branchMaxRounds = max(0, round(getfield_default_local(cfg.bsp, 'branchMaxRounds', Inf)));
	cfg.bsp.branchMinActiveBranches = max(0, round(getfield_default_local(cfg.bsp, 'branchMinActiveBranches', 1)));
	cfg.bsp.branchFastMaxIter = max(0, round(getfield_default_local(cfg.bsp, 'branchFastMaxIter', cfg.bsp.fastMaxIter)));
	cfg.bsp.branchFastMaxFunEvals = max(100, round(getfield_default_local(cfg.bsp, 'branchFastMaxFunEvals', cfg.bsp.fastMaxFunEvals)));
	cfg.bsp.branchRelImproveTol = max(0, getfield_default_local(cfg.bsp, 'branchRelImproveTol', cfg.bsp.relImproveTol));
	cfg.bsp.branchAbsImproveTol = max(0, getfield_default_local(cfg.bsp, 'branchAbsImproveTol', cfg.bsp.absImproveTol));
	cfg.bsp.branchPatience = max(0, round(getfield_default_local(cfg.bsp, 'branchPatience', cfg.bsp.patience)));
	cfg.bsp.branchKeepParentsInBeam = getfield_default_local(cfg.bsp, 'branchKeepParentsInBeam', cfg.bsp.keepParentsInBeam);
	cfg.bsp.branchSizePenalty = getfield_default_local(cfg.bsp, 'branchSizePenalty', 0);
	cfg.bsp.branchMode = lower(strtrim(char(getfield_default_local(cfg.bsp, 'branchMode', 'saliency_greedy'))));
	if any(strcmpi(cfg.bsp.branchMode, {'saliency', 'greedy_saliency', 'jacobian_saliency', 'jacobian_greedy'}))
		cfg.bsp.branchMode = 'saliency_greedy';
	elseif any(strcmpi(cfg.bsp.branchMode, {'beam', 'beam_validation', 'validation_beam', 'bsp'}))
		cfg.bsp.branchMode = 'beam_validation';
	end
	cfg.bsp.branchSaliencyTopK = max(1, round(getfield_default_local(cfg.bsp, 'branchSaliencyTopK', 3)));
	cfg.bsp.branchSaliencyUseLayerNorm = getfield_default_local(cfg.bsp, 'branchSaliencyUseLayerNorm', false);
	cfg.bsp.branchSaliencyAcceptByValidation = getfield_default_local(cfg.bsp, 'branchSaliencyAcceptByValidation', true);
	cfg.bsp.branchSaliencyMaxRelValIncrease = max(0, getfield_default_local(cfg.bsp, 'branchSaliencyMaxRelValIncrease', 0));
	cfg.bsp.branchSaliencyRefineBeforeScoring = getfield_default_local(cfg.bsp, 'branchSaliencyRefineBeforeScoring', true);
	cfg.bsp.branchSaliencyScoreMode = lower(strtrim(char(getfield_default_local(cfg.bsp, 'branchSaliencyScoreMode', 'topk_mean_output_delta'))));
	cfg.bsp.basisEnable = getfield_default_local(cfg.bsp, 'basisEnable', true);

	cfg.multiStart = getfield_default_local(opts.init, 'multistart', struct());
	cfg.multiStart.enable = getfield_default_local(cfg.multiStart, 'enable', false);
	cfg.multiStart.numStarts = max(1, round(getfield_default_local(cfg.multiStart, 'numStarts', 5)));
	cfg.multiStart.screenMaxIter = max(0, round(getfield_default_local(cfg.multiStart, 'screenMaxIter', 20)));
	cfg.multiStart.screenMaxFunEvals = max(100, round(getfield_default_local(cfg.multiStart, 'screenMaxFunEvals', 2000)));
	cfg.multiStart.useParallel = getfield_default_local(cfg.multiStart, 'useParallel', true);
	cfg.multiStart.autoStartParallelPool = getfield_default_local(cfg.multiStart, 'autoStartParallelPool', true);

	cfg.skip = getfield_default_local(opts.init, 'skip', struct());
	cfg.skip.useExternalStage0Seed = getfield_default_local(cfg.skip, 'useExternalStage0Seed', true);
	cfg.skip.useBaselineScreen = getfield_default_local(cfg.skip, 'useBaselineScreen', true);
	cfg.skip.numSeedCandidates = max(1, round(getfield_default_local(cfg.skip, 'numSeedCandidates', 40)));
	cfg.skip.numFinalBPStarts = max(1, round(getfield_default_local(cfg.skip, 'numFinalBPStarts', 8)));
	cfg.skip.screenMaxIter = max(0, round(getfield_default_local(cfg.skip, 'screenMaxIter', 30)));
	cfg.skip.screenMaxFunEvals = max(100, round(getfield_default_local(cfg.skip, 'screenMaxFunEvals', 1500)));
	cfg.skip.useParallel = getfield_default_local(cfg.skip, 'useParallel', false);
	cfg.skip.autoStartParallelPool = getfield_default_local(cfg.skip, 'autoStartParallelPool', false);
	cfg.skip.hasExternalSeed = isfield(cfg.seed, 'externalTheta') && ~isempty(cfg.seed.externalTheta);

	cfg.singleLayerSTLS = getfield_default_local(opts.init, 'singleLayerSTLS', struct());
	cfg.singleLayerSTLS.enable = getfield_default_local(cfg.singleLayerSTLS, 'enable', true);
	cfg.singleLayerSTLS.threshold = getfield_default_local(cfg.singleLayerSTLS, 'threshold', []);
	cfg.singleLayerSTLS.relThreshold = getfield_default_local(cfg.singleLayerSTLS, 'relThreshold', 0);
	cfg.singleLayerSTLS.maxIter = max(0, round(getfield_default_local(cfg.singleLayerSTLS, 'maxIter', 10)));
	cfg.singleLayerSTLS.lambda2 = max(0, getfield_default_local(cfg.singleLayerSTLS, 'lambda2', 0));
	cfg.singleLayerSTLS.minTermsPerRow = max(0, round(getfield_default_local(cfg.singleLayerSTLS, 'minTermsPerRow', 1)));
	cfg.singleLayerSTLS.useValidationSelection = getfield_default_local(cfg.singleLayerSTLS, 'useValidationSelection', false);
	cfg.singleLayerSTLS.verbose = getfield_default_local(cfg.singleLayerSTLS, 'verbose', true);

	cfg.postBPPrune = getfield_default_local(opts.init, 'postBPPrune', struct());
	cfg.postBPPrune.enable = getfield_default_local(cfg.postBPPrune, 'enable', true);
	cfg.postBPPrune.numIterations = max(0, round(getfield_default_local(cfg.postBPPrune, 'numIterations', 1)));
	% v50a-compatible default: prune by mean absolute contribution, then run a short
	% support-fixed BP-LSQ refinement.  Raw coefficient pruning remains available
	% through postBPPrune.scoreMode='coef_abs'.
	cfg.postBPPrune.scoreMode = lower(strtrim(char(getfield_default_local(cfg.postBPPrune, 'scoreMode', 'contribution_abs_mean'))));
	if any(strcmpi(cfg.postBPPrune.scoreMode, {'coef','coef_abs','coefficient','coefficient_abs','abs_coef','raw_coef_abs'}))
		cfg.postBPPrune.scoreMode = 'coef_abs';
	elseif any(strcmpi(cfg.postBPPrune.scoreMode, {'contribution','contribution_abs_mean','mean_abs_contribution','legacy_contribution'}))
		cfg.postBPPrune.scoreMode = 'contribution_abs_mean';
	else
		warning('Unknown postBPPrune.scoreMode=%s. Falling back to coef_abs.', cfg.postBPPrune.scoreMode);
		cfg.postBPPrune.scoreMode = 'coef_abs';
	end
	% Use explicit absThreshold/relThreshold when provided. Legacy
	% contributionAbsThreshold/contributionRelThreshold are accepted as fallbacks
	% only when absThreshold/relThreshold are empty.
	absThr = getfield_default_local(cfg.postBPPrune, 'absThreshold', []);
	relThr = getfield_default_local(cfg.postBPPrune, 'relThreshold', []);
	legacyAbsThr = getfield_default_local(cfg.postBPPrune, 'contributionAbsThreshold', []);
	legacyRelThr = getfield_default_local(cfg.postBPPrune, 'contributionRelThreshold', []);
	if isempty(absThr)
		absThr = legacyAbsThr;
	end
	if isempty(relThr)
		relThr = legacyRelThr;
	end
	if isempty(absThr) || ~isfinite(absThr)
		absThr = 1e-10;
	end
	if isempty(relThr) || ~isfinite(relThr)
		relThr = 0;
	end
	cfg.postBPPrune.absThreshold = absThr;
	cfg.postBPPrune.relThreshold = relThr;
	cfg.postBPPrune.contributionAbsThreshold = cfg.postBPPrune.absThreshold;
	cfg.postBPPrune.contributionRelThreshold = cfg.postBPPrune.relThreshold;
	if isempty(cfg.singleLayerSTLS.threshold) || ~isfinite(cfg.singleLayerSTLS.threshold)
		cfg.singleLayerSTLS.threshold = cfg.postBPPrune.absThreshold;
	end
	if isempty(cfg.singleLayerSTLS.relThreshold) || ~isfinite(cfg.singleLayerSTLS.relThreshold)
		cfg.singleLayerSTLS.relThreshold = cfg.postBPPrune.relThreshold;
	end
	cfg.postBPPrune.minTermsPerXiRow = getfield_default_local(cfg.postBPPrune, 'minTermsPerXiRow', 0);
	cfg.postBPPrune.refineMaxIter = getfield_default_local(cfg.postBPPrune, 'refineMaxIter', 50);
	cfg.postBPPrune.refineMaxFunEvals = getfield_default_local(cfg.postBPPrune, 'refineMaxFunEvals', 5000);
	cfg.postBPPrune.acceptByValidation = getfield_default_local(cfg.postBPPrune, 'acceptByValidation', true);
	cfg.postBPPrune.maxRelValIncrease = getfield_default_local(cfg.postBPPrune, 'maxRelValIncrease', 1e-4);
	cfg.postBPPrune.verbose = getfield_default_local(cfg.postBPPrune, 'verbose', true);
	cfg.objective = getfield_default_local(opts.init, 'objective', struct());
	cfg.objective.normalizeResidual = getfield_default_local(cfg.objective, 'normalizeResidual', true);
	cfg.objective.residualScale = getfield_default_local(cfg.objective, 'residualScale', 'std');
	cfg.objective.invalidPenalty = getfield_default_local(cfg.objective, 'invalidPenalty', 1e6);
	cfg.objective.lambda2 = getfield_default_local(cfg.objective, 'lambda2', 0);
	cfg.objective.lambda1 = getfield_default_local(cfg.objective, 'lambda1', 0);
	cfg.objective.epsSmoothL1 = getfield_default_local(cfg.objective, 'epsSmoothL1', 1e-8);
	if any(strcmpi(cfg.mode, {'bsp_lsq','bsp-lsq','beam_search_pruning','beam-search-pruning','beam_pruning_lsq','beam_pruned_lsq'}))
		cfg.mode = 'bsp_lsq';
		cfg.ga.enable = false;
		cfg.bsp.enable = true;
		cfg.lsq.numStarts = 1;
	elseif any(strcmpi(cfg.mode, {'multistart_lsq_bp','multi_start_lsq_bp','multi-start-lsq-bp','multistart-lsq-bp','multi_start_bp_lsq','multistart_bp_lsq'}))
		cfg.mode = 'multistart_lsq_bp';
		cfg.ga.enable = false;
		cfg.bsp.enable = false;
		cfg.multiStart.enable = true;
		cfg.seed.numCandidates = max(1, cfg.multiStart.numStarts);
		cfg.lsq.numStarts = 1;
	elseif any(strcmpi(cfg.mode, {'direct_lsq_bp','direct-lsq-bp','direct_lsq','direct-bp-lsq','final_lsq_bp'}))
		cfg.mode = 'direct_lsq_bp';
		cfg.ga.enable = false;
		cfg.bsp.enable = false;
		cfg.seed.numCandidates = 1;
		cfg.lsq.numStarts = 1;
	elseif any(strcmpi(cfg.mode, {'skip','skip_stage1','skip-stage1','stage1_skip','stage1-skip','mask_only','mask-only'}))
		% v60c skip route: Stage I coefficient optimization is skipped, but
		% coefficient initialization is still made robust.  If Stage 1 returned a
		% light-BP coefficient seed, prepend it and run final BP from it.  If Stage
		% 0 was bypassed (e.g., strong_prior or weak_prior_lv3), run a cheap
		% fixed-mask multi-start light-BP screening to select a final-BP seed.
		cfg.mode = 'skip';
		cfg.ga.enable = false;
		cfg.bsp.enable = false;
		if cfg.skip.useExternalStage0Seed && cfg.skip.hasExternalSeed
			cfg.multiStart.enable = false;
			cfg.seed.numCandidates = 1;
			cfg.lsq.numStarts = 1;
		elseif cfg.skip.useBaselineScreen
			cfg.multiStart.enable = true;
			cfg.multiStart.numStarts = cfg.skip.numFinalBPStarts;
			cfg.multiStart.screenMaxIter = cfg.skip.screenMaxIter;
			cfg.multiStart.screenMaxFunEvals = cfg.skip.screenMaxFunEvals;
			cfg.multiStart.useParallel = cfg.skip.useParallel;
			cfg.multiStart.autoStartParallelPool = cfg.skip.autoStartParallelPool;
			cfg.seed.numCandidates = max(cfg.seed.numCandidates, cfg.skip.numSeedCandidates);
			cfg.lsq.numStarts = 1;
		else
			cfg.multiStart.enable = false;
			cfg.seed.numCandidates = 1;
			cfg.lsq.numStarts = 1;
		end
	elseif strcmpi(cfg.mode, 'lsq_only')
		cfg.ga.enable = false; cfg.seed.mode = 'zero'; cfg.seed.numCandidates = 1; cfg.lsq.numStarts = 1;
	elseif strcmpi(cfg.mode, 'random_lsq')
		cfg.ga.enable = false;
	elseif strcmpi(cfg.mode, 'ga_lsq')
		cfg.ga.enable = true;
		cfg.seed.numCandidates = cfg.ga.populationSize;
	else
		error('Unknown initialization mode: %s', cfg.mode);
	end
end
function cfg = configure_ga_effort_cfg_local(cfg, nTheta)
%CONFIGURE_GA_EFFORT_CFG_LOCAL Recompute GA budget after sparse pre-screening.
% This mirrors configure_ga_effort.m but works directly on the normalized cfg
% structure, so it can be used inside masked_lsq_initialize after the active
% mask dimension has changed.
	if nargin < 2 || isempty(nTheta) || ~isfinite(nTheta)
		nTheta = 1;
	end
	nTheta = max(1, round(nTheta));
	effort = lower(strtrim(char(getfield_default_local(cfg.ga, 'effort', 'default'))));

	switch effort
		case 'quick'
			targetETPDDefault = 50;
			stallFrac = 0.40;
			minPop = 40;
			minGen = 10;
		case 'default'
			targetETPDDefault = 100;
			stallFrac = 0.35;
			minPop = 60;
			minGen = 15;
		case 'robust'
			targetETPDDefault = 200;
			stallFrac = 0.30;
			minPop = 80;
			minGen = 20;
		otherwise
			targetETPDDefault = 100;
			stallFrac = 0.35;
			minPop = 60;
			minGen = 15;
	end

	targetETPD = getfield_default_local(cfg.ga, 'targetETPD', []);
	if isempty(targetETPD) || ~isfinite(targetETPD) || targetETPD <= 0
		targetETPD = targetETPDDefault;
	end
	genPopRatio = getfield_default_local(cfg.ga, 'generationPopulationRatio', 1/3);
	if ~isfinite(genPopRatio) || genPopRatio <= 0
		genPopRatio = 1/3;
	end

	targetEvalBudget = ceil(targetETPD * nTheta);
	pop = round(sqrt(targetEvalBudget / genPopRatio));
	pop = max(minPop, pop);
	gen = round(genPopRatio * pop);
	gen = max(minGen, gen);

	capLimited = false;
	respectAutoCaps = getfield_default_local(cfg.ga, 'respectAutoCaps', false);
	if respectAutoCaps
		maxPop = getfield_default_local(cfg.ga, 'maxPopulationSizeAuto', Inf);
		maxGen = getfield_default_local(cfg.ga, 'maxGenerationsAuto', Inf);
		maxBudget = getfield_default_local(cfg.ga, 'maxEvalBudgetAuto', Inf);
		if isfinite(maxPop) && maxPop > 0 && pop > maxPop
			pop = maxPop;
			capLimited = true;
		end
		if isfinite(maxGen) && maxGen > 0 && gen > maxGen
			gen = maxGen;
			capLimited = true;
		end
		if isfinite(maxBudget) && maxBudget > 0 && pop * gen > maxBudget
			gen = max(1, floor(maxBudget / max(1, pop)));
			capLimited = true;
		end
	end

	cfg.ga.populationSize = max(1, round(pop));
	cfg.ga.maxGenerations = max(1, round(gen));
	cfg.ga.maxStallGenerations = max(5, ceil(stallFrac * cfg.ga.maxGenerations));
	cfg.ga.autoConfiguredNTheta = nTheta;
	cfg.ga.autoConfiguredEvalBudget = cfg.ga.populationSize * cfg.ga.maxGenerations;
	cfg.ga.effectiveETPD = cfg.ga.autoConfiguredEvalBudget / nTheta;
	cfg.ga.capLimited = capLimited;
end

function f = scalar_objective_local(theta, mask, coefZero, X, Y, arch, normOpt, cfgObjective, scaleY)
	r = residual_vector_local(theta, mask, coefZero, X, Y, arch, normOpt, cfgObjective, scaleY);
	f = mean(r(:).^2);
	if ~isfinite(f)
		f = cfgObjective.invalidPenalty;
	end
end

function [resFcn, resJacFcn, nPadding] = make_lsq_residual_handles_local( ...
	mask, coefZero, X, Y, arch, normOpt, cfgObjective, scaleY)
%MAKE_LSQ_RESIDUAL_HANDLES_LOCAL Preserve bounded trust-region LSQ for m<n.
%
% lsqnonlin's trust-region-reflective algorithm requires at least as many
% residual equations as optimization variables.  Small-sample PhDN cases can
% have fewer data residuals than active coefficients.  MATLAB otherwise emits
% one warning per validation chunk and silently switches to unbounded LM.
% Appending exact-zero residual/Jacobian rows changes neither the objective nor
% its gradient, but keeps the residual count compatible with the bounded solver.
	nVars = count_active_mask(mask);
	baseLength = residual_length_local(Y, zeros(nVars,1), cfgObjective);
	nPadding = max(0, nVars - baseLength);
	resFcn = @(theta) padded_residual_vector_local(theta(:), mask, coefZero, ...
		X, Y, arch, normOpt, cfgObjective, scaleY, nPadding);
	resJacFcn = @(theta) padded_residual_jacobian_local(theta(:), mask, coefZero, ...
		X, Y, arch, normOpt, cfgObjective, scaleY, nPadding);
end

function r = padded_residual_vector_local(theta, mask, coefZero, X, Y, arch, ...
	normOpt, cfgObjective, scaleY, nPadding)
	r = residual_vector_local(theta, mask, coefZero, X, Y, arch, normOpt, cfgObjective, scaleY);
	if nPadding > 0
		r = [r; zeros(nPadding,1)];
	end
end

function [r,J] = padded_residual_jacobian_local(theta, mask, coefZero, X, Y, arch, ...
	normOpt, cfgObjective, scaleY, nPadding)
	[r,J] = residual_jacobian_masked_phdn(theta, mask, coefZero, X, Y, arch, ...
		normOpt, cfgObjective, scaleY);
	if nPadding > 0
		r = [r; zeros(nPadding,1)];
		J = [J; zeros(nPadding,size(J,2))];
	end
end

function r = residual_vector_local(theta, mask, coefZero, X, Y, arch, normOpt, cfgObjective, scaleY)
	try
		Coef = unpack_Coef_M_by_mask(theta, mask, coefZero);
		Ypred = model_forward(X, Coef, arch, normOpt);
		if ~isreal(Ypred) || any(~isfinite(Ypred(:)))
			r = cfgObjective.invalidPenalty * ones(residual_length_local(Y, theta, cfgObjective), 1);
			return;
		end
		err = (Ypred - Y) ./ scaleY;
		if ~isreal(err) || any(~isfinite(err(:)))
			r = cfgObjective.invalidPenalty * ones(residual_length_local(Y, theta, cfgObjective), 1);
			return;
		end
		r = err(:);

		if isfield(cfgObjective, 'lambda2') && cfgObjective.lambda2 > 0
			r = [r; sqrt(cfgObjective.lambda2) * theta(:)];
		end
		if isfield(cfgObjective, 'lambda1') && cfgObjective.lambda1 > 0
			r = [r; sqrt(cfgObjective.lambda1) * (theta(:).^2 + cfgObjective.epsSmoothL1).^(1/4)];
		end

		if any(~isfinite(r))
			r = cfgObjective.invalidPenalty * ones(residual_length_local(Y, theta, cfgObjective), 1);
		end
	catch
		r = cfgObjective.invalidPenalty * ones(residual_length_local(Y, theta, cfgObjective), 1);
	end
end

function mse = eval_mse_local(theta, mask, coefZero, X, Y, arch, normOpt)
	try
		Coef = unpack_Coef_M_by_mask(theta, mask, coefZero);
		Ypred = model_forward(X, Coef, arch, normOpt);
		if ~isreal(Ypred) || any(~isfinite(Ypred(:)))
			mse = Inf;
			return;
		end
		err = Ypred - Y;
		if ~isreal(err) || any(~isfinite(err(:)))
			mse = Inf;
			return;
		end
		mse = mean(err(:).^2);
		if ~isfinite(mse) || mse < 0
			mse = Inf;
		end
	catch
		mse = Inf;
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

function b = expand_bound_local(bIn, n)
	if isscalar(bIn)
		b = bIn * ones(1, n);
	else
		b = reshape(bIn, 1, []);
		if numel(b) ~= n
			error('Bound vector length must be either 1 or nVars.');
		end
	end
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

function A = unique_rows_stable_local(A)
	if isempty(A)
		return;
	end
	[~, ia] = unique(round(A, 14), 'rows', 'stable');
	A = A(ia, :);
end

function print_mask_summary_local(mask, name)
	nTotal = 0;
	fprintf('%s\n', name);
	for j = 1:size(mask, 2)
		for i = 1:j
			if isempty(mask{i, j})
				continue;
			end
			n = nnz(mask{i, j});
			nTotal = nTotal + n;
			if n > 0
				fprintf('  A{%d,%d}: active %d / %d\n', i, j, n, numel(mask{i, j}));
			end
		end
	end
	fprintf('  total active = %d\n', nTotal);
end

function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end



function [thetaOut, trainMSE, valMSE, exitflagOut, outputOut] = run_lsq_refine_once_local( ...
	thetaStart, mask, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, scaleY, maxIter, maxFunEvals)
	nVars = count_active_mask(mask);
	thetaOut = thetaStart(:);
	exitflagOut = NaN;
	outputOut = [];
	if nVars == 0
		trainMSE = Inf;
		valMSE = Inf;
		return;
	end
	if numel(thetaOut) ~= nVars
		% The start can be generated from a coefficient struct, but make sure that
		% it has the right length for the current mask.
		thetaOut = thetaOut(1:min(numel(thetaOut), nVars));
		if numel(thetaOut) < nVars
			thetaOut(end+1:nVars, 1) = 0;
		end
	end
	lb = expand_bound_local(cfg.bounds.lower, nVars);
	ub = expand_bound_local(cfg.bounds.upper, nVars);
	[resTrain, resJacTrain] = make_lsq_residual_handles_local( ...
		mask, coefZero, Xtr, Ytr, arch, normOpt, cfg.objective, scaleY);
	if exist('lsqnonlin', 'file') == 2
		lsqOpts = optimoptions('lsqnonlin', ...
			'Display', cfg.lsq.display, ...
			'MaxIterations', maxIter, ...
			'MaxFunctionEvaluations', maxFunEvals, ...
			'StepTolerance', cfg.lsq.stepTolerance, ...
			'OptimalityTolerance', cfg.lsq.optimalityTolerance);
		useAnalytic = cfg.lsq.useAnalyticJacobian;
		if useAnalytic
			try
				lsqOpts = optimoptions(lsqOpts, 'SpecifyObjectiveGradient', true);
			catch
				try
					lsqOpts = optimoptions(lsqOpts, 'Jacobian', 'on');
				catch
					useAnalytic = false;
				end
			end
		end
		try
			if useAnalytic
				[thetaOut, ~, ~, exitflagOut, outputOut] = lsqnonlin(resJacTrain, thetaOut(:), lb(:), ub(:), lsqOpts);
			else
				[thetaOut, ~, ~, exitflagOut, outputOut] = lsqnonlin(resTrain, thetaOut(:), lb(:), ub(:), lsqOpts);
			end
			thetaOut = thetaOut(:);
		catch ME
			warning('BSP/post-BP LSQ refinement failed; keeping the current point. Reason: %s', ME.message);
		end
	else
		warning('lsqnonlin was not found. BSP/post-BP LSQ refinement keeps the current point.');
	end
	trainMSE = eval_mse_local(thetaOut, mask, coefZero, Xtr, Ytr, arch, normOpt);
	valMSE = eval_mse_local(thetaOut, mask, coefZero, Xval, Yval, arch, normOpt);
end


function [thetaBest, trainBest, valBest, exitflagBest, outputBest, earlyStats] = run_lsq_with_validation_early_stop_local( ...
	thetaStart, resTrain, resJacTrain, useAnalyticJacobian, lsqOptsBase, lb, ub, lsqCfg, ...
	evalTrainFcn, evalValFcn, maxIterBudget, maxFunEvalsBudget, startIndex)
	% Run lsqnonlin either as one ordinary refinement or as validation-gated chunks.
	% The chunked mode mimics neural-network early stopping: after each short
	% nonlinear-LSQ refinement, validation MSE is checked and the best-validation
	% theta is restored at the end.
	earlyCfg = getfield_default_local(lsqCfg, 'earlyStop', struct());
	enableEarly = getfield_default_local(earlyCfg, 'enable', false);

	if ~enableEarly
		try
			if useAnalyticJacobian
				[thetaBest, ~, ~, exitflagBest, outputBest] = lsqnonlin(resJacTrain, thetaStart, lb, ub, lsqOptsBase);
			else
				[thetaBest, ~, ~, exitflagBest, outputBest] = lsqnonlin(resTrain, thetaStart, lb, ub, lsqOptsBase);
			end
			thetaBest = thetaBest(:);
		catch ME
			warning('lsqnonlin failed at start %d; keeping the start point. Reason: %s', startIndex, ME.message);
			thetaBest = thetaStart(:);
			exitflagBest = NaN;
			outputBest = [];
		end
		trainBest = evalTrainFcn(thetaBest);
		valBest = evalValFcn(thetaBest);
		earlyStats = struct('enabled', false, 'numChunks', 1, 'bestChunk', 1, ...
			'bestTrainMSE', trainBest, 'bestValMSE', valBest, 'history', []);
		return;
	end

	chunkMaxIter = max(1, round(getfield_default_local(earlyCfg, 'chunkMaxIter', 50)));
	if isempty(chunkMaxIter) || ~isfinite(chunkMaxIter)
		chunkMaxIter = 50;
	end
	maxIterBudget = max(1, round(maxIterBudget));
	chunkMaxIter = min(chunkMaxIter, maxIterBudget);

	maxChunksDefault = max(1, ceil(maxIterBudget / chunkMaxIter));
	maxChunks = getfield_default_local(earlyCfg, 'maxChunks', []);
	if isempty(maxChunks) || ~isfinite(maxChunks)
		maxChunks = maxChunksDefault;
	else
		maxChunks = max(1, round(maxChunks));
		maxChunks = min(maxChunks, maxChunksDefault);
	end

	chunkMaxFunEvals = getfield_default_local(earlyCfg, 'chunkMaxFunEvals', []);
	if isempty(chunkMaxFunEvals) || ~isfinite(chunkMaxFunEvals)
		chunkMaxFunEvals = max(100, ceil(maxFunEvalsBudget / maxChunks));
	else
		chunkMaxFunEvals = max(100, round(chunkMaxFunEvals));
	end
	chunkMaxFunEvals = min(chunkMaxFunEvals, maxFunEvalsBudget);

	minChunks = max(1, round(getfield_default_local(earlyCfg, 'minChunks', 1)));
	valPatience = max(0, round(getfield_default_local(earlyCfg, 'valPatience', 3)));
	relTol = max(0, getfield_default_local(earlyCfg, 'relImproveTol', 1e-4));
	absTol = max(0, getfield_default_local(earlyCfg, 'absImproveTol', 0));
	restoreBest = getfield_default_local(earlyCfg, 'restoreBestValidation', true);
	stopOnSolverConvergence = getfield_default_local(earlyCfg, 'stopOnSolverConvergence', true);
	stopOnInvalidVal = getfield_default_local(earlyCfg, 'stopOnInvalidVal', true);
	verbose = getfield_default_local(earlyCfg, 'verbose', false);

	thetaCur = thetaStart(:);
	train0 = evalTrainFcn(thetaCur);
	val0 = evalValFcn(thetaCur);
	thetaBest = thetaCur;
	trainBest = train0;
	valBest = val0;
	exitflagBest = NaN;
	outputBest = [];
	bestChunk = 0;
	stallCount = 0;
	history = repmat(struct('chunk', [], 'trainMSE', [], 'valMSE', [], 'exitflag', [], 'iterations', []), maxChunks, 1);

	if verbose
		fprintf('    LSQ early-stop start %d: initial train/val %.6e / %.6e, chunkIter %d, maxChunks %d, patience %d\n', ...
			startIndex, train0, val0, chunkMaxIter, maxChunks, valPatience);
	end

	for kk = 1:maxChunks
		remainingIter = maxIterBudget - (kk - 1) * chunkMaxIter;
		if remainingIter <= 0
			break;
		end
		iterThis = min(chunkMaxIter, remainingIter);
		funEvalRemaining = maxFunEvalsBudget - (kk - 1) * chunkMaxFunEvals;
		if funEvalRemaining <= 0
			break;
		end
		funEvalThis = min(chunkMaxFunEvals, funEvalRemaining);

		lsqOptsChunk = optimoptions(lsqOptsBase, ...
			'MaxIterations', iterThis, ...
			'MaxFunctionEvaluations', funEvalThis);

		try
			if useAnalyticJacobian
				[thetaCur, ~, ~, exitflagCur, outputCur] = lsqnonlin(resJacTrain, thetaCur, lb, ub, lsqOptsChunk);
			else
				[thetaCur, ~, ~, exitflagCur, outputCur] = lsqnonlin(resTrain, thetaCur, lb, ub, lsqOptsChunk);
			end
			thetaCur = thetaCur(:);
		catch ME
			warning('lsqnonlin failed at start %d chunk %d; keeping the current point. Reason: %s', startIndex, kk, ME.message);
			exitflagCur = NaN;
			outputCur = [];
		end

		trainCur = evalTrainFcn(thetaCur);
		valCur = evalValFcn(thetaCur);
		iterDone = getfield_default_local(outputCur, 'iterations', NaN);

		history(kk).chunk = kk;
		history(kk).trainMSE = trainCur;
		history(kk).valMSE = valCur;
		history(kk).exitflag = exitflagCur;
		history(kk).iterations = iterDone;

		% Use a true relative tolerance with respect to the current validation
		% scale.  The old max(1,abs(valBest)) rule made small-MSE problems require
		% an unintended absolute improvement of relTol, which could hide useful
		% final BP refinements after the light multi-start stage.
		oldValBest = valBest;
		improveTol = max(absTol, relTol * max(eps, abs(oldValBest)));
		exactImproved = isfinite(valCur) && (~isfinite(oldValBest) || valCur < oldValBest);
		significantImproved = exactImproved && (~isfinite(oldValBest) || valCur < oldValBest - improveTol);
		if exactImproved
			thetaBest = thetaCur;
			trainBest = trainCur;
			valBest = valCur;
			exitflagBest = exitflagCur;
			outputBest = outputCur;
			bestChunk = kk;
		end
		if significantImproved
			stallCount = 0;
		else
			stallCount = stallCount + 1;
		end

		if verbose
			fprintf('    LSQ early-stop start %d chunk %d: train/val %.6e / %.6e, best val %.6e, stall %d/%d\n', ...
				startIndex, kk, trainCur, valCur, valBest, stallCount, valPatience);
		end

		if kk >= minChunks
			if stopOnInvalidVal && (~isfinite(valCur) || valCur >= getfield_default_local(lsqCfg, 'invalidValThreshold', Inf))
				break;
			end
			if valPatience > 0 && stallCount >= valPatience
				break;
			end
			if stopOnSolverConvergence && isfinite(exitflagCur) && exitflagCur > 0
				break;
			end
		end
	end

	nChunksUsed = find(~arrayfun(@(h) isempty(h.chunk), history), 1, 'last');
	if isempty(nChunksUsed)
		nChunksUsed = 0;
		history = struct([]);
	else
		history = history(1:nChunksUsed);
	end
	if ~restoreBest
		thetaBest = thetaCur;
		trainBest = evalTrainFcn(thetaBest);
		valBest = evalValFcn(thetaBest);
	end
	if isempty(outputBest)
		outputBest = struct('iterations', nChunksUsed * chunkMaxIter);
	end

	earlyStats = struct();
	earlyStats.enabled = true;
	earlyStats.numChunks = nChunksUsed;
	earlyStats.bestChunk = bestChunk;
	earlyStats.bestTrainMSE = trainBest;
	earlyStats.bestValMSE = valBest;
	earlyStats.initialTrainMSE = train0;
	earlyStats.initialValMSE = val0;
	earlyStats.chunkMaxIter = chunkMaxIter;
	earlyStats.maxChunks = maxChunks;
	earlyStats.valPatience = valPatience;
	earlyStats.history = history;
end


function stats = empty_postbp_iter_stats_local()
	stats = struct( ...
		'iteration', {}, ...
		'activeBefore', {}, ...
		'activeAfterPrune', {}, ...
		'nPruned', {}, ...
		'trainBefore', {}, ...
		'valBefore', {}, ...
		'pruneInfo', {}, ...
		'accepted', {}, ...
		'reason', {}, ...
		'trainAfter', {}, ...
		'valAfter', {}, ...
		'exitflag', {}, ...
		'output', {});
end

function iterStats = make_postbp_iter_stats_local(iteration, activeBefore, activeAfterPrune, nPruned, trainBefore, valBefore, pruneInfo)
	iterStats = struct();
	iterStats.iteration = iteration;
	iterStats.activeBefore = activeBefore;
	iterStats.activeAfterPrune = activeAfterPrune;
	iterStats.nPruned = nPruned;
	iterStats.trainBefore = trainBefore;
	iterStats.valBefore = valBefore;
	iterStats.pruneInfo = pruneInfo;
	iterStats.accepted = false;
	iterStats.reason = '';
	iterStats.trainAfter = NaN;
	iterStats.valAfter = NaN;
	iterStats.exitflag = NaN;
	iterStats.output = [];
end


function [thetaOut, CoefOut, maskOut, stats] = post_bp_prune_refine_local(thetaIn, maskIn, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, scaleY)
	stats = struct();
	stats.applied = false;
	% Use a fixed-field empty struct array. MATLAB cannot append structs
	% with different top-level fields, so every post-BP prune iteration
	% record must share this exact schema.
	stats.iterations = empty_postbp_iter_stats_local();
	stats.elapsedTime = 0;
	stats.originalActive = count_active_mask(maskIn);
	stats.finalActive = stats.originalActive;
	timer = tic;

	thetaOut = thetaIn(:);
	maskOut = maskIn;
	CoefOut = unpack_Coef_M_by_mask(thetaOut, maskOut, coefZero);

	nIter = max(0, round(cfg.postBPPrune.numIterations));
	if nIter <= 0 || count_active_mask(maskOut) == 0
		stats.elapsedTime = toc(timer);
		return;
	end

	for it = 1:nIter
		activeBefore = count_active_mask(maskOut);
		trainBefore = eval_mse_local(thetaOut, maskOut, coefZero, Xtr, Ytr, arch, normOpt);
		valBefore = eval_mse_local(thetaOut, maskOut, coefZero, Xval, Yval, arch, normOpt);
		[maskCand, pruneInfo] = prune_mask_by_postbp_score_local(CoefOut, maskOut, Xtr, arch, normOpt, cfg.postBPPrune);
		activeAfterPrune = count_active_mask(maskCand);

		iterStats = make_postbp_iter_stats_local(it, activeBefore, activeAfterPrune, ...
			activeBefore - activeAfterPrune, trainBefore, valBefore, pruneInfo);

		if activeAfterPrune >= activeBefore
			iterStats.reason = 'no_terms_below_score_threshold';
			stats.iterations(end + 1) = iterStats; %#ok<AGROW>
			if cfg.postBPPrune.verbose
				fprintf('  post-BP prune iter %d/%d: active %d -> %d, no terms pruned [score=%s, thr %.3e, rel %.3e, belowThr %d]\n', ...
					it, nIter, activeBefore, activeAfterPrune, pruneInfo.scoreMode, pruneInfo.threshold, cfg.postBPPrune.relThreshold, pruneInfo.nBelowThresholdBeforeRowFloor);
			end
			break;
		end

		% Repack current coefficients on the pruned mask and run a fast BP/LSQ refine.
		thetaCand0 = pack_Coef_M_by_mask(CoefOut, maskCand);
		[thetaCand, trainCand, valCand, exitflagCand, outputCand] = run_lsq_refine_once_local( ...
			thetaCand0, maskCand, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, scaleY, ...
			cfg.postBPPrune.refineMaxIter, cfg.postBPPrune.refineMaxFunEvals);

		accept = true;
		if cfg.postBPPrune.acceptByValidation
			accept = isfinite(valCand) && valCand <= valBefore * (1 + cfg.postBPPrune.maxRelValIncrease);
		end
		if accept
			thetaOut = thetaCand(:);
			maskOut = maskCand;
			CoefOut = unpack_Coef_M_by_mask(thetaOut, maskOut, coefZero);
			iterStats.accepted = true;
			iterStats.reason = 'accepted';
		else
			iterStats.reason = 'validation_rejected';
		end
		iterStats.trainAfter = trainCand;
		iterStats.valAfter = valCand;
		iterStats.exitflag = exitflagCand;
		iterStats.output = outputCand;
		stats.iterations(end + 1) = iterStats; %#ok<AGROW>

		if cfg.postBPPrune.verbose
			fprintf('  post-BP prune iter %d/%d: active %d -> %d, pruned %d, train/val %.3e/%.3e -> %.3e/%.3e, accepted %d\n', ...
				it, nIter, activeBefore, activeAfterPrune, activeBefore - activeAfterPrune, ...
				trainBefore, valBefore, trainCand, valCand, iterStats.accepted);
		end

		if ~iterStats.accepted
			break;
		end
	end

	stats.applied = true;
	stats.finalActive = count_active_mask(maskOut);
	stats.elapsedTime = toc(timer);
end

function [maskOut, info] = prune_mask_by_postbp_score_local(Coef, maskIn, X, arch, normOpt, pruneCfg)
	maskOut = maskIn;
	info = struct();
	info.scoreMode = getfield_default_local(pruneCfg, 'scoreMode', 'coef_abs');
	info.threshold = NaN;
	info.nRawPruned = 0;
	info.nProtected = 0;
	info.nBelowThresholdBeforeRowFloor = 0;
	info.scoreMin = NaN;
	info.scoreMedian = NaN;
	info.scoreMax = NaN;

	if isempty(maskIn) || count_active_mask(maskIn) == 0
		return;
	end

	scoreMode = lower(strtrim(char(getfield_default_local(pruneCfg, 'scoreMode', 'coef_abs'))));
	if any(strcmpi(scoreMode, {'coef','coef_abs','coefficient','coefficient_abs','abs_coef','raw_coef_abs'}))
		scoreMode = 'coef_abs';
	elseif any(strcmpi(scoreMode, {'contribution','contribution_abs_mean','mean_abs_contribution','legacy_contribution'}))
		scoreMode = 'contribution_abs_mean';
	else
		scoreMode = 'coef_abs';
	end
	info.scoreMode = scoreMode;

	cache = [];
	if strcmpi(scoreMode, 'contribution_abs_mean')
		try
			[~, cache] = model_forward(X, Coef, arch, normOpt);
		catch
			return;
		end
	end

	allScores = [];
	for ell = 1:size(maskIn, 2)
		for src = 1:ell
			M = maskIn{src, ell};
			if isempty(M) || ~any(M(:))
				continue;
			end
			C = Coef{src, ell};
			active = find(M);
			[rowIdx, colIdx] = ind2sub(size(M), active);
			for k = 1:numel(active)
				r = rowIdx(k); c = colIdx(k);
				s = postbp_term_score_local(C, cache, src, ell, r, c, scoreMode);
				if isfinite(s)
					allScores(end + 1, 1) = s; %#ok<AGROW>
				end
			end
		end
	end

	if isempty(allScores)
		return;
	end
	info.scoreMin = min(allScores);
	info.scoreMedian = median(allScores);
	info.scoreMax = max(allScores);
	threshold = max(pruneCfg.absThreshold, pruneCfg.relThreshold * info.scoreMax);
	info.threshold = threshold;
	minPerRow = max(0, round(getfield_default_local(pruneCfg, 'minTermsPerXiRow', 0)));

	for ell = 1:size(maskIn, 2)
		for src = 1:ell
			M = maskIn{src, ell};
			if isempty(M) || ~any(M(:))
				continue;
			end
			C = Coef{src, ell};
			scoreMat = inf(size(M));
			active = find(M);
			[rowIdx, colIdx] = ind2sub(size(M), active);
			for k = 1:numel(active)
				r = rowIdx(k); c = colIdx(k);
				scoreMat(r, c) = postbp_term_score_local(C, cache, src, ell, r, c, scoreMode);
			end
			remove = M & (scoreMat <= threshold);
			info.nBelowThresholdBeforeRowFloor = info.nBelowThresholdBeforeRowFloor + nnz(remove);
			% Optional row floor.
			if minPerRow > 0
				for r = 1:size(M, 1)
					idxRow = find(M(r, :));
					if isempty(idxRow), continue; end
					willKeep = idxRow(~remove(r, idxRow));
					if numel(willKeep) < minPerRow
						[~, ord] = sort(scoreMat(r, idxRow), 'descend');
						keepNeed = idxRow(ord(1:min(minPerRow, numel(idxRow))));
						remove(r, keepNeed) = false;
					end
				end
			end
			info.nRawPruned = info.nRawPruned + nnz(remove);
			maskOut{src, ell} = M & ~remove;
		end
	end
end

function s = postbp_term_score_local(C, cache, src, ell, r, c, scoreMode)
	if r > size(C, 1) || c > size(C, 2)
		s = Inf;
		return;
	end
	coefAbs = abs(C(r, c));
	if strcmpi(scoreMode, 'coef_abs')
		s = coefAbs;
		return;
	end
	if strcmpi(scoreMode, 'contribution_abs_mean')
		% Legacy optional mode retained for reproducibility.
		try
			Phi = cache.branch{src, ell}.Phi;
			if c <= size(Phi, 1)
				s = mean(abs(C(r, c) .* Phi(c, :)));
			else
				s = Inf;
			end
		catch
			s = Inf;
		end
		return;
	end
	s = coefAbs;
end

function [thetaOut, maskOut, CoefOut, stats] = bsp_lsq_init_local( ...
	thetaStart, maskIn, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg)
%BSP_LSQ_INIT_LOCAL compact-mask route Hierarchical BSP with BP-LSQ fast fitting.
%
% Stage 1 deletes one whole branch/block A{src,ell} at a time.
% Stage 2 inherits the best branch-level structure and deletes one dictionary
% basis column from one surviving branch/block at a time.
% In both stages, candidate structures are evaluated by lightweight nonlinear
% LSQ/BP and validation MSE.  Candidate evaluations are parallelized when a
% parallel pool is available or can be started.
	stats = struct();
	stats.applied = false;
	stats.history = struct([]);
	stats.branch = struct('applied', false);
	stats.basis = struct('applied', false);
	stats.elapsedTime = 0;
	stats.originalActive = count_active_mask(maskIn);
	stats.finalActive = stats.originalActive;
	timer = tic;

	mask0 = maskIn;
	Coef0 = unpack_Coef_M_by_mask(thetaStart(:), mask0, coefZero);
	Coef0 = zero_coefficients_outside_mask_local(Coef0, mask0);
	theta0 = pack_Coef_M_by_mask(Coef0, mask0);

	cfgB = cfg.bsp;
	if count_active_mask(mask0) == 0 || (~getfield_default_local(cfgB, 'branchEnable', true) && ~getfield_default_local(cfgB, 'basisEnable', true))
		thetaOut = theta0;
		maskOut = mask0;
		CoefOut = Coef0;
		stats.elapsedTime = toc(timer);
		return;
	end

	if cfgB.verbose
		fprintf('\n========================================\n');
		fprintf('compact-mask route Hierarchical Beam Search Pruning with BP-LSQ refinement (HBSP-LSQ)\n');
		fprintf('========================================\n');
		fprintf('  initial active coefficients       = %d\n', count_active_mask(mask0));
		fprintf('  initial active branches           = %d\n', count_active_branches_local(mask0));
		fprintf('  initial active dictionary bases   = %d\n', count_active_dictionary_bases_local(mask0));
		fprintf('  parallel candidate evaluation     = %d\n', getfield_default_local(cfgB, 'useParallel', false));
		fprintf('  branch pruning enabled            = %d, mode %s, beam %d, rounds %s, fast BP %d/%d\n', ...
			getfield_default_local(cfgB, 'branchEnable', true), getfield_default_local(cfgB, 'branchMode', 'saliency_greedy'), ...
			getfield_default_local(cfgB, 'branchBeamWidth', cfgB.beamWidth), ...
			format_count_for_log_local(getfield_default_local(cfgB, 'branchMaxRounds', Inf)), ...
			getfield_default_local(cfgB, 'branchFastMaxIter', cfgB.fastMaxIter), ...
			getfield_default_local(cfgB, 'branchFastMaxFunEvals', cfgB.fastMaxFunEvals));
		fprintf('  basis BSP enabled                 = %d, beam %d, rounds %s, fast BP %d/%d\n', ...
			getfield_default_local(cfgB, 'basisEnable', true), cfgB.beamWidth, format_count_for_log_local(cfgB.maxRounds), ...
			cfgB.fastMaxIter, cfgB.fastMaxFunEvals);
		if cfgB.printBranchMasks
			print_bsp_branch_dictionary_masks_local(mask0, arch, 'Initial HBSP branch dictionary masks');
		end
	end

	% Fit the full uniform compact support once.  Both pruning stages inherit
	% this coefficient state rather than evaluating raw random/scale-aware seeds.
	[thetaFit0, train0, val0, exit0, out0] = run_bsp_fast_fit_local( ...
		theta0, mask0, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, cfgB);
	CoefFit0 = unpack_Coef_M_by_mask(thetaFit0(:), mask0, coefZero);
	score0 = val0 + cfgB.sizePenalty * count_active_dictionary_bases_local(mask0);
	state0 = make_bsp_state_local(mask0, thetaFit0, CoefFit0, train0, val0, score0, 0, 'full_support', exit0, out0);
	current = state0;

	if getfield_default_local(cfgB, 'branchEnable', true)
		branchMode = lower(strtrim(char(getfield_default_local(cfgB, 'branchMode', 'saliency_greedy'))));
		if strcmpi(branchMode, 'saliency_greedy')
			[current, branchStats] = branch_saliency_greedy_stage_local(current, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg);
		else
			[current, branchStats] = branch_bsp_stage_local(current, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg);
		end
		stats.branch = branchStats;
	else
		branchStats = struct('applied', false, 'reason', 'disabled');
		stats.branch = branchStats;
	end

	if getfield_default_local(cfgB, 'basisEnable', true)
		[current, basisStats] = basis_bsp_stage_local(current, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg);
		stats.basis = basisStats;
	else
		basisStats = struct('applied', false, 'reason', 'disabled');
		stats.basis = basisStats;
	end

	thetaOut = current.theta(:);
	maskOut = current.mask;
	CoefOut = current.Coef;
	stats.applied = true;
	stats.finalActive = count_active_mask(maskOut);
	stats.finalDictionaryBases = count_active_dictionary_bases_local(maskOut);
	stats.finalBranches = count_active_branches_local(maskOut);
	stats.originalDictionaryBases = count_active_dictionary_bases_local(mask0);
	stats.originalBranches = count_active_branches_local(mask0);
	stats.bestTrainMSE = current.trainMSE;
	stats.bestValMSE = current.valMSE;
	stats.bestScore = current.score;
	stats.roundsUsed = getfield_default_local(branchStats, 'roundsUsed', 0) + getfield_default_local(basisStats, 'roundsUsed', 0);
	stats.bestReason = current.reason;
	stats.initialMask = mask0;
	stats.finalMask = maskOut;
	stats.elapsedTime = toc(timer);

	if cfgB.verbose
		fprintf('\n  HBSP-LSQ active                  = %d -> %d\n', stats.originalActive, stats.finalActive);
		fprintf('  HBSP-LSQ branches                = %d -> %d\n', stats.originalBranches, stats.finalBranches);
		fprintf('  HBSP-LSQ dictionary bases        = %d -> %d\n', stats.originalDictionaryBases, stats.finalDictionaryBases);
		fprintf('  HBSP-LSQ train/val               = %.6e / %.6e\n', stats.bestTrainMSE, stats.bestValMSE);
		fprintf('  HBSP-LSQ time                    = %.3f s\n', stats.elapsedTime);
		fprintf('  HBSP-LSQ selected support reason = %s\n', stats.bestReason);
		if cfgB.printBranchMasks
			print_bsp_branch_dictionary_masks_local(maskOut, arch, 'Selected HBSP branch dictionary masks');
		end
	end
end


function [best, stats] = branch_saliency_greedy_stage_local(startState, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg)
%BRANCH_SALIENCY_GREEDY_STAGE_LOCAL compact-mask route branch pruning by output-impact saliency.
%
% This stage is intentionally simpler than validation-beam branch BSP.
% In each round, the current structure is first lightly refined by BP-LSQ.
% Then each active branch is scored by a no-refit output-impact saliency:
% deleting one active dictionary column in that branch and measuring the
% output change.  The branch score is the mean of the largest TopK column
% scores.  The least significant legal branch is pruned, followed by a short
% BP-LSQ refinement.  The deletion is accepted only if validation error does
% not increase beyond the configured tolerance.
	cfgB = cfg.bsp;
	stats = struct();
	stats.applied = true;
	stats.mode = 'saliency_greedy';
	stats.history = struct([]);
	stats.originalActive = count_active_mask(startState.mask);
	stats.originalBranches = count_active_branches_local(startState.mask);
	stats.finalActive = stats.originalActive;
	stats.finalBranches = stats.originalBranches;
	timer = tic;

	maxRounds = getfield_default_local(cfgB, 'branchMaxRounds', Inf);
	if isempty(maxRounds) || ~isfinite(maxRounds)
		maxRounds = max(0, count_active_branches_local(startState.mask) - max(0, round(getfield_default_local(cfgB, 'branchMinActiveBranches', 1))));
	end
	maxRounds = max(0, round(maxRounds));

	cfgFast = cfgB;
	cfgFast.fastMaxIter = getfield_default_local(cfgB, 'branchFastMaxIter', cfgB.fastMaxIter);
	cfgFast.fastMaxFunEvals = getfield_default_local(cfgB, 'branchFastMaxFunEvals', cfgB.fastMaxFunEvals);

	current = startState;
	best = startState;
	roundCount = 0;

	if cfgB.verbose
		fprintf('\n  Branch saliency greedy pruning stage\n');
		fprintf('  ------------------------------------\n');
		fprintf('    start branches/coefs/dict bases = %d / %d / %d\n', ...
			count_active_branches_local(startState.mask), count_active_mask(startState.mask), count_active_dictionary_bases_local(startState.mask));
		fprintf('    max rounds / topK              = %s / %d\n', ...
			format_count_for_log_local(maxRounds), getfield_default_local(cfgB, 'branchSaliencyTopK', 3));
		fprintf('    score mode / layer norm        = %s / %d\n', ...
			getfield_default_local(cfgB, 'branchSaliencyScoreMode', 'topk_mean_output_delta'), ...
			getfield_default_local(cfgB, 'branchSaliencyUseLayerNorm', false));
		fprintf('    accept by validation / max inc = %d / %.3e\n', ...
			getfield_default_local(cfgB, 'branchSaliencyAcceptByValidation', true), ...
			getfield_default_local(cfgB, 'branchSaliencyMaxRelValIncrease', 0));
	end

	for rr = 1:maxRounds
		roundCount = rr;

		% Re-polish the current parameter state before saliency evaluation.
		trainBefore = current.trainMSE;
		valBefore = current.valMSE;
		if getfield_default_local(cfgB, 'branchSaliencyRefineBeforeScoring', true)
			[thetaRef, trainRef, valRef, exitRef, outRef] = run_bsp_fast_fit_local( ...
				current.theta, current.mask, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, cfgFast);
			CoefRef = unpack_Coef_M_by_mask(thetaRef(:), current.mask, coefZero);
			current = make_bsp_state_local(current.mask, thetaRef(:), CoefRef, trainRef, valRef, ...
				valRef + cfgB.sizePenalty * count_active_dictionary_bases_local(current.mask), rr, current.reason, exitRef, outRef);
		end

		if cfgB.verbose
			fprintf('    Saliency round %d: before-refine train/val %.3e / %.3e, current train/val %.3e / %.3e\n', ...
				rr, trainBefore, valBefore, current.trainMSE, current.valMSE);
		end

		sal = compute_branch_saliency_scores_local(current.Coef, current.mask, Xval, arch, normOpt, cfgB);
		if isempty(sal)
			if cfgB.verbose
				fprintf('    Saliency round %d: no active branch saliency scores; stop.\n', rr);
			end
			break;
		end
		if cfgB.verbose
			print_branch_saliency_table_local(sal, rr);
		end

		% Lowest saliency among legal branch deletions is pruned.
		scoreVec = [sal.rankScore];
		[~, ord] = sort(scoreVec, 'ascend');
		selected = [];
		for kk = ord(:).'
			if sal(kk).legal && isfinite(sal(kk).rankScore)
				selected = sal(kk);
				break;
			end
		end
		if isempty(selected)
			if cfgB.verbose
				fprintf('    Saliency round %d: no legal finite-saliency branch deletion candidate; stop.\n', rr);
			end
			break;
		end

		move = struct('src', selected.src, 'ell', selected.ell);
		maskCand = apply_branch_deletion_local(current.mask, move);
		thetaStartCand = pack_Coef_M_by_mask(current.Coef, maskCand);
		[thetaCand, trainCand, valCand, exitCand, outCand] = run_bsp_fast_fit_local( ...
			thetaStartCand, maskCand, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, cfgFast);
		CoefCand = unpack_Coef_M_by_mask(thetaCand(:), maskCand, coefZero);
		scoreCand = valCand + cfgB.sizePenalty * count_active_dictionary_bases_local(maskCand);

		acceptByVal = getfield_default_local(cfgB, 'branchSaliencyAcceptByValidation', true);
		maxRelInc = getfield_default_local(cfgB, 'branchSaliencyMaxRelValIncrease', 0);
		accepted = true;
		if acceptByVal
			accepted = isfinite(valCand) && isfinite(current.valMSE) && (valCand <= current.valMSE * (1 + maxRelInc));
		end

		stats.history(rr).round = rr;
		stats.history(rr).trainBeforeRefine = trainBefore;
		stats.history(rr).valBeforeRefine = valBefore;
		stats.history(rr).trainCurrent = current.trainMSE;
		stats.history(rr).valCurrent = current.valMSE;
		stats.history(rr).saliency = sal;
		stats.history(rr).selectedBranch = sprintf('A{%d,%d}', selected.src, selected.ell);
		stats.history(rr).selectedRankScore = selected.rankScore;
		stats.history(rr).selectedTopKMean = selected.topKMean;
		stats.history(rr).selectedFullDrop = selected.fullDrop;
		stats.history(rr).trainAfterPrune = trainCand;
		stats.history(rr).valAfterPrune = valCand;
		stats.history(rr).accepted = accepted;

		if cfgB.verbose
			fprintf('    Saliency round %d selected prune branch A{%d,%d}: rankScore %.3e, topK %.3e, fullDrop %.3e\n', ...
				rr, selected.src, selected.ell, selected.rankScore, selected.topKMean, selected.fullDrop);
			fprintf('    Saliency round %d prune+BP train/val %.3e / %.3e, accepted %d\n', ...
				rr, trainCand, valCand, accepted);
		end

		if ~accepted
			if cfgB.verbose
				fprintf('    Branch saliency stop: validation increased after pruning A{%d,%d}; rollback and stop.\n', selected.src, selected.ell);
			end
			break;
		end

		current = make_bsp_state_local(maskCand, thetaCand(:), CoefCand, trainCand, valCand, scoreCand, rr, ...
			sprintf('saliency drop branch A{%d,%d}', selected.src, selected.ell), exitCand, outCand);
		best = current;
	end

	stats.finalActive = count_active_mask(best.mask);
	stats.finalBranches = count_active_branches_local(best.mask);
	stats.finalDictionaryBases = count_active_dictionary_bases_local(best.mask);
	stats.bestTrainMSE = best.trainMSE;
	stats.bestValMSE = best.valMSE;
	stats.bestScore = best.score;
	stats.roundsUsed = roundCount;
	stats.bestReason = best.reason;
	stats.elapsedTime = toc(timer);

	if cfgB.verbose
		fprintf('    Branch saliency selected branches = %d -> %d\n', stats.originalBranches, stats.finalBranches);
		fprintf('    Branch saliency selected active   = %d -> %d\n', stats.originalActive, stats.finalActive);
		fprintf('    Branch saliency train/val         = %.6e / %.6e\n', stats.bestTrainMSE, stats.bestValMSE);
		fprintf('    Branch saliency time              = %.3f s\n', stats.elapsedTime);
		if cfgB.printBranchMasks
			print_bsp_branch_dictionary_masks_local(best.mask, arch, 'After branch saliency pruning masks');
		end
	end
end

function [best, stats] = branch_bsp_stage_local(startState, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg)
	cfgB = cfg.bsp;
	stats = struct();
	stats.applied = true;
	stats.history = struct([]);
	stats.originalActive = count_active_mask(startState.mask);
	stats.originalBranches = count_active_branches_local(startState.mask);
	stats.finalActive = stats.originalActive;
	stats.finalBranches = stats.originalBranches;
	timer = tic;

	beamWidth = max(1, round(getfield_default_local(cfgB, 'branchBeamWidth', cfgB.beamWidth)));
	maxRounds = getfield_default_local(cfgB, 'branchMaxRounds', Inf);
	if isempty(maxRounds) || ~isfinite(maxRounds)
		maxRounds = count_active_branches_local(startState.mask);
	end
	maxRounds = max(0, round(maxRounds));
	patience = max(0, round(getfield_default_local(cfgB, 'branchPatience', cfgB.patience)));
	keepParents = getfield_default_local(cfgB, 'branchKeepParentsInBeam', cfgB.keepParentsInBeam);

	beam = startState;
	best = startState;
	noImprove = 0;
	roundCount = 0;

	if cfgB.verbose
		fprintf('\n  Branch-level BSP stage\n');
		fprintf('  ----------------------\n');
		fprintf('    start branches/coefs/dict bases = %d / %d / %d\n', ...
			count_active_branches_local(startState.mask), count_active_mask(startState.mask), count_active_dictionary_bases_local(startState.mask));
		fprintf('    beam width / max rounds         = %d / %s\n', beamWidth, format_count_for_log_local(maxRounds));
	end

	for rr = 1:maxRounds
		roundCount = rr;
		[maskList, thetaList, reasonList] = collect_branch_deletion_candidates_local(beam, cfgB);
		if isempty(maskList)
			if cfgB.verbose
				fprintf('    Branch BSP round %d: no legal branch deletion candidates; stop.\n', rr);
			end
			break;
		end
		cand = evaluate_bsp_candidate_list_local(maskList, thetaList, reasonList, rr, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, cfgB, 'branch');
		if isempty(cand)
			if cfgB.verbose
				fprintf('    Branch BSP round %d: all candidate evaluations failed; stop.\n', rr);
			end
			break;
		end

		[~, ordCand] = sort([cand.score], 'ascend');
		cand = cand(ordCand);
		bestCand = cand(1);
		% compact-mask route fix2: use the most direct improvement rule for BSP.
		% A candidate is improved if and only if its score is lower than
		% the current reference score. No absolute/relative tolerance is used.
		isImproved = isfinite(bestCand.score) && (~isfinite(best.score) || bestCand.score < best.score);
		if isImproved
			best = bestCand;
			noImprove = 0;
		else
			noImprove = noImprove + 1;
		end

		pool = cand;
		if keepParents
			pool = [beam(:); cand(:)]; %#ok<AGROW>
		end
		[~, ordPool] = sort([pool.score], 'ascend');
		pool = pool(ordPool);
		beam = unique_bsp_states_local(pool, beamWidth);

		stats.history(rr).round = rr;
		stats.history(rr).numCandidates = numel(cand);
		stats.history(rr).bestCandidateValMSE = bestCand.valMSE;
		stats.history(rr).bestCandidateScore = bestCand.score;
		stats.history(rr).bestOverallValMSE = best.valMSE;
		stats.history(rr).bestOverallScore = best.score;
		stats.history(rr).bestActive = count_active_mask(best.mask);
		stats.history(rr).bestBranches = count_active_branches_local(best.mask);
		stats.history(rr).bestDictBases = count_active_dictionary_bases_local(best.mask);
		stats.history(rr).improved = isImproved;

		if cfgB.verbose
			fprintf('    Branch BSP round %d: candidates %d, branches %d -> %d, active %d -> %d, best cand val %.3e, best overall val %.3e, improved %d\n', ...
				rr, numel(cand), count_active_branches_local(startState.mask), count_active_branches_local(best.mask), ...
				count_active_mask(startState.mask), count_active_mask(best.mask), bestCand.valMSE, best.valMSE, isImproved);
			fprintf('      best candidate move           : %s\n', bestCand.reason);
			fprintf('      best overall support reason   : %s\n', best.reason);
		end

		if ~isImproved && noImprove > patience
			if cfgB.verbose
				fprintf('    Branch BSP stop: no validation-score improvement after round %d (patience %d).\n', rr, patience);
			end
			break;
		end
	end

	stats.finalActive = count_active_mask(best.mask);
	stats.finalBranches = count_active_branches_local(best.mask);
	stats.finalDictionaryBases = count_active_dictionary_bases_local(best.mask);
	stats.bestTrainMSE = best.trainMSE;
	stats.bestValMSE = best.valMSE;
	stats.bestScore = best.score;
	stats.roundsUsed = roundCount;
	stats.bestReason = best.reason;
	stats.elapsedTime = toc(timer);

	if cfgB.verbose
		fprintf('    Branch BSP selected branches    = %d -> %d\n', stats.originalBranches, stats.finalBranches);
		fprintf('    Branch BSP selected active      = %d -> %d\n', stats.originalActive, stats.finalActive);
		fprintf('    Branch BSP train/val            = %.6e / %.6e\n', stats.bestTrainMSE, stats.bestValMSE);
		fprintf('    Branch BSP time                 = %.3f s\n', stats.elapsedTime);
		if cfgB.printBranchMasks
			print_bsp_branch_dictionary_masks_local(best.mask, arch, 'After branch-level BSP masks');
		end
	end
end

function [best, stats] = basis_bsp_stage_local(startState, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg)
	cfgB = cfg.bsp;
	stats = struct();
	stats.applied = true;
	stats.history = struct([]);
	stats.originalActive = count_active_mask(startState.mask);
	stats.originalDictionaryBases = count_active_dictionary_bases_local(startState.mask);
	stats.finalActive = stats.originalActive;
	stats.finalDictionaryBases = stats.originalDictionaryBases;
	timer = tic;

	beamWidth = max(1, round(cfgB.beamWidth));
	maxRounds = cfgB.maxRounds;
	if isempty(maxRounds) || ~isfinite(maxRounds)
		maxRounds = count_active_dictionary_bases_local(startState.mask);
	end
	maxRounds = max(0, round(maxRounds));
	patience = max(0, round(cfgB.patience));

	beam = startState;
	best = startState;
	noImprove = 0;
	roundCount = 0;

	if cfgB.verbose
		fprintf('\n  Dictionary-basis-level BSP stage\n');
		fprintf('  --------------------------------\n');
		fprintf('    start branches/coefs/dict bases = %d / %d / %d\n', ...
			count_active_branches_local(startState.mask), count_active_mask(startState.mask), count_active_dictionary_bases_local(startState.mask));
		fprintf('    beam width / max rounds         = %d / %s\n', beamWidth, format_count_for_log_local(maxRounds));
		fprintf('    min terms per Xi block          = %d\n', cfgB.minTermsPerXiBlock);
	end

	for rr = 1:maxRounds
		roundCount = rr;
		[maskList, thetaList, reasonList] = collect_basis_deletion_candidates_local(beam, cfgB);
		if isempty(maskList)
			if cfgB.verbose
				fprintf('    Basis BSP round %d: no legal one-basis deletion candidates; stop.\n', rr);
			end
			break;
		end
		cand = evaluate_bsp_candidate_list_local(maskList, thetaList, reasonList, rr, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, cfgB, 'basis');
		if isempty(cand)
			if cfgB.verbose
				fprintf('    Basis BSP round %d: all candidate evaluations failed; stop.\n', rr);
			end
			break;
		end

		[~, ordCand] = sort([cand.score], 'ascend');
		cand = cand(ordCand);
		bestCand = cand(1);
		% compact-mask route fix2: use the most direct improvement rule for BSP.
		% A candidate is improved if and only if its score is lower than
		% the current reference score. No absolute/relative tolerance is used.
		isImproved = isfinite(bestCand.score) && (~isfinite(best.score) || bestCand.score < best.score);
		if isImproved
			best = bestCand;
			noImprove = 0;
		else
			noImprove = noImprove + 1;
		end

		pool = cand;
		if cfgB.keepParentsInBeam
			pool = [beam(:); cand(:)]; %#ok<AGROW>
		end
		[~, ordPool] = sort([pool.score], 'ascend');
		pool = pool(ordPool);
		beam = unique_bsp_states_local(pool, beamWidth);

		stats.history(rr).round = rr;
		stats.history(rr).numCandidates = numel(cand);
		stats.history(rr).bestCandidateValMSE = bestCand.valMSE;
		stats.history(rr).bestCandidateScore = bestCand.score;
		stats.history(rr).bestOverallValMSE = best.valMSE;
		stats.history(rr).bestOverallScore = best.score;
		stats.history(rr).bestActive = count_active_mask(best.mask);
		stats.history(rr).bestDictBases = count_active_dictionary_bases_local(best.mask);
		stats.history(rr).improved = isImproved;

		if cfgB.verbose
			fprintf('    Basis BSP round %d: candidates %d, active %d -> %d, dict bases %d -> %d, best cand val %.3e, best overall val %.3e, improved %d\n', ...
				rr, numel(cand), count_active_mask(startState.mask), count_active_mask(best.mask), ...
				count_active_dictionary_bases_local(startState.mask), count_active_dictionary_bases_local(best.mask), ...
				bestCand.valMSE, best.valMSE, isImproved);
			fprintf('      best candidate move           : %s\n', bestCand.reason);
			fprintf('      best overall support reason   : %s\n', best.reason);
		end

		if ~isImproved && noImprove > patience
			if cfgB.verbose
				fprintf('    Basis BSP stop: no validation-score improvement after round %d (patience %d).\n', rr, patience);
			end
			break;
		end
	end

	stats.finalActive = count_active_mask(best.mask);
	stats.finalDictionaryBases = count_active_dictionary_bases_local(best.mask);
	stats.bestTrainMSE = best.trainMSE;
	stats.bestValMSE = best.valMSE;
	stats.bestScore = best.score;
	stats.roundsUsed = roundCount;
	stats.bestReason = best.reason;
	stats.elapsedTime = toc(timer);

	if cfgB.verbose
		fprintf('    Basis BSP selected active       = %d -> %d\n', stats.originalActive, stats.finalActive);
		fprintf('    Basis BSP dictionary bases      = %d -> %d\n', stats.originalDictionaryBases, stats.finalDictionaryBases);
		fprintf('    Basis BSP train/val             = %.6e / %.6e\n', stats.bestTrainMSE, stats.bestValMSE);
		fprintf('    Basis BSP time                  = %.3f s\n', stats.elapsedTime);
		if cfgB.printBranchMasks
			print_bsp_branch_dictionary_masks_local(best.mask, arch, 'After dictionary-basis-level BSP masks');
		end
	end
end


function sal = compute_branch_saliency_scores_local(Coef, mask, Xscore, arch, normOpt, cfgB)
%COMPUTE_BRANCH_SALIENCY_SCORES_LOCAL Branch output-impact saliency.
%
% For each active branch A{src,ell}, the score is computed from active
% dictionary columns in that branch.  A column score is the no-refit output
% RMSE change after zeroing that column only.  The branch score is the mean
% of the largest TopK column scores.  This is a finite-ablation version of a
% Jacobian-style functional saliency: it measures how much the current model
% output depends on that branch without retraining each candidate structure.
	sal = repmat(struct('src', [], 'ell', [], 'activeCoeff', [], 'activeCols', [], ...
		'colScores', [], 'topKMean', [], 'meanScore', [], 'sumScore', [], ...
		'fullDrop', [], 'layerNormScore', [], 'rankScore', [], 'legal', []), 0, 1);
	try
		Yfull = model_forward(Xscore, Coef, arch, normOpt);
		if ~isreal(Yfull) || any(~isfinite(Yfull(:)))
			return;
		end
	catch
		return;
	end

	topK = max(1, round(getfield_default_local(cfgB, 'branchSaliencyTopK', 3)));
	for ell = 1:size(mask, 2)
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M) || ~any(M(:))
				continue;
			end
			M = logical(M);
			activeCols = find(any(M, 1));
			colScores = inf(1, numel(activeCols));
			for jj = 1:numel(activeCols)
				cc = activeCols(jj);
				CoefDrop = Coef;
				A = CoefDrop{src, ell};
				if isempty(A) || cc > size(A, 2)
					continue;
				end
				A(:, cc) = 0;
				CoefDrop{src, ell} = A;
				colScores(jj) = output_delta_rmse_local(Yfull, Xscore, CoefDrop, arch, normOpt);
			end

			validColScores = colScores(isfinite(colScores));
			if isempty(validColScores)
				topKMean = Inf;
				meanScore = Inf;
				sumScore = Inf;
			else
				sortedScores = sort(validColScores, 'descend');
				kk = min(topK, numel(sortedScores));
				topKMean = mean(sortedScores(1:kk));
				meanScore = mean(validColScores);
				sumScore = sum(validColScores);
			end

			CoefFullDrop = Coef;
			Afull = CoefFullDrop{src, ell};
			if ~isempty(Afull)
				Afull(:) = 0;
				CoefFullDrop{src, ell} = Afull;
			end
			fullDrop = output_delta_rmse_local(Yfull, Xscore, CoefFullDrop, arch, normOpt);

			move = struct('src', src, 'ell', ell);
			legal = is_branch_bsp_mask_legal_local(apply_branch_deletion_local(mask, move), cfgB);

			sal(end + 1) = struct('src', src, 'ell', ell, 'activeCoeff', nnz(M), ...
				'activeCols', activeCols, 'colScores', colScores, 'topKMean', topKMean, ...
				'meanScore', meanScore, 'sumScore', sumScore, 'fullDrop', fullDrop, ...
				'layerNormScore', NaN, 'rankScore', NaN, 'legal', legal); %#ok<AGROW>
		end
	end

	if isempty(sal)
		return;
	end

	% Layer-normalized score is reported for scale diagnostics.  It can also be
	% used for pruning when branchSaliencyUseLayerNorm=true.
	for ell = 1:size(mask, 2)
		idx = find([sal.ell] == ell);
		if isempty(idx)
			continue;
		end
		den = sum([sal(idx).topKMean]);
		if ~isfinite(den) || den <= 0
			den = eps;
		end
		for ii = idx(:).'
			sal(ii).layerNormScore = sal(ii).topKMean / den;
		end
	end

	useLayerNorm = getfield_default_local(cfgB, 'branchSaliencyUseLayerNorm', false);
	mode = lower(strtrim(char(getfield_default_local(cfgB, 'branchSaliencyScoreMode', 'topk_mean_output_delta'))));
	for ii = 1:numel(sal)
		if useLayerNorm
			sal(ii).rankScore = sal(ii).layerNormScore;
		else
			switch mode
				case {'mean_output_delta', 'mean'}
					sal(ii).rankScore = sal(ii).meanScore;
				case {'sum_output_delta', 'sum'}
					sal(ii).rankScore = sal(ii).sumScore;
				case {'full_drop_output_delta', 'full_drop', 'branch_drop'}
					sal(ii).rankScore = sal(ii).fullDrop;
				otherwise
					sal(ii).rankScore = sal(ii).topKMean;
			end
		end
	end
end

function d = output_delta_rmse_local(Yfull, X, CoefDrop, arch, normOpt)
	try
		Ydrop = model_forward(X, CoefDrop, arch, normOpt);
		if ~isreal(Ydrop) || any(~isfinite(Ydrop(:)))
			d = Inf;
			return;
		end
		delta = Ydrop - Yfull;
		if ~isreal(delta) || any(~isfinite(delta(:)))
			d = Inf;
		else
			d = sqrt(mean(delta(:).^2));
		end
	catch
		d = Inf;
	end
end

function print_branch_saliency_table_local(sal, roundIdx)
	if isempty(sal)
		return;
	end
	[~, ord] = sort([sal.rankScore], 'descend');
	fprintf('\n    Branch output-impact saliency scores, round %d\n', roundIdx);
	fprintf('    %4s  %8s  %12s  %12s  %12s  %12s  %8s  %12s  %s\n', ...
		'rank', 'block', 'rankScore', 'topKMean', 'layerNorm', 'fullDrop', 'active', 'cols', 'legal');
	for rr = 1:numel(ord)
		ii = ord(rr);
		fprintf('    %4d  A{%d,%d}  %12.4e  %12.4e  %12.4e  %12.4e  %8d  %12s  %d\n', ...
			rr, sal(ii).src, sal(ii).ell, sal(ii).rankScore, sal(ii).topKMean, ...
			sal(ii).layerNormScore, sal(ii).fullDrop, sal(ii).activeCoeff, ...
			format_numeric_vector_local(sal(ii).activeCols), sal(ii).legal);
	end
end

function [maskList, thetaList, reasonList] = collect_branch_deletion_candidates_local(beam, cfgB)
	maskList = {};
	thetaList = {};
	reasonList = {};
	seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
	for b = 1:numel(beam)
		moves = enumerate_one_branch_deletions_local(beam(b).mask, cfgB);
		for mm = 1:numel(moves)
			maskCand = apply_branch_deletion_local(beam(b).mask, moves(mm));
			if ~is_branch_bsp_mask_legal_local(maskCand, cfgB)
				continue;
			end
			key = mask_signature_local(maskCand);
			if isKey(seen, key)
				continue;
			end
			seen(key) = true;
			maskList{end + 1, 1} = maskCand; %#ok<AGROW>
			thetaList{end + 1, 1} = pack_Coef_M_by_mask(beam(b).Coef, maskCand); %#ok<AGROW>
			reasonList{end + 1, 1} = sprintf('drop branch A{%d,%d}', moves(mm).src, moves(mm).ell); %#ok<AGROW>
		end
	end
end

function [maskList, thetaList, reasonList] = collect_basis_deletion_candidates_local(beam, cfgB)
	maskList = {};
	thetaList = {};
	reasonList = {};
	seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
	for b = 1:numel(beam)
		moves = enumerate_one_basis_deletions_local(beam(b).mask, cfgB);
		for mm = 1:numel(moves)
			maskCand = apply_basis_deletion_local(beam(b).mask, moves(mm));
			if ~is_bsp_mask_legal_local(maskCand, cfgB)
				continue;
			end
			key = mask_signature_local(maskCand);
			if isKey(seen, key)
				continue;
			end
			seen(key) = true;
			maskList{end + 1, 1} = maskCand; %#ok<AGROW>
			thetaList{end + 1, 1} = pack_Coef_M_by_mask(beam(b).Coef, maskCand); %#ok<AGROW>
			reasonList{end + 1, 1} = sprintf('drop A{%d,%d} basis %d', moves(mm).src, moves(mm).ell, moves(mm).col); %#ok<AGROW>
		end
	end
end

function cand = evaluate_bsp_candidate_list_local(maskList, thetaList, reasonList, roundIdx, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, cfgB, stageName)
	nCand = numel(maskList);
	cand = empty_bsp_state_array_local();
	if nCand == 0
		return;
	end

	usePar = should_use_parallel_bsp_local(cfgB);
	if usePar && getfield_default_local(cfgB, 'parallelVerbose', true) && cfgB.verbose
		fprintf('      evaluating %d %s candidates in parallel ...\n', nCand, stageName);
	elseif cfgB.verbose
		fprintf('      evaluating %d %s candidates serially ...\n', nCand, stageName);
	end

	thetaCell = cell(nCand, 1);
	CoefCell = cell(nCand, 1);
	trainVec = inf(nCand, 1);
	valVec = inf(nCand, 1);
	scoreVec = inf(nCand, 1);
	exitVec = nan(nCand, 1);
	outputCell = cell(nCand, 1);

	if usePar
		parfor kk = 1:nCand
			maskK = maskList{kk};
			cfgBK = cfgB;
			if strcmpi(stageName, 'branch')
				cfgBK.fastMaxIter = getfield_default_local(cfgB, 'branchFastMaxIter', cfgB.fastMaxIter);
				cfgBK.fastMaxFunEvals = getfield_default_local(cfgB, 'branchFastMaxFunEvals', cfgB.fastMaxFunEvals);
			end
			[thetaK, trainK, valK, exitK, outK] = run_bsp_fast_fit_local( ...
				thetaList{kk}, maskK, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, cfgBK);
			CoefK = unpack_Coef_M_by_mask(thetaK(:), maskK, coefZero);
			thetaCell{kk} = thetaK(:);
			CoefCell{kk} = CoefK;
			trainVec(kk) = trainK;
			valVec(kk) = valK;
			exitVec(kk) = exitK;
			outputCell{kk} = outK;
			if strcmpi(stageName, 'branch')
				scoreVec(kk) = valK + getfield_default_local(cfgB, 'branchSizePenalty', 0) * count_active_branches_local(maskK) + cfgB.sizePenalty * count_active_dictionary_bases_local(maskK);
			else
				scoreVec(kk) = valK + cfgB.sizePenalty * count_active_dictionary_bases_local(maskK);
			end
		end
	else
		for kk = 1:nCand
			maskK = maskList{kk};
			cfgBK = cfgB;
			if strcmpi(stageName, 'branch')
				cfgBK.fastMaxIter = getfield_default_local(cfgB, 'branchFastMaxIter', cfgB.fastMaxIter);
				cfgBK.fastMaxFunEvals = getfield_default_local(cfgB, 'branchFastMaxFunEvals', cfgB.fastMaxFunEvals);
			end
			[thetaK, trainK, valK, exitK, outK] = run_bsp_fast_fit_local( ...
				thetaList{kk}, maskK, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, cfgBK);
			CoefK = unpack_Coef_M_by_mask(thetaK(:), maskK, coefZero);
			thetaCell{kk} = thetaK(:);
			CoefCell{kk} = CoefK;
			trainVec(kk) = trainK;
			valVec(kk) = valK;
			exitVec(kk) = exitK;
			outputCell{kk} = outK;
			if strcmpi(stageName, 'branch')
				scoreVec(kk) = valK + getfield_default_local(cfgB, 'branchSizePenalty', 0) * count_active_branches_local(maskK) + cfgB.sizePenalty * count_active_dictionary_bases_local(maskK);
			else
				scoreVec(kk) = valK + cfgB.sizePenalty * count_active_dictionary_bases_local(maskK);
			end
		end
	end

	for kk = 1:nCand
		cand(end + 1) = make_bsp_state_local(maskList{kk}, thetaCell{kk}, CoefCell{kk}, trainVec(kk), valVec(kk), scoreVec(kk), roundIdx, reasonList{kk}, exitVec(kk), outputCell{kk}); %#ok<AGROW>
	end
end


function tol = compute_bsp_improve_tol_local(~, ~, ~)
%COMPUTE_BSP_IMPROVE_TOL_LOCAL Deprecated in compact-mask route fix2.
%
% BSP improvement now uses the direct rule: candidateScore < referenceScore.
% This helper is kept only for compatibility if older local code paths call it.
	tol = 0;
end

function tf = should_use_parallel_bsp_local(cfgB)
	tf = false;
	if ~getfield_default_local(cfgB, 'useParallel', false)
		return;
	end
	try
		if exist('gcp', 'file') ~= 2
			return;
		end
		pool = gcp('nocreate');
		if isempty(pool) && getfield_default_local(cfgB, 'autoStartParallelPool', true)
			try
				pool = parpool;
			catch
				pool = [];
			end
		end
		tf = ~isempty(pool);
	catch
		tf = false;
	end
end


function print_bsp_branch_dictionary_masks_local(mask, arch, titleText)
%PRINT_BSP_BRANCH_DICTIONARY_MASKS_LOCAL Print branch/block-wise dictionary masks.
%
% This report is intentionally branch-wise.  Each A{src,ell} block owns its
% own dictionary submask.  The global dictionary-basis count printed by the
% BSP summary is the sum of active dictionary columns over these blocks.
	if nargin < 3 || isempty(titleText)
		titleText = 'BSP branch dictionary masks';
	end
	fprintf('\n  %s\n', titleText);
	fprintf('  %s\n', repmat('-', 1, numel(titleText)));

	dims = [];
	try
		dims = get_arch_dims(arch);
	catch
		if isfield(arch, 'dims')
			dims = arch.dims;
		end
	end
	if isempty(dims)
		% Fallback: still print columns, but term names may be unavailable.
		dims = ones(1, size(mask, 2) + 1);
	end

	for ell = 1:size(mask, 2)
		for src = 1:ell
			if src > size(mask, 1)
				continue;
			end
			M = mask{src, ell};
			if isempty(M)
				fprintf('    A{%d,%d}: cols [] | terms {} | active coeff 0 | row counts []\n', src, ell);
				continue;
			end
			M = logical(M);
			cols = find(any(M, 1));
			rowCounts = sum(M, 2).';
			termNames = get_branch_term_names_for_mask_local(cols, src, ell, dims, arch);
			fprintf('    A{%d,%d}: cols %s | terms {%s} | active coeff %d | row counts %s\n', ...
				src, ell, format_numeric_vector_local(cols), strjoin(termNames, ', '), nnz(M), format_numeric_vector_local(rowCounts));
		end
	end
end

function names = get_branch_term_names_for_mask_local(cols, src, ell, dims, arch)
	if isempty(cols)
		names = {};
		return;
	end
	sourceIndex = ell - src + 1;
	if sourceIndex < 1 || sourceIndex > numel(dims)
		inputDim = size(cols, 2);
		varPrefix = 'h';
	else
		inputDim = dims(sourceIndex);
		if sourceIndex == 1
			varPrefix = 'x';
		else
			varPrefix = 'h';
		end
	end
	try
		terms = branch_dictionary_terms(inputDim, arch, varPrefix, ell);
		names = cell(1, numel(cols));
		for kk = 1:numel(cols)
			cc = cols(kk);
			if cc >= 1 && cc <= numel(terms)
				names{kk} = terms(cc).name;
			else
				names{kk} = sprintf('col%d', cc);
			end
		end
	catch
		names = arrayfun(@(cc) sprintf('col%d', cc), cols(:).', 'UniformOutput', false);
	end
end

function s = format_numeric_vector_local(v)
	if isempty(v)
		s = '[]';
		return;
	end
	if islogical(v)
		v = double(v);
	end
	s = ['[' strtrim(sprintf('%g ', v)) ']'];
end

function [thetaOut, trainMSE, valMSE, exitflagOut, outputOut] = run_bsp_fast_fit_local( ...
	thetaStart, mask, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, cfgB)
	nVars = count_active_mask(mask);
	if nVars == 0
		thetaOut = zeros(0, 1);
		exitflagOut = NaN;
		outputOut = [];
		trainMSE = eval_mse_local(thetaOut, mask, coefZero, Xtr, Ytr, arch, normOpt);
		valMSE = eval_mse_local(thetaOut, mask, coefZero, Xval, Yval, arch, normOpt);
		return;
	end
	[thetaFast, trainFast, valFast, exitflagOut, outputOut] = run_lsq_refine_once_local( ...
		thetaStart(:), mask, coefZero, Xtr, Ytr, Xval, Yval, arch, normOpt, cfg, ...
		make_residual_scale_local(Ytr, cfg.objective), cfgB.fastMaxIter, cfgB.fastMaxFunEvals);
	valStart = eval_mse_local(thetaStart(:), mask, coefZero, Xval, Yval, arch, normOpt);
	trainStart = eval_mse_local(thetaStart(:), mask, coefZero, Xtr, Ytr, arch, normOpt);
	if isfinite(valStart) && (~isfinite(valFast) || valFast > valStart * (1 + cfgB.maxRelValIncrease))
		thetaOut = thetaStart(:);
		trainMSE = trainStart;
		valMSE = valStart;
	else
		thetaOut = thetaFast(:);
		trainMSE = trainFast;
		valMSE = valFast;
	end
end

function s = make_bsp_state_local(mask, theta, Coef, trainMSE, valMSE, score, round, reason, exitflag, output)
	s = struct();
	s.mask = mask;
	s.theta = theta(:);
	s.Coef = Coef;
	s.trainMSE = trainMSE;
	s.valMSE = valMSE;
	s.score = score;
	s.round = round;
	s.reason = reason;
	s.exitflag = exitflag;
	s.output = output;
end

function arr = empty_bsp_state_array_local()
	arr = repmat(struct('mask', [], 'theta', [], 'Coef', [], 'trainMSE', [], 'valMSE', [], ...
		'score', [], 'round', [], 'reason', [], 'exitflag', [], 'output', []), 0, 1);
end

function out = unique_bsp_states_local(states, maxKeep)
	out = empty_bsp_state_array_local();
	seen = containers.Map('KeyType', 'char', 'ValueType', 'logical');
	for k = 1:numel(states)
		key = mask_signature_local(states(k).mask);
		if isKey(seen, key)
			continue;
		end
		seen(key) = true;
		out(end + 1) = states(k); %#ok<AGROW>
		if numel(out) >= maxKeep
			break;
		end
	end
end

function moves = enumerate_one_branch_deletions_local(mask, cfgB)
	moves = repmat(struct('src', [], 'ell', []), 0, 1);
	minBranches = max(0, round(getfield_default_local(cfgB, 'branchMinActiveBranches', 1)));
	if count_active_branches_local(mask) <= minBranches
		return;
	end
	for ell = 1:size(mask, 2)
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M) || ~any(M(:))
				continue;
			end
			moves(end + 1) = struct('src', src, 'ell', ell); %#ok<AGROW>
		end
	end
end

function maskOut = apply_branch_deletion_local(maskIn, move)
	maskOut = maskIn;
	M = maskOut{move.src, move.ell};
	if ~isempty(M)
		M(:) = false;
		maskOut{move.src, move.ell} = M;
	end
end

function tf = is_branch_bsp_mask_legal_local(mask, cfgB)
	tf = count_active_mask(mask) >= cfgB.minTotalActive;
	if ~tf
		return;
	end
	minBranches = max(0, round(getfield_default_local(cfgB, 'branchMinActiveBranches', 1)));
	tf = count_active_branches_local(mask) >= minBranches;
end

function moves = enumerate_one_basis_deletions_local(mask, cfgB)
	moves = repmat(struct('src', [], 'ell', [], 'col', []), 0, 1);
	for ell = 1:size(mask, 2)
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M) || ~any(M(:))
				continue;
			end
			activeCols = find(any(M, 1));
			for cc = activeCols(:).'
				Mtmp = M;
				Mtmp(:, cc) = false;
				if count_active_dictionary_columns_in_block_local(Mtmp) < cfgB.minTermsPerXiBlock
					continue;
				end
				moves(end + 1) = struct('src', src, 'ell', ell, 'col', cc); %#ok<AGROW>
			end
		end
	end
end

function maskOut = apply_basis_deletion_local(maskIn, move)
	maskOut = maskIn;
	M = maskOut{move.src, move.ell};
	if ~isempty(M) && move.col <= size(M, 2)
		M(:, move.col) = false;
		maskOut{move.src, move.ell} = M;
	end
end

function tf = is_bsp_mask_legal_local(mask, cfgB)
	tf = count_active_mask(mask) >= cfgB.minTotalActive;
	if ~tf
		return;
	end
	if cfgB.minTermsPerXiBlock <= 0
		return;
	end
	for ell = 1:size(mask, 2)
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M)
				continue;
			end
			if count_active_dictionary_columns_in_block_local(M) < cfgB.minTermsPerXiBlock
				tf = false;
				return;
			end
		end
	end
end

function n = count_active_branches_local(mask)
	n = 0;
	for ell = 1:size(mask, 2)
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M)
				continue;
			end
			if any(M(:))
				n = n + 1;
			end
		end
	end
end

function n = count_active_dictionary_bases_local(mask)
	n = 0;
	for ell = 1:size(mask, 2)
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M)
				continue;
			end
			n = n + count_active_dictionary_columns_in_block_local(M);
		end
	end
end

function n = count_active_dictionary_columns_in_block_local(M)
	if isempty(M)
		n = 0;
	else
		n = nnz(any(M, 1));
	end
end

function key = mask_signature_local(mask)
	parts = cell(0, 1);
	for ell = 1:size(mask, 2)
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M)
				parts{end + 1, 1} = sprintf('%d,%d:[]', src, ell); %#ok<AGROW>
			else
				parts{end + 1, 1} = sprintf('%d,%d:%s', src, ell, sprintf('%d', M(:))); %#ok<AGROW>
			end
		end
	end
	key = strjoin(parts, '|');
end

function out = format_count_for_log_local(x)
	if isempty(x) || (isnumeric(x) && isinf(x))
		out = 'all';
	else
		out = sprintf('%d', round(x));
	end
end

function Coef = zero_coefficients_outside_mask_local(Coef, mask)
	for ell = 1:size(mask, 2)
		for src = 1:size(mask, 1)
			if src <= ell && ~isempty(mask{src, ell}) && ~isempty(Coef{src, ell})
				Coef{src, ell}(~mask{src, ell}) = 0;
			end
		end
	end
end
