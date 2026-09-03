function result = phdnn_identify(task, opts)
%PHDNN_IDENTIFY Per-output PySR-guided augmented PhDN identification.
%
% Stage 0: per-output fixed-SINDy bypass plus native PySR for unresolved outputs.
% Stage 1: compile the selected symbolic cores into one augmented PhDN DAG,
%          then refine only the exact SR support with a fixed-support BP/LSQ
%          handoff.  Constant+Poly_2 augmentation channels remain frozen at zero.
% Stage 2: release the full augmented PhDN network, continue from the polished
%          Stage-1 coefficients, and run the existing BP/LSQ plus optional
%          contribution-pruning route.
%

	runTimerTotal = tic;
	timeStats = struct();
	timeStats.taskName = task.name;

	if isfield(task, 'domain') && ~isempty(task.domain)
		if isfield(task, 'variableNames')
			task.domain = normalize_task_domain(task.domain, task.nx, task.variableNames);
		else
			task.domain = normalize_task_domain(task.domain, task.nx);
		end
		timeStats.domainSummary = format_variable_domain(task.domain);
	end

	% Make the normalized ID domain available to deterministic domain-safety filters.
	% This is not OOD information; it is the declared training/input domain.
	if isfield(task, 'domain') && ~isempty(task.domain)
		if ~isfield(opts, 'training') || isempty(opts.training)
			opts.training = struct();
		end
		if ~isfield(opts, 'init') || isempty(opts.init)
			opts.init = struct();
		end
		if ~isfield(opts.init, 'domainFilter') || isempty(opts.init.domainFilter)
			opts.init.domainFilter = struct();
		end
		opts.training.inputDomain = task.domain;
		opts.init.domainFilter.inputDomain = task.domain;
	end


	% The fixed SINDy library is first tested as a preliminary fast path. After
	% bypass rejection, native PySR independently identifies one complete symbolic
	% expression per output. Stage 1 keeps SR operators as structural DAG channels
	% and adds only a zero-initialized constant+Poly_2 augmentation family.


	arch_base = task.arch;
	arch_base.nx = task.nx;
	arch_base.ny = task.ny;

	% Optional demo/option-level architecture overrides.  If omitted,
	% task.arch.hiddenDims or the default ny-width hidden states are used.
	if isfield(opts, 'arch') && ~isempty(opts.arch)
		if isfield(opts.arch, 'dims') && ~isempty(opts.arch.dims)
			arch_base.dims = opts.arch.dims;
		end
		if isfield(opts.arch, 'hiddenDims') && ~isempty(opts.arch.hiddenDims)
			arch_base.hiddenDims = opts.arch.hiddenDims;
		end
		if isfield(opts.arch, 'hiddenWidth') && ~isempty(opts.arch.hiddenWidth)
			arch_base.hiddenWidth = opts.arch.hiddenWidth;
		end
	end

	arch_base.safety = opts.safety;
	arch_base.feasibility = opts.training;

	% Case dictionaries are written in raw physical-coordinate support.  If affine
	% input/output normalization is enabled, close the compact support under the
	% normalized coordinate system before templates and masks are created.
	% For rowTerms this remains hard row-wise support; for termsByBlock/layer/dim
	% this augments the candidate dictionary without restoring old full dictionaries.
	[normClosedArch, normClosedInfo] = augment_case_dictionary_for_normalization(arch_base, opts.norm);
	arch_base = normClosedArch;
	if isfield(normClosedInfo, 'applied') && normClosedInfo.applied
		fprintf('Case-dictionary normalization closure: %s\n', normClosedInfo.message);
	end

	if ~isfield(opts.training, 'opArgPolyOrderList') || isempty(opts.training.opArgPolyOrderList)
		if isfield(arch_base, 'interact') && isfield(arch_base.interact, 'opArgPolyOrder') && ...
				~isempty(arch_base.interact.opArgPolyOrder)
			opArgPolyOrderList = arch_base.interact.opArgPolyOrder;
		else
			opArgPolyOrderList = 1:max(1, arch_base.polyOrder);
		end
	else
		opArgPolyOrderList = opts.training.opArgPolyOrderList;
	end

	opArgPolyOrderList = unique(round(opArgPolyOrderList(:).'));
	opArgPolyOrderList = opArgPolyOrderList(opArgPolyOrderList >= 1);
	if isempty(opArgPolyOrderList)
		opArgPolyOrderList = 1;
	end
	opts.training.opArgPolyOrderList = opArgPolyOrderList;

	% ---------------------------------------------------------------------
	% Data.
	% ---------------------------------------------------------------------
	timerData = tic;
	[X, Y] = sample_task_data(task, opts.data.nSamples);
	timeStats.dataGenerationTime = toc(timerData);

	timerSplit = tic;
	[Xtr, Ytr, Xval, Yval, Xte, Yte] = split_train_val_test( ...
		X, Y, opts.data.ratioTrain, opts.data.ratioVal);

	% Optional robustness-only perturbation of derivative observations.  The
	% clean test targets Yte and all OOD targets remain untouched.  Noise is
	% applied after the exact deterministic split so the state samples are
	% identical to the corresponding clean-data run.
	labelNoiseInfo = struct('enabled',false);
	if isfield(opts.data,'derivativeLabelNoise') && ...
			isstruct(opts.data.derivativeLabelNoise) && ...
			isfield(opts.data.derivativeLabelNoise,'enable') && ...
			logical(opts.data.derivativeLabelNoise.enable)
		[Ytr,Yval,labelNoiseInfo] = apply_derivative_label_noise( ...
			Ytr,Yval,opts.data.derivativeLabelNoise);
	end

	normOpt = fit_norm_options(Xtr, Ytr, opts.norm.useInputOutputNorm, opts.norm.useLayerNorm);
	timeStats.splitAndNormTime = toc(timerSplit);
	timeStats.derivativeLabelNoise = labelNoiseInfo;


	Xood = [];
	Yood = [];
	oodDomain = [];
	timeStats.oodEnabled = opts.ood.enable;
	timeStats.oodDataGenerationTime = 0;
	if opts.ood.enable
		timerOOD = tic;
		oodDomain = make_ood_domain(task, opts);
		[Xood, Yood] = sample_task_data(task, opts.ood.nSamples, oodDomain);
		timeStats.oodDataGenerationTime = toc(timerOOD);
		timeStats.nOODSamples = size(Xood, 1);
		timeStats.oodDomainLb = reshape(oodDomain.lb, 1, []);
		timeStats.oodDomainUb = reshape(oodDomain.ub, 1, []);
		timeStats.oodDomainSummary = format_variable_domain(oodDomain);
	else
		timeStats.nOODSamples = 0;
	end

	% ---------------------------------------------------------------------
	% Stage 0: fixed-SINDy bypass plus native per-output PySR search.
	% ---------------------------------------------------------------------
	[archTrialSpecs, stage0Info] = maybe_prepare_per_output_pysr_stage0_trial_specs_local( ...
		task, opts, arch_base, normOpt, Xtr, Ytr, Xval, Yval, Xte, Yte, Xood, Yood, labelNoiseInfo);
	if ~(isstruct(stage0Info) && isfield(stage0Info, 'applied') && stage0Info.applied)
		error('Stage 0 per-output PySR did not produce a usable PhDN candidate. Reason: %s', ...
			getfield_default_local(stage0Info, 'reason', 'unknown'));
	end
	timeStats.stage0 = stage0Info;

	% A successful mechanical-precision SINDy bypass is already the final
	% single-layer identification result. Do not compile it into a deeper PhDN
	% and do not run Stage 1 or Stage 2.
	if logical(getfield_default_local(stage0Info, 'usedSingleLayerBypass', false))
		result = finalize_sindy_stage0_bypass_result_local(task, opts, stage0Info, ...
			Xtr, Ytr, Xval, Yval, Xte, Yte, Xood, Yood, oodDomain, ...
			timeStats, runTimerTotal);
		return;
	end

	% ---------------------------------------------------------------------
	% Architecture/mask trials.
	% ---------------------------------------------------------------------
	bestValMSE = Inf;
	bestArch = [];
	bestCoefTemplate = [];
	bestCoef = [];
	bestTheta = [];
	bestInitStats = [];
	bestTrainMask = [];
	bestOpArgPolyOrder = opArgPolyOrderList(1);
	bestOpts = opts;
	bestTrialSpec = struct();

	timeStats.initStatsByOpArg = struct([]);
	timerModelSelection = tic;

	for kop = 1:numel(archTrialSpecs)
		trialSpec = archTrialSpecs(kop);
		opArgPolyOrder = getfield_default_local(trialSpec, 'opArgPolyOrder', 1);

		arch_cur = trialSpec.arch;
		if ~isfield(arch_cur, 'interact') || isempty(arch_cur.interact)
			arch_cur.interact = struct();
		end
		arch_cur.interact.opArgPolyOrder = opArgPolyOrder;
		arch_cur.safety = opts.safety;
		arch_cur.feasibility = opts.training;
		isStage0CandidateCur = isfield(trialSpec, 'isStage0Candidate') && trialSpec.isStage0Candidate;
		preserveRawStage0HiddenStates = isStage0CandidateCur && logical(getfield_default_local( ...
			opts.stage1, 'preserveRawStage0HiddenStates', true));
		if preserveRawStage0HiddenStates
			% The hidden nodes of a compiled symbolic DAG are compiler temporaries,
			% not user-model saturation layers.  Applying the generic PhDN hidden
			% clip here changes, for example, c*((exp(z)-d)^3+e)/sigma whenever
			% the cube exceeds the clip before the small factor c is applied.
			% Preserve official-PySR forward semantics on these temporary states.
			arch_cur.safety.hiddenLayerOutputClip = Inf;
		end

		fprintf('\n========================================\n');
		if isfield(trialSpec, 'isStage0Candidate') && trialSpec.isStage0Candidate
			fprintf('Stage-0 compiled PhDN candidate trial %d/%d: %s\n', ...
				kop, numel(archTrialSpecs), getfield_default_local(trialSpec, 'label', 'stage0_candidate'));
		else
			fprintf('Masked-LSQ architecture trial: opArgPolyOrder = %d\n', opArgPolyOrder);
		end
		fprintf('========================================\n');

		fprintf('Operator mode for this trial       = %s\n', get_operator_label_local(arch_cur));
		if isfield(trialSpec, 'isStage0Candidate') && trialSpec.isStage0Candidate
			fprintf('Stage-1 augmented DAG source      = %s\n', arch_cur.caseDictionary.source);
			fprintf('Stage-0 selected core expressions = %s\n', strjoin(getfield_default_local(trialSpec, 'expressions', {'<unavailable>'}), '; '));
			fprintf('Compiled SR skeleton              = %s\n', strjoin(getfield_default_local(trialSpec, 'compiledTerms', {'<unavailable>'}), '; '));
			if preserveRawStage0HiddenStates
				fprintf('Compiled SR hidden-state clipping = disabled (raw Stage-0 expression semantics)\n');
			end
			if isfield(trialSpec, 'compileInfo')
				fprintf('Stage-0 PhDN compile mode     = %s, layers = %d, hiddenDims = [%s]\n', ...
					getfield_default_local(trialSpec.compileInfo, 'compileMode', 'unknown'), ...
					getfield_default_local(trialSpec.compileInfo, 'nLayers', NaN), ...
					num2str(getfield_default_local(trialSpec.compileInfo, 'hiddenDims', [])));
			end
		end

		Coef_template_cur = create_coef_template(arch_cur);
		opts_cur = opts;
		isStage0CandidateCur = isfield(trialSpec, 'isStage0Candidate') && trialSpec.isStage0Candidate;

		% The compiled Stage-0 DAG is augmented with zero-initialized uniform augmentation
		% channels.  Training now uses an explicit two-step handoff:
		%   Stage 1: keep only the exact nonzero SR support active and polish its
		%            continuous coefficients with support-fixed BP/LSQ;
		%   Stage 2: release the full augmented PhDN mask, initialize the newly
		%            released channels at zero, and run the existing refinement and
		%            optional contribution-pruning route.
		rowwiseMaskInfoCur = struct('applied',false,'active',0, ...
			'reason','SR fixed-support handoff followed by full augmented-PhDN release');
		if isStage0CandidateCur
			opts_cur.training.useAdmissibleMask = false;
			if isfield(opts_cur.training, 'admissibleA')
				opts_cur.training = rmfield(opts_cur.training, 'admissibleA');
			end
			opts_cur.training.admissibleDefaultAllowed = true;
		else
			[rowwiseMaskCur, rowwiseMaskInfoCur] = make_rowwise_mask_from_case_dictionary(Coef_template_cur, arch_cur);
			if isfield(rowwiseMaskInfoCur, 'applied') && rowwiseMaskInfoCur.applied
				opts_cur.training.useAdmissibleMask = true;
				opts_cur.training.admissibleA = rowwiseMaskCur;
				opts_cur.training.admissibleDefaultAllowed = false;
				fprintf('Row-wise hard admissible mask from caseDictionary.rowTerms: active = %d\n', rowwiseMaskInfoCur.active);
			end
		end
		trainMask_cur = build_training_mask(Coef_template_cur, opts_cur);

		% Deterministic structural filters remain unchanged for non-SR routes.  The
		% SR route deliberately preserves the exact compiler support during the
		% fixed-support handoff and performs sparsification only after the full PhDN
		% network has been released.
		if isStage0CandidateCur
			identityCancellationReportCur = struct('skipped',true, ...
				'reason','SR exact-support handoff preserves compiler semantics');
			inputDomainInvReportCur = struct('skipped',true, ...
				'reason','SR exact-support handoff preserves compiler semantics');
		else
			[trainMask_cur, identityCancellationReportCur] = remove_redundant_identity_terms_from_mask( ...
				trainMask_cur, arch_cur, opts_cur);
			if is_true_operator_arch_local(arch_cur)
				[trainMask_cur, inputDomainInvReportCur] = remove_input_domain_singular_inv_terms_from_mask( ...
					trainMask_cur, arch_cur, opts_cur);
			else
				inputDomainInvReportCur = struct('skipped', true, 'reason', 'surrogate-operator mode');
				fprintf('Surrogate-operator mode: applied identity-cancellation filter and skipped true-domain singular-inverse filter.\n');
			end
		end

		timerInit = tic;
		stage1StatsCur = struct();
		stage1StatsCur.applied = isStage0CandidateCur;
		stage1StatsCur.method = 'sr_fixed_support_bp_handoff_then_full_phdn_release';
		stage1StatsCur.dictionaryMode = getfield_default_local(opts_cur.stage1, ...
			'dictionaryMode', 'sr_structural_dag_plus_uniform_poly2_augmentation');
		stage1StatsCur.stage1OnlyTime = 0;
		stage1StatsCur.elapsedTime = 0;
		stage1StatsCur.fixedSupportRefinementEnabled = false;

		% Verify exact Stage-0 reproduction before any coefficient update.  This
		% catches variable indexing and operator-semantic translation errors without
		% spending time optimizing a corrupted initialization.
		stage0ReproductionPrecheck = struct();
		if isStage0CandidateCur && logical(getfield_default_local(opts_cur.stage1, 'requireExactStage0Reproduction', true))
			YseedVal = model_forward(Xval, trialSpec.seedCoef, arch_cur, normOpt);
			YrefVal = getfield_default_local(getfield_default_local(trialSpec, 'stage0Model', struct()), 'YhatVal', []);
			if isempty(YrefVal)
				error('Stage 1 exact-reproduction precheck cannot run because Stage-0 validation predictions are unavailable.');
			end
			if ~isequal(size(YseedVal), size(YrefVal))
				error('Stage 1 exact-reproduction prediction size mismatch: seed [%s], Stage0 [%s].', ...
					num2str(size(YseedVal)), num2str(size(YrefVal)));
			end
			absTolRepro = getfield_default_local(opts_cur.stage1, 'stage0ReproductionAbsTolerance', 1e-10);
			relTolRepro = getfield_default_local(opts_cur.stage1, 'stage0ReproductionRelTolerance', 1e-6);
			maxAbsDiff = max(abs(YseedVal(:) - YrefVal(:)));
			predTol = absTolRepro + relTolRepro * max(max(abs(YrefVal(:))), eps);
			seedValMetrics = compute_regression_metrics(YseedVal, Yval);
			refValMetrics = compute_regression_metrics(YrefVal, Yval);
			stage0ReproductionPrecheck.maxPredictionDifference = maxAbsDiff;
			stage0ReproductionPrecheck.predictionTolerance = predTol;
			stage0ReproductionPrecheck.seedValMSE = seedValMetrics.mse;
			stage0ReproductionPrecheck.referenceValMSE = refValMetrics.mse;
			if ~isfinite(maxAbsDiff) || maxAbsDiff > predTol
				error(['Stage 1 failed to reproduce the Stage-0 PySR validation prediction before BP/LSQ: ', ...
					'max prediction difference %.3e, tolerance %.3e, Stage0 MSE %.6e, seed MSE %.6e.'], ...
					maxAbsDiff, predTol, refValMetrics.mse, seedValMetrics.mse);
			end
			fprintf('Stage 1 exact Stage-0 reproduction precheck passed: max prediction difference %.3e.\n', maxAbsDiff);
		end

		% ------------------------------------------------------------------
		% Stage 1 fixed-support BP handoff.
		% ------------------------------------------------------------------
		handoffCoefCur = [];
		if isStage0CandidateCur
			if ~isfield(trialSpec, 'seedCoef') || isempty(trialSpec.seedCoef)
				error('Stage 1 SR-to-PhDN coefficient initialization failed: trialSpec.seedCoef is missing or empty.');
			end

			fixedCfg = getfield_default_local(opts_cur.stage1, 'fixedSupportRefine', struct());
			fixedEnable = logical(getfield_default_local(fixedCfg, 'enable', true));
			stage1FixedMaskCur = make_seed_support_mask_local(trialSpec.seedCoef, trainMask_cur);
			assert_seed_support_active_local(trialSpec.seedCoef, stage1FixedMaskCur);
			fixedActive = count_active_mask(stage1FixedMaskCur);
			fullActive = count_active_mask(trainMask_cur);
			if fixedActive <= 0
				error('Stage 1 fixed-support handoff produced an empty SR support mask.');
			end
			fprintf('Stage 1 fixed SR-support mask: active = %d / %d full augmented coefficients.\n', ...
				fixedActive, fullActive);

			if fixedEnable
				tStage1Fixed = tic;
				thetaSeedFixed = pack_Coef_M_by_mask(trialSpec.seedCoef, stage1FixedMaskCur);
				optsStage1Fixed = opts_cur;
				optsStage1Fixed.init.mode = 'skip';
				optsStage1Fixed.init.seed.mode = 'zero';
				optsStage1Fixed.init.seed.externalTheta = thetaSeedFixed(:);
				optsStage1Fixed.init.seed.externalThetaSource = 'stage0_exact_sr_support_seed';
				optsStage1Fixed.init.seed.prependExternalTheta = true;
				optsStage1Fixed.init.seed.forceExternalThetaOnly = true;
				optsStage1Fixed.init.seed.numCandidates = 1;
				optsStage1Fixed.init.lsq.enable = true;
				optsStage1Fixed.init.lsq.numStarts = 1;
				optsStage1Fixed.init.lsq.maxIter = max(1, round(getfield_default_local( ...
					fixedCfg, 'maxIter', opts_cur.init.lsq.maxIter)));
				optsStage1Fixed.init.lsq.maxFunEvals = max(1, round(getfield_default_local( ...
					fixedCfg, 'maxFunEvals', opts_cur.init.lsq.maxFunEvals)));
				optsStage1Fixed.init.lsq.useAnalyticJacobian = logical(getfield_default_local( ...
					fixedCfg, 'useAnalyticJacobian', opts_cur.init.lsq.useAnalyticJacobian));
				optsStage1Fixed.init.lsq.acceptByValidation = logical(getfield_default_local( ...
					fixedCfg, 'acceptByValidation', true));
				optsStage1Fixed.init.lsq.maxRelValIncrease = getfield_default_local( ...
					fixedCfg, 'maxRelValIncrease', 0);
				if isfield(optsStage1Fixed.init, 'ga')
					optsStage1Fixed.init.ga.enable = false;
				end
				if isfield(optsStage1Fixed.init, 'multiStart')
					optsStage1Fixed.init.multiStart.enable = false;
				end
				if isfield(optsStage1Fixed.init, 'skip')
					optsStage1Fixed.init.skip.useExternalStage0Seed = true;
					optsStage1Fixed.init.skip.useBaselineScreen = false;
				end
				% The handoff stage polishes coefficients only; it must not alter support.
				optsStage1Fixed.init.postBPPrune.enable = false;
				optsStage1Fixed.init.postBPPrune.numIterations = 0;
				if logical(getfield_default_local(opts_cur.stage1, 'expandBoundsToIncludeStage0Seed', true))
					margin = getfield_default_local(opts_cur.stage1, 'stage0SeedBoundMargin', 1e-6);
					[optsStage1Fixed.init.bounds.lower, optsStage1Fixed.init.bounds.upper] = ...
						expand_bounds_to_include_seed_local(optsStage1Fixed.init.bounds.lower, ...
							optsStage1Fixed.init.bounds.upper, thetaSeedFixed(:), margin);
				end

				Ystage0Tr = model_forward(Xtr, trialSpec.seedCoef, arch_cur, normOpt);
				Ystage0Val = model_forward(Xval, trialSpec.seedCoef, arch_cur, normOpt);
				stage0TrainMetrics = compute_regression_metrics(Ystage0Tr, Ytr);
				stage0ValMetrics = compute_regression_metrics(Ystage0Val, Yval);

				[thetaStage1Fixed, fixedStats, CoefStage1Fixed, maskStage1Fixed] = masked_lsq_initialize( ...
					Coef_template_cur, stage1FixedMaskCur, Xtr, Ytr, Xval, Yval, ...
					arch_cur, normOpt, optsStage1Fixed);
				if ~masks_equal_local(maskStage1Fixed, stage1FixedMaskCur)
					error('Stage 1 fixed-support BP unexpectedly changed the SR support mask.');
				end
				YfixedTr = model_forward(Xtr, CoefStage1Fixed, arch_cur, normOpt);
				YfixedVal = model_forward(Xval, CoefStage1Fixed, arch_cur, normOpt);
				fixedTrainMetrics = compute_regression_metrics(YfixedTr, Ytr);
				fixedValMetrics = compute_regression_metrics(YfixedVal, Yval);
				stage1FixedTime = toc(tStage1Fixed);

				stage1StatsCur.fixedSupportRefinementEnabled = true;
				stage1StatsCur.fixedSupportActive = fixedActive;
				stage1StatsCur.fullAugmentedActive = fullActive;
				stage1StatsCur.seedTrainMSE = stage0TrainMetrics.mse;
				stage1StatsCur.seedValMSE = stage0ValMetrics.mse;
				stage1StatsCur.refinedTrainMSE = fixedTrainMetrics.mse;
				stage1StatsCur.refinedValMSE = fixedValMetrics.mse;
				stage1StatsCur.fixedSupportTheta = thetaStage1Fixed;
				stage1StatsCur.fixedSupportStats = fixedStats;
				stage1StatsCur.stage1OnlyTime = stage1FixedTime;
				stage1StatsCur.elapsedTime = stage1FixedTime;
				handoffCoefCur = CoefStage1Fixed;

				fprintf(['Stage 1 fixed-support BP handoff complete: train/val MSE ', ...
					'%.6e/%.6e -> %.6e/%.6e | time %.3f s.\n'], ...
					stage0TrainMetrics.mse, stage0ValMetrics.mse, ...
					fixedTrainMetrics.mse, fixedValMetrics.mse, stage1FixedTime);
			else
				stage1StatsCur.fixedSupportActive = fixedActive;
				stage1StatsCur.fullAugmentedActive = fullActive;
				stage1StatsCur.reason = 'fixedSupportRefine.enable=false';
				handoffCoefCur = trialSpec.seedCoef;
				fprintf('Stage 1 fixed-support BP handoff disabled; releasing the exact Stage-0 seed directly.\n');
			end

			% --------------------------------------------------------------
			% Stage 2 full augmented-PhDN release.
			% --------------------------------------------------------------
			thetaHandoffFull = pack_Coef_M_by_mask(handoffCoefCur, trainMask_cur);
			if logical(getfield_default_local(opts_cur.stage1, 'expandBoundsToIncludeStage0Seed', true))
				margin = getfield_default_local(opts_cur.stage1, 'stage0SeedBoundMargin', 1e-6);
				[opts_cur.init.bounds.lower, opts_cur.init.bounds.upper] = ...
					expand_bounds_to_include_seed_local(opts_cur.init.bounds.lower, ...
						opts_cur.init.bounds.upper, thetaHandoffFull(:), margin);
			end
			opts_cur.init.seed.externalTheta = thetaHandoffFull(:);
			opts_cur.init.seed.externalThetaSource = 'stage1_fixed_sr_support_bp_handoff';
			opts_cur.init.seed.prependExternalTheta = true;
			opts_cur.init.seed.forceExternalThetaOnly = logical(getfield_default_local( ...
				opts_cur.stage1, 'forceStage0SeedOnly', true));
			opts_cur.init.seed.numCandidates = 1;
			opts_cur.init.lsq.numStarts = 1;
			fprintf(['Stage 2 full-network release: %d coefficients trainable; ', ...
				'non-SR augmentation channels start at zero.\n'], fullActive);
		end

		[theta_cur, initStatsCur, Coef_cur, trainMask_used] = masked_lsq_initialize( ...
			Coef_template_cur, trainMask_cur, Xtr, Ytr, Xval, Yval, arch_cur, normOpt, opts_cur);
		initTimeCur = toc(timerInit);
		initStatsCur.initializationTime = initTimeCur;
		initStatsCur.stage1 = stage1StatsCur;
		initStatsCur.opArgPolyOrder = opArgPolyOrder;
		initStatsCur.identityCancellationReport = identityCancellationReportCur;
		initStatsCur.inputDomainInverseReport = inputDomainInvReportCur;
		initStatsCur.rowwiseMaskInfo = rowwiseMaskInfoCur;
		initStatsCur.stage0ReproductionPrecheck = stage0ReproductionPrecheck;
		if isfield(trialSpec, 'isStage0Candidate') && trialSpec.isStage0Candidate
			initStatsCur.stage0Candidate = trialSpec;
		end

		% Exact Stage-0 reproduction was verified before BP/LSQ.

		Ytr_pred = model_forward(Xtr, Coef_cur, arch_cur, normOpt);
		Yval_pred = model_forward(Xval, Coef_cur, arch_cur, normOpt);
		trainMetrics = compute_regression_metrics(Ytr_pred, Ytr);
		valMetrics = compute_regression_metrics(Yval_pred, Yval);
		trainMSE = trainMetrics.mse;
		valMSE = valMetrics.mse;
		if ~isfinite(valMSE) || ~isfinite(trainMSE)
			valMSE = Inf;
		end
		trainValConsistent = true;
		if isStage0CandidateCur && logical(getfield_default_local(opts.stage0, 'enforceTrainValidationConsistency', true))
			maxRatio = getfield_default_local(opts.stage0, 'maxTrainValidationMSERatio', 20);
			trainValConsistent = trainMSE <= maxRatio * max(valMSE, eps);
			if ~trainValConsistent
				fprintf('Stage-0 candidate rejected from final selection: train/val MSE ratio %.3e exceeds %.3e.\n', ...
					trainMSE / max(valMSE, eps), maxRatio);
			end
		end

		timeStats.initStatsByOpArg(kop).opArgPolyOrder = opArgPolyOrder;
		timeStats.initStatsByOpArg(kop).initializationTime = initTimeCur;
		timeStats.initStatsByOpArg(kop).stats = initStatsCur;
		timeStats.initStatsByOpArg(kop).valMSE = valMSE;
		timeStats.initStatsByOpArg(kop).trainMSE = trainMSE;
		timeStats.initStatsByOpArg(kop).trainValConsistent = trainValConsistent;
		if isfield(trialSpec, 'isStage0Candidate') && trialSpec.isStage0Candidate
			timeStats.initStatsByOpArg(kop).stage0CandidateLabel = getfield_default_local(trialSpec, 'label', 'stage0_candidate');
			timeStats.initStatsByOpArg(kop).stage0Expressions = getfield_default_local(trialSpec, 'expressions', {});
		end

		fprintf('Masked-LSQ validation MSE = %.6e, active mask = %d\n', ...
			valMSE, count_active_mask(trainMask_used));

		if trainValConsistent && isfinite(valMSE) && valMSE < bestValMSE
			bestValMSE = valMSE;
			bestArch = arch_cur;
			bestCoefTemplate = Coef_template_cur;
			bestCoef = Coef_cur;
			bestTheta = theta_cur;
			bestInitStats = initStatsCur;
			bestTrainMask = trainMask_used;
			bestOpArgPolyOrder = opArgPolyOrder;
			bestOpts = opts_cur;
			bestTrialSpec = trialSpec;
		end
	end

	timeStats.modelSelectionTime = toc(timerModelSelection);

	if isempty(bestArch)
		error('No valid masked-LSQ model was found. Check mask, bounds, and initialization settings.');
	end

	fprintf('\n========================================\n');
	fprintf('Selected masked-LSQ opArgPolyOrder = %d\n', bestOpArgPolyOrder);
	fprintf('Selected Masked-LSQ validation MSE = %.6e\n', bestValMSE);
	selectedSkeletonSource = getfield_default_local(bestTrialSpec, 'skeletonSource', 'per_output_pysr');
	fprintf('Selected Stage-0 skeleton source = %s\n', selectedSkeletonSource);
	fprintf('========================================\n');

	timeStats.stage0SelectedSkeletonSource = selectedSkeletonSource;
	result = finalize_galsq_result_local( ...
		task, bestOpts, bestArch, bestCoefTemplate, bestCoef, bestTheta, bestTrainMask, ...
		bestInitStats, normOpt, Xtr, Ytr, Xval, Yval, Xte, Yte, ...
		Xood, Yood, oodDomain, timeStats, runTimerTotal, bestValMSE);
	result.stage0SelectedSkeletonSource = selectedSkeletonSource;
	result.stage0SelectedCandidateLabel = getfield_default_local(bestTrialSpec, 'label', '');
end



function [specs, info] = maybe_prepare_per_output_pysr_stage0_trial_specs_local(task, opts, archBase, normOpt, Xtr, Ytr, Xval, Yval, Xte, Yte, Xood, Yood, labelNoiseInfo)
%MAYBE_PREPARE_PER_OUTPUT_PYSR_STAGE0_TRIAL_SPECS_LOCAL Run Stage 0 and compile one augmented DAG.

    specs = struct('arch', {}, 'opArgPolyOrder', {}, 'isStage0Candidate', {}, ...
        'label', {}, 'expressions', {}, 'compiledTerms', {}, ...
        'seedCoef', {}, 'compileInfo', {}, 'stage0Model', {}, 'skeletonSource', {});
    info = struct('applied', false, 'reason', 'not_requested');
    if ~isfield(opts, 'stage0') || ~isstruct(opts.stage0) || ...
            ~logical(getfield_default_local(opts.stage0, 'enable', false))
        info.reason = 'stage0_enable_false';
        return;
    end

    fprintf('\n========================================\n');
    fprintf('PhDN Stage 0: per-output SINDy bypass and native PySR\n');
    fprintf('========================================\n');
    fprintf('The fixed SINDy dictionary is validated independently for every output.\n');
    fprintf('Only outputs that fail the SINDy threshold are searched by official PySR.\n');
    fprintf('Each PySR core is selected from the 4x-best-validation structure-score pool.\n');
    fprintf('Stage 1 recursively decomposes only these core trees into one shared compact DAG.\n');
    augmentationMode = lower(strtrim(char(getfield_default_local(opts.stage1,'augmentationMode','polynomial'))));
    if any(strcmp(augmentationMode,{'neural','neural_ridge','fixed_neural_ridge','fixed-neural-ridge'}))
        if logical(getfield_default_local(opts.stage1,'augmentationNeuralIncludeLinearTerms',false))
            fprintf(['Every active branch receives a constant, all branch-coordinate ', ...
                'linear terms, and %d fixed neural-ridge bases.\n'], ...
                getfield_default_local(opts.stage1,'augmentationNeuralCount',task.nx));
        else
            fprintf('Every active branch receives a constant plus %d fixed neural-ridge bases.\n', ...
                getfield_default_local(opts.stage1,'augmentationNeuralCount',task.nx));
        end
    else
        fprintf('Every active branch receives the same dimension-dependent constant+Poly_%d augmentation family.\n', ...
            opts.stage1.augmentationPolyOrder);
    end

    data = struct('Xtr',Xtr,'Ytr',Ytr,'Xval',Xval,'Yval',Yval,'Xte',Xte,'Yte',Yte, ...
        'Xood',Xood,'Yood',Yood);
    % Preserve the known derivative-label-noise protocol for the general
    % Stage-0 rescue controller.  The controller only uses this metadata to
    % raise its validation-quality thresholds above the expected noise floor;
    % clean test/OOD targets remain untouched.
    if nargin < 13 || ~isstruct(labelNoiseInfo)
        labelNoiseInfo = struct('enabled',false);
    end
    data.derivativeLabelNoise = labelNoiseInfo;
    result0 = run_phdn_per_output_pysr_stage0(task, archBase, data, opts.stage0);
    candidates = result0.candidates;
    if isempty(candidates)
        error('Per-output PySR Stage 0 produced no usable skeleton.');
    end

    if ~result0.usedSingleLayerBypass
        model = result0.bestModel;
        coreExpressions = result0.coreExpressions;
        compileCfg = struct();
        compileCfg.polyOrder = getfield_default_local(opts.stage1, 'augmentationPolyOrder', ...
            getfield_default_local(opts.stage0.baseDictionary, 'polyOrder', 2));
        compileCfg.enableAugmentation = logical(getfield_default_local(opts.stage1, 'enableAugmentation', true));
        compileCfg.augmentationIncludeCrossTerms = true;
        compileCfg.augmentationMode = getfield_default_local(opts.stage1,'augmentationMode','polynomial');
        compileCfg.neuralCount = getfield_default_local(opts.stage1,'augmentationNeuralCount',task.nx);
        compileCfg.neuralActivation = getfield_default_local(opts.stage1,'augmentationNeuralActivation','tanh');
        compileCfg.neuralQuantiles = getfield_default_local(opts.stage1,'augmentationNeuralQuantiles',[0.25,0.50,0.75]);
        compileCfg.neuralScales = getfield_default_local(opts.stage1,'augmentationNeuralScales',[0.5,1,2]);
        compileCfg.neuralPoolRatio = getfield_default_local(opts.stage1,'augmentationNeuralPoolRatio',3);
        compileCfg.neuralSeed = getfield_default_local(opts.stage1,'augmentationNeuralSeed',1701);
        compileCfg.neuralStdFloor = getfield_default_local(opts.stage1,'augmentationNeuralStdFloor',1e-10);
        compileCfg.neuralVarianceThreshold = getfield_default_local(opts.stage1,'augmentationNeuralVarianceThreshold',1e-8);
        compileCfg.neuralCorrelationThreshold = getfield_default_local(opts.stage1,'augmentationNeuralCorrelationThreshold',0.995);
        compileCfg.neuralEnsureFullDirectionalSpan = getfield_default_local(opts.stage1,'augmentationNeuralEnsureFullDirectionalSpan',true);
        compileCfg.neuralIncludeLinearTerms = getfield_default_local(opts.stage1,'augmentationNeuralIncludeLinearTerms',false);
        compileCfg.trainingInputs = Xtr;
        compileCfg.normOpt = normOpt;
        [coreExpressions, model, fallbackInfo] = ...
            select_compilable_stage0_core_local(coreExpressions, model, ...
            archBase, task, compileCfg, Ytr, Yval, Yte, Yood);
        [archCand, compileInfo] = compile_sr_skeleton_set_to_phdn_arch( ...
            coreExpressions, archBase, task, model, compileCfg);
        compileInfo.fallbackInfo = fallbackInfo;
        result0.bestModel = model;
        result0.coreExpressions = coreExpressions;
        result0.bestScoreExpressions = coreExpressions;
        result0.bestExpressions = coreExpressions; % backward-compatible alias

        specs(1).arch = archCand;
        specs(1).opArgPolyOrder = 1;
        specs(1).isStage0Candidate = true;
        specs(1).label = sprintf('stage0_per_output_selected_core_augmented_dag_L%d', compileInfo.nLayers);
        specs(1).skeletonSource = 'per_output_sindy_or_pysr_structure_score';
        specs(1).expressions = coreExpressions;
        specs(1).compiledTerms = compileInfo.compiledTerms;
        specs(1).compileInfo = compileInfo;
        specs(1).stage0Model = model;
        specs(1).seedCoef = archCand.stage0SeedCoef;

        fprintf('Stage 1 compiled the per-output structure-score cores into one augmented PhDN DAG:\n');
        fprintf('  layers=%d | hidden=[%s] | exact seed terms=%d\n', ...
            compileInfo.nLayers, num2str(compileInfo.hiddenDims), compileInfo.nExactActiveTerms);
        for k = 1:numel(coreExpressions)
            fprintf('  y%d structure-score core = %s\n', k, coreExpressions{k});
        end
    end

    info.applied = true;
    info.reason = result0.reason;
    info.result = result0;
    info.nCandidates = numel(specs);
    info.trainTime = result0.trainTime;
    info.searchTime = result0.searchTime;
    info.usedSingleLayerBypass = result0.usedSingleLayerBypass;
    info.usedPerOutputSindyBypass = getfield_default_local(result0,'usedPerOutputSindyBypass',false);
    info.bypassOutputMask = getfield_default_local(result0,'bypassOutputMask',false(1,task.ny));
    info.bestModel = result0.bestModel;
    info.coreExpressions = result0.coreExpressions;
    info.bestScoreExpressions = result0.coreExpressions;
    info.bestExpressions = result0.coreExpressions; % backward-compatible alias
    info.engine = 'per_output_sindy_bypass_plus_official_pysr_structure_score';
end

function [coreExpr, model, info] = select_compilable_stage0_core_local(coreExpr, model, archBase, task, compileCfg, Ytr, Yval, Yte, Yood)
    info = struct('usedFallback',false,'outputs',struct([]));
    selections = getfield_default_local(model, 'outputSelections', struct([]));
    for r = 1:task.ny
        ordered = collect_compile_candidates_local(coreExpr{r}, selections, r);
        [chosenCore, chosenMeta, coreRank, coreErrors] = first_compilable_candidate_local(ordered, archBase, task, compileCfg);
        if isempty(chosenCore)
            error('All Stage-0 core candidates for output y%d failed compact-DAG parsing. Errors: %s', ...
                r, strjoin(coreErrors, ' | '));
        end
        changed = ~strcmp(chosenCore, coreExpr{r});
        if changed
            primaryError = '<compiler did not return an error message>';
            if ~isempty(coreErrors); primaryError = coreErrors{1}; end
            globalRank = find_global_structure_rank_local(selections, r, chosenCore);
            if isfinite(globalRank)
                rankText = sprintf('global structure-score rank %d', globalRank);
            else
                rankText = sprintf('fallback-order rank %d', coreRank);
            end
            fprintf(['Stage-0 compile fallback for y%d: selected structure-score core failed. ', ...
                'Compiler detail: %s\n'], r, primaryError);
            fprintf('Stage-0 compile fallback for y%d: using %s: %s\n', ...
                r, rankText, chosenCore);
            info.usedFallback = true;
            [predTr, predVal, predTe, predOod] = load_candidate_predictions_local(chosenMeta, model, r);
            model.YhatTrain(:,r) = predTr;
            model.YhatVal(:,r) = predVal;
            model.YhatTest(:,r) = predTe;
            if ~isempty(predOod)
                model.YhatOod(:,r) = predOod;
            end
        end
        coreExpr{r} = chosenCore;
        if isstruct(selections) && numel(selections) >= r
            selections(r).core = chosenMeta;
            selections(r).bestScore = chosenMeta;
            selections(r).best = chosenMeta; % backward-compatible alias
        end
        info.outputs(r).coreRank = coreRank;
        info.outputs(r).globalStructureRank = find_global_structure_rank_local(selections, r, chosenCore);
        info.outputs(r).compilerErrors = coreErrors;
        info.outputs(r).usedFallback = changed;
    end
    model.outputExpressions = coreExpr;
    model.coreExpressions = coreExpr;
    model.bestScoreExpressions = coreExpr;
    model.bestExpressions = coreExpr; % backward-compatible alias
    model.outputSelections = selections;
    model.trainMetrics = compute_regression_metrics(model.YhatTrain, Ytr);
    model.valMetrics = compute_regression_metrics(model.YhatVal, Yval);
    model.testMetrics = compute_regression_metrics(model.YhatTest, Yte);
    if ~isempty(Yood) && isfield(model,'YhatOod') && ~isempty(model.YhatOod)
        model.oodMetrics = compute_regression_metrics(model.YhatOod, Yood);
    end
end

function rank = find_global_structure_rank_local(selections, r, expression)
    rank = NaN;
    if ~isstruct(selections) || numel(selections) < r || isempty(expression)
        return;
    end
    rows = getfield_default_local(selections(r), 'structureScoreRanking', struct([]));
    if ~isstruct(rows) || isempty(rows); return; end
    for k = 1:numel(rows)
        rowExpression = char(string(getfield_default_local(rows(k), 'expression', '')));
        if strcmp(rowExpression, char(expression))
            rank = getfield_default_local(rows(k), 'rank', k);
            if ~isscalar(rank) || ~isfinite(rank); rank = k; end
            return;
        end
    end
end

function ordered = collect_compile_candidates_local(primary, selections, r)
    ordered = struct('expression',{},'prediction_paths',{},'complexity',{},'validation_mse',{},'score',{});
    ordered = append_candidate_local(ordered, struct('expression',primary));
    if isstruct(selections) && numel(selections) >= r
        ordered = append_candidate_local(ordered, getfield_default_local(selections(r),'core',struct()));
        ordered = append_candidate_local(ordered, getfield_default_local(selections(r),'bestScore',struct()));
        C = getfield_default_local(selections(r),'candidates',struct([]));
        for k=1:numel(C); ordered = append_candidate_local(ordered,C(k)); end
    end
end

function ordered = append_candidate_local(ordered, c)
    % Prefer the round-trip-safe compiler serialization exported beside the
    % native display expression.  Older result files remain compatible.
    expr = getfield_default_local(c,'compiler_expression', ...
        getfield_default_local(c,'expression',''));
    if isempty(expr); return; end
    expr = char(expr);
    duplicateIndex = find(arrayfun(@(q) strcmp(char(q.expression),expr), ordered), 1);
    incomingPaths = getfield_default_local(c,'prediction_paths',struct());
    if ~isempty(duplicateIndex)
        if isstruct(incomingPaths) && ~isempty(fieldnames(incomingPaths)) && ...
                (~isstruct(ordered(duplicateIndex).prediction_paths) || isempty(fieldnames(ordered(duplicateIndex).prediction_paths)))
            ordered(duplicateIndex).prediction_paths = incomingPaths;
            ordered(duplicateIndex).complexity = getfield_default_local(c,'complexity',ordered(duplicateIndex).complexity);
            ordered(duplicateIndex).validation_mse = getfield_default_local(c,'validation_mse',ordered(duplicateIndex).validation_mse);
            ordered(duplicateIndex).score = getfield_default_local(c,'score',ordered(duplicateIndex).score);
        end
        return;
    end
    rec = struct('expression',expr,'prediction_paths',incomingPaths, ...
        'complexity',getfield_default_local(c,'complexity',NaN), ...
        'validation_mse',getfield_default_local(c,'validation_mse',NaN), ...
        'score',getfield_default_local(c,'score',NaN));
    ordered(end+1)=rec;
end

function [expr, meta, rank, errors] = first_compilable_candidate_local(ordered, archBase, task, compileCfg)
    expr=''; meta=struct(); rank=NaN; errors={};
    taskOne=task; taskOne.ny=1; taskOne.outputNames={'y'};
    archOne=archBase; archOne.ny=1;

    % A Stage-0 core must be accepted or rejected only by structural DAG
    % parsing. Data-aware Stage-1 augmentation is attached after all selected
    % cores have been compiled together. Coupling these checks can incorrectly
    % replace an accurate SR core merely because a low-dimensional branch does
    % not yet provide enough screened neural features.
    structureOnlyCfg = compileCfg;
    structureOnlyCfg.enableAugmentation = false;

    for k=1:numel(ordered)
        try
            compile_sr_skeleton_set_to_phdn_arch( ...
                {ordered(k).expression},archOne,taskOne,struct(),structureOnlyCfg);
            expr=ordered(k).expression; meta=ordered(k); rank=k; return;
        catch ME
            errors{end+1}=sprintf('candidate rank %d: %s',k,ME.message); %#ok<AGROW>
        end
    end
end

function [A,B,C,D] = load_candidate_predictions_local(meta, model, r)
    A=model.YhatTrain(:,r); B=model.YhatVal(:,r); C=model.YhatTest(:,r);
    if isfield(model,'YhatOod') && ~isempty(model.YhatOod); D=model.YhatOod(:,r); else; D=[]; end
    P=getfield_default_local(meta,'prediction_paths',struct());
    if ~isstruct(P) || isempty(fieldnames(P))
        error(['Stage-0 selected a fallback expression, but its exported prediction paths are unavailable. ' ...
            'Rerun Stage 0 with the updated PySR adapter so candidate predictions are exported.']);
    end
    A=read_candidate_matrix_local(getfield_default_local(P,'train',''));
    B=read_candidate_matrix_local(getfield_default_local(P,'validation',''));
    C=read_candidate_matrix_local(getfield_default_local(P,'test',''));
    po=getfield_default_local(P,'ood',''); if ~isempty(po) && exist(po,'file'); D=read_candidate_matrix_local(po); end
end

function x = read_candidate_matrix_local(path)
    if isempty(path) || ~exist(path,'file'); error('Stage-0 fallback prediction file is missing: %s',path); end
    if exist('readmatrix','file')==2; x=readmatrix(path); else; x=csvread(path); end
    x=x(:);
end

function result = finalize_sindy_stage0_bypass_result_local(task, opts, stage0Info, ...
	Xtr, Ytr, Xval, Yval, Xte, Yte, Xood, Yood, oodDomain, timeStats, runTimerTotal)
%FINALIZE_SINDY_STAGE0_BYPASS_RESULT_LOCAL Return the accepted SINDy model directly.

	model = stage0Info.bestModel;
	timeStats.totalWallTime = toc(runTimerTotal);
	timeStats.trainingWallTime = getfield_default_local(stage0Info, 'trainTime', 0);
	timeStats.modelSelectionTime = 0;
	timeStats.stage0StructuralTime = getfield_default_local(stage0Info, 'trainTime', 0);
	timeStats.stage1StructuralTime = 0;
	timeStats.stage2RefinementTime = 0;
	timeStats.seedCandidateTime = 0;
	timeStats.testEvaluationTime = 0;
	timeStats.oodEvaluationTime = 0;
	timeStats.symbolicDisplayTime = 0;
	timeStats.stage0ValidationMSE = model.valMetrics.mse;
	timeStats.stage0ValidationRMSE = model.valMetrics.rmse;
	timeStats.stage0IDTestMSE = model.testMetrics.mse;
	timeStats.stage0IDTestRMSE = model.testMetrics.rmse;
	if isfield(model, 'oodMetrics') && isstruct(model.oodMetrics)
		timeStats.stage0OODTestMSE = getfield_default_local(model.oodMetrics, 'mse', NaN);
		timeStats.stage0OODTestRMSE = getfield_default_local(model.oodMetrics, 'rmse', NaN);
	else
		timeStats.stage0OODTestMSE = NaN;
		timeStats.stage0OODTestRMSE = NaN;
	end

	result = struct();
	result.task = task;
	result.opts = opts;
	result.arch = [];
	result.bestArch = [];
	result.modelOperatorMode = 'SINDy single-layer bypass';
	result.Coef_template = [];
	result.Coef_M_est = [];
	result.theta_est = [];
	result.initStats = struct();
	result.surrogateOperatorReport = struct();
	result.pipeline = 'stage0_sindy_single_layer_bypass';
	result.mask_final = model.activeMask;
	result.bestValidationMSE = model.valMetrics.mse;
	result.bestOpArgPolyOrder = NaN;
	result.physicalTestMSE = model.testMetrics.mse;
	result.physicalTestRMSE = model.testMetrics.rmse;
	result.testMSE = model.testMetrics.mse;
	result.testRMSE = model.testMetrics.rmse;
	result.physicalTestMetrics = model.testMetrics;
	result.testMetrics = model.testMetrics;
	result.nActiveFinal = model.nActiveCoefficients;
	result.selectedTerms = [];
	result.timeStats = timeStats;
	result.stage0 = stage0Info;
	result.stage0Expressions = getfield_default_local(stage0Info, 'bestExpressions', {});
	result.data = struct('Xtr',Xtr,'Ytr',Ytr,'Xval',Xval,'Yval',Yval, ...
		'Xte',Xte,'Yte',Yte,'oodDomain',oodDomain,'Xood',Xood,'Yood',Yood);
	result.data.derivativeLabelNoise = getfield_default_local(timeStats,'derivativeLabelNoise',struct('enabled',false));
	result.prediction = struct();
	result.prediction.YtePhysical = model.prediction.Yte;
	result.prediction.YteFinal = model.prediction.Yte;
	result.prediction.YoodPhysical = getfield_default_local(model.prediction, 'Yood', []);
	result.prediction.YoodFinal = result.prediction.YoodPhysical;
	result.oodPhysicalTestMSE = timeStats.stage0OODTestMSE;
	result.oodPhysicalTestRMSE = timeStats.stage0OODTestRMSE;
	result.oodTestMSE = timeStats.stage0OODTestMSE;
	result.oodTestRMSE = timeStats.stage0OODTestRMSE;
	result.oodPhysicalTestMetrics = getfield_default_local(model, 'oodMetrics', struct());
	result.oodTestMetrics = result.oodPhysicalTestMetrics;
	result.symbolic = struct('identified', [], 'reference', [], ...
		'identifiedMessage', 'Final model is the accepted single-layer SINDy expression.', ...
		'referenceMessage', 'Skipped by Stage-0 SINDy bypass.');
	result.symbolic.identifiedExpressionStrings = result.stage0Expressions;

	fprintf('\n========================================\n');
	fprintf('Stage-0 SINDy mechanical-precision bypass accepted.\n');
	fprintf('Returning the SINDy model directly; PySR search, Stage 1, and Stage 2 are skipped.\n');
	fprintf('Validation MSE = %.6e, ID test MSE = %.6e\n', ...
		result.bestValidationMSE, result.testMSE);
	if isfinite(result.oodTestMSE)
		fprintf('OOD test MSE = %.6e\n', result.oodTestMSE);
	end
	fprintf('========================================\n');
end

function result = finalize_galsq_result_local( ...
	task, opts, arch, Coef_template, Coef_M_est, theta_est, trainMask, initStats, normOpt_final, ...
	Xtr, Ytr, Xval, Yval, Xte, Yte, Xood, Yood, oodDomain, timeStats, runTimerTotal, bestValMSE)

	modelLabel = get_operator_label_local(arch);

	fprintf('\n========================================\n');
	fprintf('Using the masked-LSQ refined model as the final %s PhDN.\n', modelLabel);
	fprintf('========================================\n');

	mask_final = trainMask;
	nActiveFinal = count_active_mask(mask_final);

	% In-distribution test.
	timerTest = tic;
	Yte_phys = model_forward(Xte, Coef_M_est, arch, normOpt_final);
	physicalTestMetrics = compute_regression_metrics(Yte_phys, Yte);
	physicalTestMSE = physicalTestMetrics.mse;
	physicalTestRMSE = physicalTestMetrics.rmse;
	timeStats.testEvaluationTime = toc(timerTest);

	fprintf('\nIn-distribution %s PhDN test MSE/RMSE = %.6e / %.6e\n', ...
		modelLabel, physicalTestMSE, physicalTestRMSE);

	% OOD test.
	Yood_phys = [];
	oodPhysicalTestMetrics = empty_metrics_local();
	oodPhysicalTestMSE = NaN;
	oodPhysicalTestRMSE = NaN;
	timeStats.oodEvaluationTime = 0;

	if opts.ood.enable
		timerOOD = tic;
		Yood_phys = model_forward(Xood, Coef_M_est, arch, normOpt_final);
		oodPhysicalTestMetrics = compute_regression_metrics(Yood_phys, Yood);
		oodPhysicalTestMSE = oodPhysicalTestMetrics.mse;
		oodPhysicalTestRMSE = oodPhysicalTestMetrics.rmse;
		timeStats.oodEvaluationTime = toc(timerOOD);

		if isfield(timeStats, 'oodDomainSummary') && ~isempty(timeStats.oodDomainSummary)
			fprintf('OOD domain intervals = %s\n', timeStats.oodDomainSummary);
		else
			fprintf('OOD domain lb/ub = [%s] / [%s]\n', ...
				num2str(timeStats.oodDomainLb, '%.4g '), num2str(timeStats.oodDomainUb, '%.4g '));
		end
		fprintf('OOD %s PhDN test MSE/RMSE = %.6e / %.6e\n', ...
			modelLabel, oodPhysicalTestMSE, oodPhysicalTestRMSE);
	end


	% No legacy surrogate-transfer diagnostic is used in the previous-version BSP-LSQ route.
	surrogateOperatorReport = struct('available', false, 'message', 'not_used_in_masked_lsq_route');

	% Symbolic display/check.
	timerSymbolic = tic;
	expr_est = [];
	expr_ref = [];
	symbolicInfo = struct();
	symbolicInfo.hasIdentified = false;
	symbolicInfo.hasReference = false;
	symbolicInfo.identifiedMessage = '';
	symbolicInfo.referenceMessage = '';
	symbolicInfo.skipReason = '';

	if isfield(opts.output, 'skipSymbolicDisplay') && opts.output.skipSymbolicDisplay
		symbolicInfo.identifiedMessage = 'Skipped by opts.output.skipSymbolicDisplay=true.';
		symbolicInfo.referenceMessage = 'Skipped by opts.output.skipSymbolicDisplay=true.';
		symbolicInfo.skipReason = 'skipSymbolicDisplay=true';
	else
		fprintf('Identified physical PhDN symbolic function:\n');
		if isfield(task, 'modelToSymbolicFcn') && ~isempty(task.modelToSymbolicFcn)
			try
				expr_est = call_symbolic_translator_local(task.modelToSymbolicFcn, task.nx, Coef_M_est, arch);
				try
					if ~isfield(opts.output, 'simplifySymbolic') || opts.output.simplifySymbolic
						expr_est = simplify(expr_est);
					end
				catch
				end
				disp(vpa(expr_est, opts.output.symbolicDigits));
				symbolicInfo.hasIdentified = true;
			catch ME
				symbolicInfo.identifiedMessage = ME.message;
				fprintf('Identified symbolic display skipped: %s\n', ME.message);
			end
		else
			symbolicInfo.identifiedMessage = 'No symbolic translator is available.';
			fprintf('Identified symbolic display skipped: no symbolic translator is available.\n');
		end

		if isfield(task, 'referenceSymbolicFcn') && ~isempty(task.referenceSymbolicFcn)
			try
				fprintf('Reference symbolic function:\n');
				expr_ref = task.referenceSymbolicFcn();
				disp(expr_ref);
				symbolicInfo.hasReference = true;
			catch ME
				symbolicInfo.referenceMessage = ME.message;
				fprintf('Reference symbolic display skipped: %s\n', ME.message);
			end
		end
	end
	timeStats.symbolicDisplayTime = toc(timerSymbolic);

	timeStats.totalWallTime = toc(runTimerTotal);
	timeStats.trainingWallTime = timeStats.modelSelectionTime;
	timeStats.initializationTime = getfield_default_local(initStats, 'elapsedTime', NaN);
	timeStats.gaTime = getfield_default_local(initStats, 'gaTime', NaN);
	timeStats.usedGA = getfield_default_local(initStats, 'usedGA', false);
	timeStats.seedSearchTime = getfield_default_local(initStats, 'seedSearchTime', NaN);
	timeStats.randomSearchTime = getfield_default_local(initStats, 'randomSearchTime', NaN);
	timeStats.bspTime = getfield_default_local(getfield_default_local(initStats, 'bsp', struct()), 'elapsedTime', 0);
	stageStats = getfield_default_local(initStats, 'stage1', struct());
	timeStats.stage0Time = 0;
	timeStats.stage1TotalTime = getfield_default_local(stageStats, 'elapsedTime', 0);
	timeStats.stage1Time = getfield_default_local(stageStats, 'stage1OnlyTime', 0);
	timeStats.legacyMicroPruningTime = 0;
	timeStats.structureSearchTime = timeStats.bspTime + timeStats.stage1Time;
	if isfield(timeStats, 'stage0') && isstruct(timeStats.stage0) && ...
			isfield(timeStats.stage0, 'applied') && timeStats.stage0.applied
		timeStats.stage0Time = getfield_default_local(timeStats.stage0, 'trainTime', 0);
		timeStats.stage1TotalTime = timeStats.stage0Time + timeStats.stage1Time;
		timeStats.structureSearchTime = timeStats.stage1TotalTime;
	end
	if isfield(timeStats, 'stage0') && isstruct(timeStats.stage0) && ...
			isfield(timeStats.stage0, 'bestModel') && isstruct(timeStats.stage0.bestModel)
		stage0Model = timeStats.stage0.bestModel;
		stage0Val = getfield_default_local(stage0Model, 'valMetrics', struct());
		stage0Test = getfield_default_local(stage0Model, 'testMetrics', struct());
		stage0Ood = getfield_default_local(stage0Model, 'oodMetrics', struct());
		timeStats.stage0ValidationMSE = getfield_default_local(stage0Val, 'mse', NaN);
		timeStats.stage0ValidationRMSE = getfield_default_local(stage0Val, 'rmse', NaN);
		timeStats.stage0IDTestMSE = getfield_default_local(stage0Test, 'mse', NaN);
		timeStats.stage0IDTestRMSE = getfield_default_local(stage0Test, 'rmse', NaN);
		timeStats.stage0OODTestMSE = getfield_default_local(stage0Ood, 'mse', NaN);
		timeStats.stage0OODTestRMSE = getfield_default_local(stage0Ood, 'rmse', NaN);
	else
		timeStats.stage0ValidationMSE = NaN;
		timeStats.stage0ValidationRMSE = NaN;
		timeStats.stage0IDTestMSE = NaN;
		timeStats.stage0IDTestRMSE = NaN;
		timeStats.stage0OODTestMSE = NaN;
		timeStats.stage0OODTestRMSE = NaN;
	end
	timeStats.layerwiseInitTime = timeStats.structureSearchTime;
	timeStats.lsqTime = getfield_default_local(initStats, 'lsqTime', NaN);
	timeStats.bestValidationMSE = bestValMSE;
	timeStats.finalActiveNumber = nActiveFinal;
	timeStats.modelLabel = modelLabel;
	timeStats.bestOpArgPolyOrder = get_oparg_order_local(arch);
	timeStats.physicalTestMSE = physicalTestMSE;
	timeStats.physicalTestRMSE = physicalTestRMSE;
	timeStats.oodPhysicalTestMSE = oodPhysicalTestMSE;
	timeStats.oodPhysicalTestRMSE = oodPhysicalTestRMSE;

	print_time_statistics_galsq_local(timeStats);

	selectedTerms = collect_selected_terms(mask_final, Coef_M_est, arch);

	if isfield(opts, 'output') && isfield(opts.output, 'printFinalXiMatrices') && opts.output.printFinalXiMatrices
		print_final_xi_matrices_local(Coef_M_est, arch, opts);
	end

	result = struct();
	result.task = task;
	result.opts = opts;
	result.arch = arch;
	result.bestArch = arch;
	result.modelOperatorMode = modelLabel;
	result.Coef_template = Coef_template;
	result.Coef_M_est = Coef_M_est;
	result.theta_est = theta_est;
	result.initStats = initStats;
	result.stage1FixedSupportHandoff = getfield_default_local(initStats, 'stage1', struct());
	result.surrogateOperatorReport = surrogateOperatorReport;
	result.pipeline = 'masked_lsq';
	result.mask_final = mask_final;
	result.bestValidationMSE = bestValMSE;
	result.bestOpArgPolyOrder = get_oparg_order_local(arch);
	result.physicalTestMSE = physicalTestMSE;
	result.physicalTestRMSE = physicalTestRMSE;
	result.testMSE = physicalTestMSE;
	result.testRMSE = physicalTestRMSE;
	result.physicalTestMetrics = physicalTestMetrics;
	result.testMetrics = physicalTestMetrics;
	result.nActiveFinal = nActiveFinal;
	result.selectedTerms = selectedTerms;
	result.timeStats = timeStats;
	result.stage0 = getfield_default_local(timeStats, 'stage0', struct());
	result.stage0Expressions = getfield_default_local(result.stage0, 'bestExpressions', {});
	result.data.Xtr = Xtr;
	result.data.Ytr = Ytr;
	result.data.Xval = Xval;
	result.data.Yval = Yval;
	result.data.Xte = Xte;
	result.data.Yte = Yte;
	result.data.oodDomain = oodDomain;
	result.data.Xood = Xood;
	result.data.Yood = Yood;
	result.data.derivativeLabelNoise = getfield_default_local(timeStats,'derivativeLabelNoise',struct('enabled',false));
	result.prediction.YtePhysical = Yte_phys;
	result.prediction.YteFinal = Yte_phys;
	result.prediction.YoodPhysical = Yood_phys;
	result.prediction.YoodFinal = Yood_phys;
	if isfield(surrogateOperatorReport, 'available') && surrogateOperatorReport.available
		result.prediction.YteSurrogateOperator = surrogateOperatorReport.Yte;
		if isfield(surrogateOperatorReport, 'Yood')
			result.prediction.YoodSurrogateOperator = surrogateOperatorReport.Yood;
		end
	end
	result.oodPhysicalTestMSE = oodPhysicalTestMSE;
	result.oodPhysicalTestRMSE = oodPhysicalTestRMSE;
	result.oodTestMSE = oodPhysicalTestMSE;
	result.oodTestRMSE = oodPhysicalTestRMSE;
	result.oodPhysicalTestMetrics = oodPhysicalTestMetrics;
	result.oodTestMetrics = oodPhysicalTestMetrics;
	result.symbolic = symbolicInfo;
	result.symbolic.identified = expr_est;
	result.symbolic.reference = expr_ref;
end



function print_final_xi_matrices_local(Coef_M, arch, opts)
	fprintf('\n========================================\n');
	fprintf('Final Xi coefficient matrices (debug)\n');
	fprintf('========================================\n');
	dims = get_arch_dims(arch);
	prec = getfield_default_local(opts.output, 'finalXiPrintPrecision', 4);
	onlyActive = getfield_default_local(opts.output, 'finalXiPrintOnlyActive', false);
	activeTol = getfield_default_local(opts.output, 'finalXiActiveTol', 1e-10);
	fmt = sprintf('%%+.%de', prec);
	for ell = 1:arch.layer
		for src = 1:ell
			if isempty(Coef_M{src, ell})
				continue;
			end
			k = ell - src + 1;
			inputDim = dims(k);
			if k == 1
				prefix = 'x';
			else
				prefix = 'h';
			end
			terms = branch_dictionary_terms(inputDim, arch, prefix, ell);
			C = Coef_M{src, ell};
			fprintf('\nXi{%d,%d}: size %d x %d, input state %d (%s, dim=%d)\n', ...
				src, ell, size(C, 1), size(C, 2), k, prefix, inputDim);
			fprintf('  term order:\n');
			for c = 1:min(size(C,2), numel(terms))
				if onlyActive && all(abs(C(:, c)) <= activeTol)
					continue;
				end
				fprintf('    %4d: %s\n', c, terms(c).name);
			end
			fprintf('  coefficients by row:\n');
			for r = 1:size(C, 1)
				fprintf('    row %d:', r);
				for c = 1:size(C, 2)
					if onlyActive && abs(C(r,c)) <= activeTol
						continue;
					end
					fprintf(' %s', sprintf(fmt, C(r, c)));
				end
				fprintf('\n');
			end
		end
	end
	fprintf('========================================\n');
end

function tf = is_true_operator_arch_local(arch)
	modeType = 'true';
	if isfield(arch, 'operatorMode') && ~isempty(arch.operatorMode)
		if ischar(arch.operatorMode) || isstring(arch.operatorMode)
			modeType = lower(strtrim(char(arch.operatorMode)));
		elseif isstruct(arch.operatorMode)
			if isfield(arch.operatorMode, 'type') && ~isempty(arch.operatorMode.type)
				modeType = lower(strtrim(char(arch.operatorMode.type)));
			elseif isfield(arch.operatorMode, 'mode') && ~isempty(arch.operatorMode.mode)
				modeType = lower(strtrim(char(arch.operatorMode.mode)));
			end
		end
	end
	tf = any(strcmp(modeType, {'true', 'safe', 'physical'}));
end

function label = get_operator_label_local(arch)
	if is_true_operator_arch_local(arch)
		label = 'true-operator';
	else
		label = 'surrogate-operator';
	end
end

function print_time_statistics_galsq_local(timeStats)
	fprintf('\n========================================\n');
	fprintf('Runtime and training-cost statistics\n');
	fprintf('========================================\n');
	fprintf('Task                                      : %s\n', timeStats.taskName);
	fprintf('Total wall time                          : %.3f s\n', timeStats.totalWallTime);
	fprintf('Training wall time masked-LSQ            : %.3f s\n', timeStats.trainingWallTime);
	if isfield(timeStats, 'domainSummary') && ~isempty(timeStats.domainSummary)
		fprintf('In-distribution domain intervals         : %s\n', timeStats.domainSummary);
	end
	fprintf('Data generation time                     : %.3f s\n', timeStats.dataGenerationTime);
	if isfield(timeStats, 'oodEnabled') && timeStats.oodEnabled
		fprintf('OOD data generation time                 : %.3f s for %d samples\n', ...
			timeStats.oodDataGenerationTime, timeStats.nOODSamples);
	end
	fprintf('Split/normalization time                 : %.3f s\n', timeStats.splitAndNormTime);
	seedTime = getfield_default_local(timeStats, 'seedSearchTime', getfield_default_local(timeStats, 'randomSearchTime', 0));
	gaTime = getfield_default_local(timeStats, 'gaTime', 0);
	structureSearchTime = getfield_default_local(timeStats, 'structureSearchTime', getfield_default_local(timeStats, 'layerwiseInitTime', 0));
	usedGA = getfield_default_local(timeStats, 'usedGA', gaTime > 0);
	fprintf('Seed candidate time                      : %.3f s\n', seedTime);
	if usedGA || gaTime > 0
		fprintf('GA search time                          : %.3f s\n', gaTime);
	end
	stage0Time = getfield_default_local(timeStats, 'stage0Time', 0);
	stage1Time = getfield_default_local(timeStats, 'stage1Time', 0);
	legacyMicroTime = getfield_default_local(timeStats, 'legacyMicroPruningTime', 0);
	stage1TotalTime = getfield_default_local(timeStats, 'stage1TotalTime', stage0Time + stage1Time + legacyMicroTime);
	if isfield(timeStats, 'stage0') && isstruct(timeStats.stage0) && ...
			isfield(timeStats.stage0, 'applied') && timeStats.stage0.applied
		fprintf('Per-output native PySR Stage-0 time                 : %.3f s\n', stage0Time);
		stage0ValMSE = getfield_default_local(timeStats, 'stage0ValidationMSE', NaN);
		stage0ValRMSE = getfield_default_local(timeStats, 'stage0ValidationRMSE', NaN);
		stage0IDMSE = getfield_default_local(timeStats, 'stage0IDTestMSE', NaN);
		stage0IDRMSE = getfield_default_local(timeStats, 'stage0IDTestRMSE', NaN);
		stage0OODMSE = getfield_default_local(timeStats, 'stage0OODTestMSE', NaN);
		stage0OODRMSE = getfield_default_local(timeStats, 'stage0OODTestRMSE', NaN);
		if isfinite(stage0ValMSE) || isfinite(stage0ValRMSE)
			fprintf('Stage-0 validation MSE / RMSE        : %.6e / %.6e\n', stage0ValMSE, stage0ValRMSE);
		end
		if isfinite(stage0IDMSE) || isfinite(stage0IDRMSE)
			fprintf('Stage-0 ID test MSE / RMSE           : %.6e / %.6e\n', stage0IDMSE, stage0IDRMSE);
		end
		if isfinite(stage0OODMSE) || isfinite(stage0OODRMSE)
			fprintf('Stage-0 OOD test MSE / RMSE          : %.6e / %.6e\n', stage0OODMSE, stage0OODRMSE);
		end
	elseif stage0Time > 0
		fprintf('Stage-0 structure initialization time    : %.3f s\n', stage0Time);
	end
	if stage1Time > 1e-9
		fprintf('Stage-1 fixed-support BP handoff time    : %.3f s\n', stage1Time);
	end
	if legacyMicroTime > 1e-9
		fprintf('Additional pruning/refinement time       : %.3f s\n', legacyMicroTime);
	end
	if stage1TotalTime > 0 && (stage0Time > 0 || stage1Time > 0)
		fprintf('Stage-0 + Stage-1 handoff time           : %.3f s\n', stage1TotalTime);
	end
	if structureSearchTime > 0 && abs(structureSearchTime - stage0Time - stage1Time - legacyMicroTime) > 1e-12
		fprintf('BSP-LSQ pruning search time              : %.3f s\n', structureSearchTime - stage0Time - stage1Time - legacyMicroTime);
	end
	fprintf('Stage-2 LSQ-BP refinement time           : %.3f s\n', timeStats.lsqTime);
	fprintf('In-distribution test evaluation time     : %.3f s\n', timeStats.testEvaluationTime);
	if isfield(timeStats, 'oodEnabled') && timeStats.oodEnabled
		fprintf('OOD test evaluation time                 : %.3f s\n', timeStats.oodEvaluationTime);
	end
	fprintf('Symbolic display time                    : %.3f s\n', timeStats.symbolicDisplayTime);
	fprintf('\nBest validation MSE                      : %.6e\n', timeStats.bestValidationMSE);
	modelLabel = getfield_default_local(timeStats, 'modelLabel', 'final-operator');
	fprintf('In-dist %s PhDN MSE / RMSE         : %.6e / %.6e\n', ...
		modelLabel, timeStats.physicalTestMSE, timeStats.physicalTestRMSE);
	if isfield(timeStats, 'oodEnabled') && timeStats.oodEnabled
		fprintf('OOD %s PhDN MSE / RMSE             : %.6e / %.6e\n', ...
			modelLabel, timeStats.oodPhysicalTestMSE, timeStats.oodPhysicalTestRMSE);
	end
	fprintf('Best opArgPolyOrder                      : %d\n', timeStats.bestOpArgPolyOrder);
	fprintf('Final active coefficient number          : %d\n', timeStats.finalActiveNumber);
	fprintf('========================================\n');
end


function supportMask = make_seed_support_mask_local(seedCoef, allowedMask)
%MAKE_SEED_SUPPORT_MASK_LOCAL Exact nonzero support of the compiled SR seed.
% The returned mask has the same block shapes as the full augmented-network
% mask, but only coefficients explicitly used by the Stage-0 expression are
% active.  Zero-initialized constant+Poly augmentation channels remain frozen
% until the Stage-2 full-network release.
	supportMask = cell(size(allowedMask));
	for ell = 1:size(allowedMask, 2)
		for src = 1:size(allowedMask, 1)
			if isempty(allowedMask{src,ell})
				supportMask{src,ell} = allowedMask{src,ell};
				continue;
			end
			allowed = logical(allowedMask{src,ell});
			support = false(size(allowed));
			if src <= size(seedCoef,1) && ell <= size(seedCoef,2) && ...
					~isempty(seedCoef{src,ell})
				A = seedCoef{src,ell};
				if ~isequal(size(A), size(allowed))
					error('Stage-0 seed/full-mask size mismatch at block (%d,%d): seed [%s], mask [%s].', ...
						src, ell, num2str(size(A)), num2str(size(allowed)));
				end
				if any(~isfinite(A(:)))
					error('Stage-0 seed contains nonfinite coefficients at block (%d,%d).', src, ell);
				end
				support = (A ~= 0);
				if any(support(:) & ~allowed(:))
					error('Stage-0 seed support lies outside the full augmented PhDN mask at block (%d,%d).', src, ell);
				end
			end
			supportMask{src,ell} = support & allowed;
		end
	end
end

function tf = masks_equal_local(A, B)
%MASKS_EQUAL_LOCAL Exact blockwise logical-mask comparison.
	if ~iscell(A) || ~iscell(B) || ~isequal(size(A), size(B))
		tf = false;
		return;
	end
	tf = true;
	for k = 1:numel(A)
		if isempty(A{k}) && isempty(B{k})
			continue;
		end
		if isempty(A{k}) || isempty(B{k}) || ...
				~isequal(size(A{k}), size(B{k})) || ...
				~isequal(logical(A{k}), logical(B{k}))
			tf = false;
			return;
		end
	end
end

function assert_seed_support_active_local(seedCoef, trainMask)
	for ell = 1:size(seedCoef, 2)
		for src = 1:ell
			if src > size(seedCoef,1) || isempty(seedCoef{src,ell})
				continue;
			end
			A = seedCoef{src,ell};
			if isempty(A)
				continue;
			end
			if src > size(trainMask,1) || ell > size(trainMask,2) || isempty(trainMask{src,ell})
				error('Stage-0 seed support has block (%d,%d), but the train mask is missing this block.', src, ell);
			end
			M = logical(trainMask{src,ell});
			if ~isequal(size(A), size(M))
				error('Stage-0 seed/train-mask size mismatch at block (%d,%d): seed [%s], mask [%s].', ...
					src, ell, num2str(size(A)), num2str(size(M)));
			end
			nz = abs(A) > 0;
			if any(nz(:) & ~M(:))
				error('Stage-0 seed has nonzero coefficients outside the active SR-augmented DAG train mask at block (%d,%d).', src, ell);
			end
		end
	end
end

function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end

function metrics = empty_metrics_local()
	metrics = struct();
	metrics.mse = NaN;
	metrics.rmse = NaN;
	metrics.mae = NaN;
	metrics.nrmse = NaN;
	metrics.nmae = NaN;
end

function expr = call_symbolic_translator_local(translatorFcn, nx, Coef_M, arch)
	try
		expr = translatorFcn(nx, Coef_M, arch);
	catch
		expr = translatorFcn(nx, Coef_M, arch.layer, arch.polyOrder);
	end
end


function val = get_oparg_order_local(arch)
	if isfield(arch, 'interact') && isfield(arch.interact, 'opArgPolyOrder') && ~isempty(arch.interact.opArgPolyOrder)
		val = arch.interact.opArgPolyOrder;
	elseif isfield(arch, 'polyOrder') && ~isempty(arch.polyOrder)
		val = arch.polyOrder;
	else
		val = NaN;
	end
end


function [lb, ub] = expand_bounds_to_include_seed_local(lbIn, ubIn, thetaSeed, margin)
%EXPAND_BOUNDS_TO_INCLUDE_SEED_LOCAL Preserve an exact external SR seed.
% Scalar bounds are retained because later pruning changes the active-vector
% length. The interval is enlarged only when the SR seed lies outside it.
    thetaSeed = double(thetaSeed(:));
    lb = min(double(lbIn(:)));
    ub = max(double(ubIn(:)));
    if ~isfinite(lb) || ~isfinite(ub) || lb >= ub
        error('Invalid Stage-1 coefficient bounds.');
    end
    margin = max(0, double(margin));
    adaptiveMargin = max(margin, 1e-8 .* max(1, abs(thetaSeed)));
    if any(thetaSeed <= lb)
        lb = min(thetaSeed - adaptiveMargin);
    end
    if any(thetaSeed >= ub)
        ub = max(thetaSeed + adaptiveMargin);
    end
    if lb >= ub
        error('Unable to expand Stage-1 bounds around the Stage-0 SR seed.');
    end
end
