function opts = phdnn_default_options(task)
%PHDNN_DEFAULT_OPTIONS PhDN recovery with native per-output PySR Stage 0.
%
% Active initialization modes:
%   auto
%       strong_prior / weak_strong_prior -> ga_lsq
%       general / weak prior levels       -> mlp_recovery
%
%   ga_lsq
%       Stage 1 uses GA/BP coefficient initialization under the selected mask.
%
%   mlp_recovery
%       Stage 1 may use an MLP-surrogate hidden-node/coefficient initialization,
%       then Stage 1I exact recovery is restricted by the selected mask.
%
%   skip
%       Stage 1 is skipped. If Stage 1 selects a mask, its light-BP coefficient
%       seed is reused for final BP-LSQ. If Stage 1 is bypassed, a cheap
%       allowed-mask light-BP multi-start screen selects the final-BP seed.

	if nargin < 1 || isempty(task)
		task = struct('name', 'phdn_task');
	end

	opts = struct();

	% Architecture override settings.
	opts.arch = struct('hiddenDims', [], 'hiddenWidth', [], 'dims', []);

	% Data split.
	opts.data = struct();
	opts.data.nSamples = get_task_data_default_local(task, 'nSamples', 2000);
	opts.data.ratioTrain = get_task_data_default_local(task, 'ratioTrain', 0.6);
	opts.data.ratioVal = get_task_data_default_local(task, 'ratioVal', 0.2);
	% Optional post-split derivative-label perturbation. Disabled by default;
	% targeted robustness demos may enable it without changing the task RHS or
	% contaminating the clean ID/OOD test targets.
	opts.data.derivativeLabelNoise = struct('enable',false,'relativeStd',0, ...
		'seed',1,'applyToTrain',true,'applyToValidation',true, ...
		'scaleMode','training_derivative_std','distribution','gaussian');

	% OOD set is generated only for final reporting, never for training/refinement.
	opts.ood = struct();
	opts.ood.enable = true;
	opts.ood.nSamples = get_task_data_default_local(task, 'nOODSamples', 400);
	opts.ood.autoMode = 'upper';
	opts.ood.autoGapRatio = 0.05;
	opts.ood.autoWidthRatio = 0.25;
	opts.ood.useTaskOodDomain = true;

	% Training / mask settings shared by the protected GA route.
	opts.training = struct();
	opts.training.lambda1List = 0;
	opts.training.lambda2 = 0;
	opts.training.epsSmoothL1 = 1e-8;
	opts.training.useAdmissibleMask = false;
	opts.training.admissibleA = {};
	opts.training.admissibleSpec = {};
	opts.training.admissibleDefaultAllowed = true;
	opts.training.dictionarySupportA = {};
	opts.training.dictionarySupportMode = '';
	opts.training.uniformCompactPriorDictionaryActive = false;
	opts.training.removeIdentityCancellationTerms = true;
	opts.training.identityCancellationVerbose = true;
	opts.training.opArgPolyOrderList = 1;

	% Operator feasibility fields retained for exact-operator safety filters.
	opts.training.useOperatorFeasibilityPenalty = false;
	opts.training.operatorFeasibilityPenaltyWeight = 1e-2;
	opts.training.operatorZeroMargin = 1e-4;
	opts.training.operatorPositiveMargin = 1e-4;
	opts.training.operatorDomainMargin = 1e-6;
	opts.training.operatorExpSoftLimit = 50;
	opts.training.operatorTanCosMargin = 1e-3;
	opts.training.operatorUsageEps = 1e-10;
	opts.training.operatorPenaltyVerbose = false;
	opts.training.useSymbolicFeasibility = false;
	opts.training.symbolicFeasibilityMode = 'off';

	% Normalization and numerical safety.
	opts.norm = struct();
	opts.norm.useInputOutputNorm = false;       % enable train-set-fitted IO normalization for all routes by default
	opts.norm.applyToMLPSurrogate = false;      % previous MLP-surrogate route MLP-surrogate route uses NN-toolbox-style IO normalization by default
	opts.norm.useLayerNorm = false;
	opts.norm.style = 'mapminmax';             % MATLAB feedforwardnet-style preprocessing
	opts.norm.ymin = -1;
	opts.norm.ymax = 1;
	opts.norm.epsNorm = 1e-12;
	opts.safety = struct();
	opts.safety.eps = 1e-8;
	opts.safety.tolInv = 1e-12;
	opts.safety.tolLog = 1e-12;
	opts.safety.tolSqrt = 1e-12;
	opts.safety.tolDomain = 1e-12;
	opts.safety.tolTan = 1e-10;
	opts.safety.expClip = 300;
	opts.safety.trigClip = 1e3;
	opts.safety.hyperClip = 50;
	opts.safety.polyInputClip = 500;
	opts.safety.hiddenLayerOutputClip = 1e4;
	opts.safety.finalOutputClip = 1e6;

	% Initialization mode. v63 default is skip: Stage 1 micro-pruning supplies the mask/seed, then Stage 2 LSQ-BP refines it.
	opts.init = struct();
	opts.init.method = 'masked_lsq';
	opts.init.mode = 'skip';
	opts.init.errorOnEmptyMask = true;
	opts.init.bounds = struct('lower', -3, 'upper', 3);

	% Protected GA-LSQ controls for strong/weak-strong prior only.
	opts.init.ga = struct();
	opts.init.ga.enable = true;
	opts.init.ga.autoConfigureEffort = true;
	opts.init.ga.effort = 'quick';
	opts.init.ga.populationSize = 40;
	opts.init.ga.maxGenerations = 20;
	opts.init.ga.maxStallGenerations = 8;
	opts.init.ga.respectAutoCaps = false;
	opts.init.ga.maxPopulationSizeAuto = Inf;
	opts.init.ga.maxGenerationsAuto = Inf;
	opts.init.ga.maxEvalBudgetAuto = Inf;
	opts.init.ga.targetETPD = [];
	opts.init.ga.generationPopulationRatio = 1/3;
	opts.init.ga.functionTolerance = 1e-8;
	opts.init.ga.display = 'off';
	opts.init.ga.useParallel = false;

	% Minimal seed/LSQ fields required by the protected GA route.
	opts.init.seed = struct();
	opts.init.seed.mode = 'scale_aware';
	opts.init.seed.numCandidates = 40;
	opts.init.seed.rngSeed = 1;
	opts.init.seed.useScaleAware = true;
	opts.init.seed.scaleAwareAlphaList = [0.2, 0.5, 1.0];
	opts.init.seed.scaleAwareHiddenTargetMode = 'clipped_output_std';
	opts.init.seed.scaleAwareOutputTargetMode = 'output_std';
	opts.init.seed.scaleAwareHiddenTargetStdCap = 1.0;
	opts.init.seed.scaleAwareTargetStdFloor = 1e-6;
	opts.init.seed.scaleAwareFeatureStdFloor = 1e-8;
	opts.init.seed.scaleAwareConstantStdFactor = 1.0;
	opts.init.seed.scaleAwareCalibrateLayerOutput = true;
	opts.init.seed.scaleAwareCalibrationFactor = 0.5;
	opts.init.seed.scaleAwareLayerScaleMin = 0.1;
	opts.init.seed.scaleAwareLayerScaleMax = 10;
	opts.init.seed.scaleAwareRemoveInvalidRows = true;
	opts.init.seed.scaleAwarePostJitter = 0.02;

	% Stage 2 final LSQ-BP refinement switch. It is enabled by default after
	% the Stage-1 medium-BP selection; post-BP pruning remains disabled unless
	% explicitly enabled by an experiment script.
	opts.stage2 = struct();
	opts.stage2.enable = true;
	opts.init.lsq = struct();
	opts.init.lsq.enable = opts.stage2.enable;
	opts.init.lsq.numStarts = 1;
	opts.init.lsq.optimizer = 'lsqnonlin';
	opts.init.lsq.maxIter = 500;
	opts.init.lsq.maxFunEvals = 5e4;
	opts.init.lsq.display = 'off';
	opts.init.lsq.stepTolerance = 1e-10;
	opts.init.lsq.optimalityTolerance = 1e-8;
	opts.init.lsq.acceptByValidation = true;
	opts.init.lsq.maxRelValIncrease = 1e-10;
	opts.init.lsq.useAnalyticJacobian = true;
	opts.init.lsq.invalidValThreshold = 1e8;
	% Exact dictionary-PhDN Jacobian diagnostics. Enable this only for debugging
	% GA/BP exact-recovery failures such as no LSQ improvement after a good GA seed.
	opts.init.lsq.debugJacobianCheck = false;
	opts.init.lsq.debugJacobianNumColumns = 12;
	opts.init.lsq.debugJacobianNumSamples = 40;
	opts.init.lsq.debugJacobianRelTolerance = 1e-3;
	opts.init.lsq.debugJacobianAbsTolerance = 1e-6;
	opts.init.lsq.debugJacobianEpsilon = 1e-6;
	opts.init.lsq.fallbackToFiniteDifferenceOnBadJacobian = false;
	opts.init.lsq.errorOnBadJacobian = false;
	opts.init.lsq.earlyStop = struct('enable', true, 'chunkMaxIter', 50, 'chunkMaxFunEvals', [], ...
		'maxChunks', [], 'minChunks', 1, 'valPatience', 3, 'relImproveTol', 1e-4, ...
		'absImproveTol', 0, 'restoreBestValidation', true, 'stopOnSolverConvergence', true, ...
		'stopOnInvalidVal', true, 'verbose', false);

	% Post-final-BP pruning/refinement, kept compatible with the v50a route.
	% Optional post-BP support pruning is followed by a short
	% support-fixed BP-LSQ refinement and accepted only when validation loss is not worse.
	opts.init.postBPPrune = struct();
	% Stage-2 post-BP pruning/refinement follows the full-network release.  Its
	% coefficient seed is the Stage-1 fixed-support handoff result, with all newly
	% released augmentation coefficients initialized to zero.
	opts.init.postBPPrune.enable = true;
	opts.init.postBPPrune.numIterations = 1;
	opts.init.postBPPrune.scoreMode = 'contribution_abs_mean';
	opts.init.postBPPrune.absThreshold = 1e-4;
	opts.init.postBPPrune.relThreshold = 0;
	opts.init.postBPPrune.contributionAbsThreshold = 1e-4;
	opts.init.postBPPrune.contributionRelThreshold = 0;
	opts.init.postBPPrune.minTermsPerXiRow = 0;
	opts.init.postBPPrune.refineMaxIter = 50;
	opts.init.postBPPrune.refineMaxFunEvals = 5000;
	opts.init.postBPPrune.acceptByValidation = true;
	opts.init.postBPPrune.maxRelValIncrease = 1e-4;
	opts.init.postBPPrune.verbose = true;

	% Skip-mode coefficient-seed fallback.
	% If Stage 1 provides a light-BP coefficient seed, skip uses it directly.
	% If Stage 1 is bypassed (for example strong_prior), skip performs a cheap
	% allowed-mask light-BP multi-start screening before the final full BP-LSQ.
	opts.init.skip = struct();
	opts.init.skip.useExternalStage0Seed = true;
	opts.init.skip.useBaselineScreen = true;
	opts.init.skip.numSeedCandidates = 40;
	opts.init.skip.numFinalBPStarts = 8;
	opts.init.skip.screenMaxIter = 30;
	opts.init.skip.screenMaxFunEvals = 1500;
	opts.init.skip.useParallel = false;
	opts.init.skip.autoStartParallelPool = false;



	% ------------------------------------------------------------------
	% Three-stage PySR-augmented PhDN route.
	% Stage 0: per-output fixed-SINDy bypass, then independent official PySR
	%          search only for unresolved outputs. Select each PySR core from
	%          the near-best-validation structure-score pool.
	% Stage 1: recursively decompose the selected core trees into one shared
	%          compact structural DAG. Every active branch receives the same
	%          dimension-dependent constant+total-degree Poly_2 augmentation
	%          family; SR-specific operators are structural channels, not
	%          augmentation-dictionary priors.
	% Stage 2: BP/LSQ refinement without a coefficient hard mask, followed by
	%          optional contribution pruning.
	% ------------------------------------------------------------------
	opts.stage0 = struct();
	opts.stage0.method = 'sindy_bypass_then_native_multioutput_pysr';
	opts.stage0.enable = true;
	opts.stage0.verbose = true;
	opts.stage0.singleLayerBypassEnable = true;
	opts.stage0.singleLayerBypassThreshold = 1e-12;
	opts.stage0.worstOutputWeight = 0.10;

	% Fixed Phi_SINDy: used only by the preliminary bypass. Stage 1 uses its own
	% weak uniform augmentation family (constant + total-degree Poly_2).
	opts.stage0.baseDictionary = struct();
	opts.stage0.baseDictionary.polyOrder = 2;
	opts.stage0.baseDictionary.unaryOperators = {'inv','sqrt','exp','sin','cos'};
	opts.stage0.baseDictionary.includeUnaryOnMonomials = true;
	opts.stage0.baseDictionary.includeOperatorCrossTerms = true;
	opts.stage0.baseDictionary.includeSinCosPair = false;
	opts.stage0.baseDictionary.maxLibraryTerms = Inf;
	opts.stage0.fit = struct();
	opts.stage0.fit.thresholdList = [0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3];
	opts.stage0.fit.maxSTLSQIter = 10;
	opts.stage0.fit.ridgeLambda = 1e-10;
	opts.stage0.fit.scaleFloor = 1e-12;
	opts.stage0.fit.coefficientZeroTolerance = 1e-12;
	opts.stage0.fit.complexityTieWeight = 1e-12;
	opts.stage0.fit.computeHoldoutMetrics = true;
	opts.stage0.fit.storePredictions = true;
	opts.stage0.fit.buildExpressions = true;

	% Native official PySR search. No contextual fitness, shared-dictionary
	% evolution, LOO contribution, native/contextual split, or archive union.
	opts.stage0.pysr = struct();
	opts.stage0.pysr.pythonExe = 'python';
	opts.stage0.pysr.pysrPaperRoot = '';
	opts.stage0.pysr.workRoot = fullfile(tempdir, 'phdn_stage0_pysr_runs');
	opts.stage0.pysr.keepWorkDir = true;
	opts.stage0.pysr.grammarCasemode = 'general';
	opts.stage0.pysr.populationSize = 100;
	opts.stage0.pysr.innerNumRestarts = 4;
	opts.stage0.pysr.innerNIterations = 500;
	opts.stage0.pysr.innerPopulations = 8;
	% Total population budget shared fairly by unresolved outputs. Official
	% PySR uses one equal per-output population count; adaptive unequal offspring
	% quotas require a SymbolicRegression.jl scheduler extension.
	opts.stage0.pysr.multiOutputMode = 'native_single_fit_independent_output_archives';
	opts.stage0.pysr.populationBudgetMode = 'fixed_total';
	opts.stage0.pysr.innerRandomStateStride = 100;

	% General Stage-0 per-output targeted rescue.  This is deliberately
	% output-self-referenced: heterogeneous outputs are never compared against
	% each other.  After the ordinary multi-output restarts, trigger A detects
	% a poor best candidate together with large within-output restart dispersion;
	% trigger B gives one additional chance to an output whose best candidate is
	% persistently poor even when the ordinary restarts agree.  A second rescue
	% restart is allowed only when the first one improves the running best q by
	% at least 50% but has not yet reached the soft quality threshold.
	opts.stage0.pysr.adaptiveRescueEnable = true;
	opts.stage0.pysr.adaptiveRescueSoftNormalizedMSE = 1e-6;
	opts.stage0.pysr.adaptiveRescueHardNormalizedMSE = 1e-4;
	opts.stage0.pysr.adaptiveRescueInstabilityFactor = 5;
	opts.stage0.pysr.adaptiveRescueContinueImprovementRatio = 0.5; % legacy compatibility; fixed-two policy ignores this
	opts.stage0.pysr.adaptiveRescueGuessFraction = 0.05;
	opts.stage0.pysr.adaptiveRescueUseKnownNoiseFloor = true;
	opts.stage0.pysr.adaptiveRescueNoiseFloorMultiplier = 4;
	opts.stage0.pysr.adaptiveRescueMaxRestartsPerOutput = 2;
	opts.stage0.pysr.adaptiveRescuePopulationMultiplier = 1.5;
	opts.stage0.pysr.adaptiveRescueMinPopulations = 8;
	opts.stage0.pysr.adaptiveRescueMaxPopulations = 20;
	opts.stage0.pysr.adaptiveRescuePopulations = []; % optional explicit override
	opts.stage0.pysr.adaptiveRescueNIterations = []; % [] -> same as base Stage-0
	opts.stage0.pysr.adaptiveRescueMaxOutputs = 3;
	opts.stage0.pysr.adaptiveRescueUseOutputSpecificInitialGuess = true;
	opts.stage0.pysr.adaptiveRescueSeedOffset = 10000;
	opts.stage0.pysr.adaptiveRescueOutputSeedStride = 1000;
	opts.stage0.pysr.adaptiveRescueRestartSeedStride = 100;

	% Optional mapping from the shared initial-guess library to ORIGINAL output
	% indices.  Zero means a genuinely shared guess.  This mapping is used only
	% by the single-output rescue; the ordinary multi-output PySR call continues
	% to receive the same shared library as before.
	opts.stage0.pysr.initialGuessOutputMap = [];
	opts.stage0.pysr.maxSize = 30;
	opts.stage0.pysr.maxDepth = 10;
	opts.stage0.pysr.parsimony = 1e-6;
	opts.stage0.pysr.modelSelection = 'best';
	opts.stage0.pysr.randomState = 1;
	opts.stage0.pysr.deterministic = true;
	opts.stage0.pysr.parallelism = 'serial';
	opts.stage0.pysr.strictDeterministicTestMode = false;
	opts.stage0.pysr.repeatabilityPredictionTolerance = 1e-12;
	opts.stage0.pysr.verbosity = 1;
	opts.stage0.pysr.progress = false;
	opts.stage0.pysr.binaryOperators = {'+','-','*','/'};
	opts.stage0.pysr.unaryOperators = {'square','cube','inv','sqrt','exp','sin','cos'};
	% Optional PySR complexity overrides keyed by operator name. v73d uses
	% op_custom1=4 in weak_prior_lv2 so the atomic search node retains the
	% complexity of its four-operator primitive expansion.
	opts.stage0.pysr.operatorComplexities = struct();
	% A task registry may provide an authoritative SR grammar. This makes
	% weak_prior_lv2 usable outside the live demo as well; demo-level options
	% can still override these defaults afterwards.
	if isfield(task, 'prior') && isstruct(task.prior) && ...
			isfield(task.prior, 'srGrammar') && isstruct(task.prior.srGrammar)
		Gsr = task.prior.srGrammar;
		opts.stage0.pysr.grammarCasemode = get_struct_field_local_v73d( ...
			task, 'casemode', opts.stage0.pysr.grammarCasemode);
		opts.stage0.pysr.binaryOperators = get_struct_field_local_v73d( ...
			Gsr, 'binaryOperators', opts.stage0.pysr.binaryOperators);
		opts.stage0.pysr.unaryOperators = get_struct_field_local_v73d( ...
			Gsr, 'unaryOperators', opts.stage0.pysr.unaryOperators);
		opts.stage0.pysr.operatorComplexities = get_struct_field_local_v73d( ...
			Gsr, 'operatorComplexities', opts.stage0.pysr.operatorComplexities);
	end
	opts.stage0.pysr.topKExpressionsToReport = 10;
	% Display of exported, semantically unique PySR Pareto candidates. The
	% structure-score ranking is also the Stage-0 core-selection rule.
	opts.stage0.pysr.displayCandidateRankings = false;
	opts.stage0.pysr.candidateRankingTopK = 10;
	opts.stage0.pysr.equationLossMultiplier = Inf;
	opts.stage0.pysr.maxReportComplexity = Inf;
	opts.stage0.pysr.semanticDedupTolerance = 1e-8;
	opts.stage0.pysr.structureScoreEnable = true;
	opts.stage0.pysr.structureValidationMultiplier = 4.0;
	opts.stage0.pysr.structureValidationWeight = 0.20;
	opts.stage0.pysr.structureNeighborhoodMaxDistance = 0.55;
	opts.stage0.pysr.structureNeighborhoodMinDistance = 0.10;
	opts.stage0.pysr.structureNeighborhoodComplexityWindow = 8;
	opts.stage0.pysr.structureFrontierMaxAbs = 20;

	opts.stage1 = struct();
	opts.stage1.method = 'sr_to_phdn_augmented_dag_initialization';
	opts.stage1.dictionaryMode = 'sr_structural_dag_plus_uniform_poly2_augmentation';
	opts.stage1.enableAugmentation = true;
	opts.stage1.includeBestExpressionPath = true;
	% Uniform augmentation dictionary for every active branch: constant plus all
	% total-degree polynomial terms up to order 2. No unary/operator terms are
	% added from the SR skeleton or the bypass dictionary.
	opts.stage1.augmentationPolyOrder = 2;
	opts.stage1.augmentationIncludeCrossTerms = true;
	% Optional fixed neural-ridge augmentation. Existing cases retain the
	% polynomial default; neural ridge parameters are generated from Stage-0
	% branch-state samples and remain frozen during Stage 2.
	opts.stage1.augmentationMode = 'polynomial';
	opts.stage1.augmentationNeuralCount = [];
	opts.stage1.augmentationNeuralActivation = 'tanh';
	opts.stage1.augmentationNeuralQuantiles = [0.25,0.50,0.75];
	opts.stage1.augmentationNeuralScales = [0.5,1,2];
	opts.stage1.augmentationNeuralPoolRatio = 3;
	opts.stage1.augmentationNeuralSeed = 1701;
	opts.stage1.augmentationNeuralStdFloor = 1e-10;
	opts.stage1.augmentationNeuralVarianceThreshold = 1e-8;
	opts.stage1.augmentationNeuralCorrelationThreshold = 0.995;
	opts.stage1.augmentationNeuralEnsureFullDirectionalSpan = true;
	% Preserve the first-order branch-coordinate basis v1,...,vd when the
	% quadratic polynomial augmentation is replaced by fixed neural ridges.
	% Disabled globally for backward compatibility; the Lorenz demo enables it.
	opts.stage1.augmentationNeuralIncludeLinearTerms = false;
	opts.stage1.forceStage0SeedOnly = true;
	opts.stage1.expandBoundsToIncludeStage0Seed = true;
	opts.stage1.stage0SeedBoundMargin = 1e-6;
	% Before the augmented PhDN is released, polish only the exact nonzero SR
	% support.  This fixed-support handoff prevents zero-initialized residual
	% branches from disturbing the continuous calibration of the symbolic core.
	% After this pass, Stage 2 receives the polished coefficients, activates the
	% full augmented network, and applies the existing refinement/pruning route.
	opts.stage1.fixedSupportRefine = struct();
	opts.stage1.fixedSupportRefine.enable = true;
	opts.stage1.fixedSupportRefine.maxIter = 500;
	opts.stage1.fixedSupportRefine.maxFunEvals = 5e4;
	opts.stage1.fixedSupportRefine.useAnalyticJacobian = true;
	opts.stage1.fixedSupportRefine.acceptByValidation = true;
	opts.stage1.fixedSupportRefine.maxRelValIncrease = 0;
	% A compiled PySR tree is decomposed into artificial hidden DAG states.
	% Clipping those intermediate states changes the symbolic expression before
	% later small coefficients/divisors can rescale it.  Keep the raw Stage-0
	% structural path exact; this override is applied only to compiled Stage-0
	% candidates and does not change ordinary PhDN/MLP architectures.
	opts.stage1.preserveRawStage0HiddenStates = true;
	opts.stage1.requireExactStage0Reproduction = true;
	opts.stage1.stage0ReproductionRelTolerance = 1e-6;
	opts.stage1.stage0ReproductionAbsTolerance = 1e-10;
	opts.stage1.verbose = true;

	opts.stage2 = struct();
	opts.stage2.enable = true;
	opts.stage2.method = 'bp_lsq_refinement_limited_densification_pruning';
	opts.stage2.acceptOnlyByValidation = true;

	% Objective used by MLP-surrogate LSQ-BP and protected GA-LSQ.
	opts.init.objective = struct();
	opts.init.objective.normalizeResidual = true;
	opts.init.objective.residualScale = 'std';
	opts.init.objective.invalidPenalty = 1e6;
	opts.init.objective.lambda2 = 0;
	opts.init.objective.lambda1 = 0;
	opts.init.objective.epsSmoothL1 = 1e-8;
	opts.init.verbose = true;
	opts.init.debugMaskSummary = true;
	% Reporting-only diagnostics for Stage-2 augmentation activation. These
	% fields never change seeds, masks, bounds, objectives, or model selection.
	opts.init.augmentationDiagnostics = struct();
	opts.init.augmentationDiagnostics.enable = false;
	opts.init.augmentationDiagnostics.nonzeroThreshold = 1e-10;
	opts.init.augmentationDiagnostics.printPerOutput = true;

	% Branch-MLP-surrogate controls for general/weak_prior.
	% Important separation:
	%   Stage 1 MLP-surrogate settings live in opts/task.mlpSurrogate.
	%   Stage 1I recovery/final-BP settings live in opts/task.exactRecovery.
	%   Stage 1I physical dictionaries are NOT specified here; they are taken
	%   from the existing task.arch.caseDictionary/generalDictionary path.
	%   Therefore case-specific physical bases such as inv/sqrt/exp/sin/cos
	%   should be configured through task.arch.generalDictionary.priorBasis,
	%   not through task.mlpSurrogate.
	opts.mlpSurrogate = struct();
	opts.mlpSurrogate.mlpHiddenDims = [8];          % [] = direct branch terminal link, [3 4] = two middle MLP layers
	opts.mlpSurrogate.mlpActivation = 'tanh';
	opts.mlpSurrogate.mlpInitMode = 'xavier';   % Xavier/Glorot is the default choice for tanh MLP branches. Use 'he' for relu.
	opts.mlpSurrogate.mlpInitScale = 1.0;
	opts.mlpSurrogate.surrogateNumStarts = 10;
	opts.mlpSurrogate.surrogateMaxIter = 300;
	opts.mlpSurrogate.surrogateMaxFunEvals = 5e4;
	opts.mlpSurrogate.randomSeed = 1;
	opts.mlpSurrogate.useParallelStarts = true;       % Train different Stage-1 starts in parallel when a pool is available.
	opts.mlpSurrogate.parallelAutoStartPool = true;  % Automatically open a local parpool for Stage-1 starts.

	% Stage-1 objective controls. The default remains direct MLP prediction only.
	% Set stage1DictionaryCompatibilityWeight > 0 to add a fixed-LS exact-dictionary
	% compatibility residual whose Jacobian is propagated only through MLP hidden
	% states / MLP parameters, not through the LS coefficient blocks.
	opts.mlpSurrogate.stage1MlpPredictionWeight = 1.0;
	opts.mlpSurrogate.stage1DictionaryCompatibilityWeight = 0.0;
	opts.mlpSurrogate.stage1DenseLSMode = 'ridge';       % ridge | pinv | backslash
	opts.mlpSurrogate.stage1DenseLSLambda2 = 1e-6;
	opts.mlpSurrogate.reportStage1DenseDictionary = false;  % hidden diagnostic; not printed in the main report

	if isfield(task, 'mlpSurrogate') && isstruct(task.mlpSurrogate)
		opts.mlpSurrogate = merge_structs_local(opts.mlpSurrogate, task.mlpSurrogate);
	end

	% Stage-1I exact-recovery solver controls.  These are deliberately separated
	% from mlpSurrogate because they act after the MLP surrogate has been trained.
	% The exact dictionary itself is still generated by the old task dictionary
	% path: task.arch.caseDictionary/generalDictionary.
	opts.exactRecovery = struct();
	opts.exactRecovery.stlsThreshold = 1e-4;
	opts.exactRecovery.stlsMaxIter = 10;
	opts.exactRecovery.stlsLambda2 = 1e-10;
	opts.exactRecovery.stlsMinTermsPerRow = 1;

	% Global final support-fixed BP-LSQ refinement settings shared by all cases.
	% This is not an MLP-surrogate setting; it refines the recovered exact-operator
	% PhDN after Stage-1I STLS.
	opts.exactRecovery.finalBP = struct();
	opts.exactRecovery.finalBP.enable = true;
	opts.exactRecovery.finalBP.maxIter = 1000;
	opts.exactRecovery.finalBP.maxFunEvals = 1e5;
	opts.exactRecovery.finalBP.maxRelValIncrease = 1e-4;

	if isfield(task, 'exactRecovery') && isstruct(task.exactRecovery)
		opts.exactRecovery = merge_structs_local(opts.exactRecovery, task.exactRecovery);
	end

	% Output.
	opts.output = struct();
	opts.output.saveTimingMat = true;
	opts.output.timingMatName = sprintf('%s_mlpSurrogate_timeStats.mat', task.name);
	opts.output.cleanDigits = 4;
	opts.output.zeroTol = 1e-8;
	opts.output.symbolicZeroTol = 0;
	opts.output.symbolicScreenZeroTol = 0;
	opts.output.intTol = 1e-6;
	opts.output.symbolicDigits = 3;
	opts.output.simplifySymbolic = true;
	opts.output.skipSymbolicDisplay = false;
	opts.output.cleanSymbolicDisplay = true;
	opts.output.printFinalXiMatrices = false;
	opts.output.finalXiPrintPrecision = 4;
	opts.output.finalXiPrintOnlyActive = false;
	opts.output.finalXiActiveTol = 1e-10;
end


function out = merge_structs_local(defaults, overrides)
%MERGE_STRUCTS_LOCAL Recursively merge task-level overrides into defaults.
	out = defaults;
	if ~isstruct(overrides)
		return;
	end
	names = fieldnames(overrides);
	for i = 1:numel(names)
		name = names{i};
		val = overrides.(name);
		if isstruct(val) && isfield(out, name) && isstruct(out.(name))
			out.(name) = merge_structs_local(out.(name), val);
		else
			out.(name) = val;
		end
	end
end


function value = get_task_data_default_local(task, fieldName, fallback)
%GET_TASK_DATA_DEFAULT_LOCAL Read optional task-specific benchmark data defaults.
	value = fallback;
	if isstruct(task) && isfield(task, 'dataDefaults') && isstruct(task.dataDefaults) && ...
			isfield(task.dataDefaults, fieldName) && ~isempty(task.dataDefaults.(fieldName))
		value = task.dataDefaults.(fieldName);
	end
end


function value = get_struct_field_local_v73d(S, fieldName, fallback)
	value = fallback;
	if isstruct(S) && isfield(S, fieldName) && ~isempty(S.(fieldName))
		value = S.(fieldName);
	end
end
