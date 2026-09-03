function result = run_phdn_per_output_pysr_stage0(task, archBase, data, options)
%RUN_PHDN_PER_OUTPUT_PYSR_STAGE0 SINDy bypass then one native multi-output PySR fit.
%
% Stage 0 performs only symbolic skeleton discovery. The fixed SINDy library
% is validated independently for every output. Accepted outputs retain their
% SINDy expressions; official PySR searches only the unresolved outputs in one
% two-dimensional-Y fit per restart. Outputs retain independent losses, Pareto
% archives, candidate validation, and structure scoring while sharing PySR's
% physical scheduler. Each
% restart exports the candidates whose validation MSE is at most a configured
% multiple of the best candidate. The pools are merged and ranked by robust
% structural locality and a strictly two-sided validation knee.
%
% No shared-dictionary evolution, contextual fitness, LOO credit assignment,
% or native/contextual islands are used. Inner restarts do not alter PySR;
% they are independent standard fits coordinated by this wrapper.

    tTotal = tic;
    required = {'Xtr','Ytr','Xval','Yval','Xte','Yte'};
    for i = 1:numel(required)
        if ~isfield(data, required{i})
            error('Stage-0 data is missing field %s.', required{i});
        end
    end
    if nargin < 4 || isempty(options); options = struct(); end
    options = fill_defaults_local(options);

    % Preliminary fixed-SINDy bypass. Phi_SINDy is also retained as the
    % general augmentation family used by Stage 1 after bypass rejection.
    [base, baseTime] = build_stage0_base_dictionary(task, archBase, data, options);
    options.fit.worstOutputWeight = options.worstOutputWeight;
    options.fit.computeHoldoutMetrics = true;
    baseFit = fit_sindy_per_output_local(base, data, options.fit);

    result = struct();
    result.method = 'sindy_bypass_then_native_multioutput_pysr';
    result.applied = true;
    result.baseDictionary = base;
    result.baseOnlyModel = baseFit;
    result.baseDictionaryTime = baseTime;
    result.usedSingleLayerBypass = false;
    result.usedPerOutputSindyBypass = false;
    result.bypassOutputMask = false(1, size(data.Ytr,2));
    result.searchResult = struct();

    valPerOutputMSE = getfield_default_local(baseFit, 'valPerOutputMSE', ...
        mean((baseFit.prediction.Yval - data.Yval).^2, 1));
    bypassMask = logical(options.singleLayerBypassEnable) & isfinite(valPerOutputMSE) & ...
        valPerOutputMSE <= options.singleLayerBypassThreshold;
    result.bypassOutputMask = bypassMask;
    result.usedPerOutputSindyBypass = any(bypassMask);

    if all(bypassMask)
        result.usedSingleLayerBypass = true;
        result.candidates = make_candidate_local(baseFit, 'fixed_sindy_bypass');
        result.bestModel = baseFit;
        result.coreExpressions = baseFit.outputExpressions;
        result.bestScoreExpressions = baseFit.outputExpressions;
        result.bestExpressions = baseFit.outputExpressions; % backward-compatible alias
        result.skeletonSet = make_skeleton_set_local(result.coreExpressions, struct([]));
        result.trainTime = toc(tTotal);
        result.searchTime = 0;
        result.reason = 'all_outputs_met_per_output_fixed_sindy_validation_threshold';
        result.workDir = '';
        if options.verbose
            fprintf('Stage 0 per-output fixed-SINDy bypass accepted for all %d output(s).\n', numel(bypassMask));
            for j = 1:numel(bypassMask)
                fprintf('  y%d SINDy validation MSE %.6e <= %.6e.\n', ...
                    j, valPerOutputMSE(j), options.singleLayerBypassThreshold);
            end
        end
        return;
    end

    srOutputIndices = find(~bypassMask);
    srData = subset_output_data_local(data, srOutputIndices);
    srTask = task;
    srTask.ny = numel(srOutputIndices);
    if isfield(srTask, 'outputNames') && numel(srTask.outputNames) >= max(srOutputIndices)
        srTask.outputNames = srTask.outputNames(srOutputIndices);
    end
    if options.verbose && any(bypassMask)
        fprintf('Stage 0 per-output SINDy bypass accepted y={%s}; PySR will search y={%s}.\n', ...
            strjoin(arrayfun(@num2str,find(bypassMask),'UniformOutput',false),','), ...
            strjoin(arrayfun(@num2str,srOutputIndices,'UniformOutput',false),','));
    end

    srOpts = sr_default_options();
    srOpts.reportRole = 'stage0_native_multioutput_pysr';
    srOpts.reportTitle = 'Stage-0 native multi-output PySR';
    srOpts.grammarCasemode = options.pysr.grammarCasemode;
    srOpts.pythonExe = options.pysr.pythonExe;
    srOpts.pysrPaperRoot = options.pysr.pysrPaperRoot;
    srOpts.workRoot = options.pysr.workRoot;
    srOpts.keepWorkDir = options.pysr.keepWorkDir;
    srOpts.minimumPySRVersion = char(string(getfield_default_local( ...
        options.pysr,'minimumPySRVersion','2.0.0a2')));
    srOpts.requirePySR2 = logical(getfield_default_local( ...
        options.pysr,'requirePySR2',true));
    requestedInitialGuessesEnable = logical(getfield_default_local( ...
        options.pysr,'initialGuessesEnable',false));
    srOpts.initialGuesses = normalize_initial_guesses_local(getfield_default_local( ...
        options.pysr,'initialGuesses',{}));
    % An empty library always means that Stage-0 guesses are disabled. This
    % keeps the public empty-port setting safe even if an upstream option
    % accidentally leaves initialGuessesEnable=true.
    srOpts.initialGuessesEnable = requestedInitialGuessesEnable && ...
        ~isempty(srOpts.initialGuesses);
    srOpts.fractionReplacedGuesses = getfield_default_local( ...
        options.pysr,'fractionReplacedGuesses',0.05);
    srOpts.initialGuessScope = char(string(getfield_default_local( ...
        options.pysr,'initialGuessScope','shared_all_unresolved_outputs')));
    srOpts.initialGuessOutputMap = normalize_initial_guess_output_map_local( ...
        getfield_default_local(options.pysr,'initialGuessOutputMap',[]), ...
        numel(srOpts.initialGuesses));
    srOpts.machinePrecisionEarlyStopEnable = logical(getfield_default_local( ...
        options.pysr,'machinePrecisionEarlyStopEnable',false));
    srOpts.machinePrecisionEarlyStopAbsMSE = max(0,getfield_default_local( ...
        options.pysr,'machinePrecisionEarlyStopAbsMSE',1e-12));
    srOpts.machinePrecisionEarlyStopRelMSE = max(0,getfield_default_local( ...
        options.pysr,'machinePrecisionEarlyStopRelMSE',1e-12));
    srOpts.machinePrecisionEarlyStopCheckInterval = max(1,round(getfield_default_local( ...
        options.pysr,'machinePrecisionEarlyStopCheckInterval',50)));
    srOpts.machinePrecisionEarlyStopMinIterations = max(1,round(getfield_default_local( ...
        options.pysr,'machinePrecisionEarlyStopMinIterations', ...
        srOpts.machinePrecisionEarlyStopCheckInterval)));
    srOpts.machinePrecisionEarlyStopAcrossRestarts = logical(getfield_default_local( ...
        options.pysr,'machinePrecisionEarlyStopAcrossRestarts',true));
    srOpts.populationSize = options.pysr.populationSize;
    srOpts.multiOutputMode = getfield_default_local(options.pysr, 'multiOutputMode', ...
        'native_single_fit_independent_output_archives');
    srOpts.populationBudgetMode = getfield_default_local(options.pysr, 'populationBudgetMode', 'fixed_total');
    srOpts.maxSize = options.pysr.maxSize;
    srOpts.maxDepth = options.pysr.maxDepth;
    srOpts.parsimony = options.pysr.parsimony;
    srOpts.modelSelection = options.pysr.modelSelection;
    srOpts.randomState = options.pysr.randomState;
    srOpts.deterministic = options.pysr.deterministic;
    srOpts.parallelism = options.pysr.parallelism;
    srOpts.batching = logical(getfield_default_local(options.pysr, 'batching', false));
    srOpts.batchSize = max(1, round(getfield_default_local(options.pysr, 'batchSize', 50)));
    % Optional evolutionary controls are forwarded only when explicitly set by
    % a case. This keeps all other demos on their existing PySR defaults.
    srOpts.tournamentSelectionN = getfield_default_local( ...
        options.pysr, 'tournamentSelectionN', []);
    srOpts.tournamentSelectionP = getfield_default_local( ...
        options.pysr, 'tournamentSelectionP', []);
    srOpts.annealing = getfield_default_local(options.pysr, 'annealing', []);
    srOpts.annealingAlpha = getfield_default_local( ...
        options.pysr, 'annealingAlpha', []);
    srOpts.crossoverProbability = getfield_default_local( ...
        options.pysr, 'crossoverProbability', []);
    srOpts.weightOptimize = getfield_default_local( ...
        options.pysr, 'weightOptimize', []);
    srOpts.weightAddNode = getfield_default_local(options.pysr,'weightAddNode',[]);
    srOpts.weightInsertNode = getfield_default_local(options.pysr,'weightInsertNode',[]);
    srOpts.weightDeleteNode = getfield_default_local(options.pysr,'weightDeleteNode',[]);
    srOpts.weightDoNothing = getfield_default_local(options.pysr,'weightDoNothing',[]);
    srOpts.weightMutateConstant = getfield_default_local(options.pysr,'weightMutateConstant',[]);
    srOpts.weightMutateOperator = getfield_default_local(options.pysr,'weightMutateOperator',[]);
    srOpts.weightMutateFeature = getfield_default_local(options.pysr,'weightMutateFeature',[]);
    srOpts.weightSwapOperands = getfield_default_local(options.pysr,'weightSwapOperands',[]);
    srOpts.weightRotateTree = getfield_default_local(options.pysr,'weightRotateTree',[]);
    srOpts.weightRandomize = getfield_default_local(options.pysr,'weightRandomize',[]);
    srOpts.weightSimplify = getfield_default_local(options.pysr,'weightSimplify',[]);
    srOpts.optimizeProbability = getfield_default_local(options.pysr,'optimizeProbability',[]);
    srOpts.shouldSimplify = getfield_default_local(options.pysr,'shouldSimplify',[]);
    srOpts.verbosity = options.pysr.verbosity;
    srOpts.progress = options.pysr.progress;
    srOpts.binaryOperators = options.pysr.binaryOperators;
    srOpts.unaryOperators = options.pysr.unaryOperators;

    caseSupportsTypedPrior = is_single_generator_dynamic_case_local(task);
    typedPriorEnable = logical(getfield_default_local( ...
        options.pysr,'typedPhysicalPriorEnable',caseSupportsTypedPrior));
    typedPriorMode = char(string(getfield_default_local( ...
        options.pysr,'typedPhysicalConstraints','single_generator_dynamic')));
    if typedPriorEnable
        if ~caseSupportsTypedPrior
            error(['The typed generator physical prior was requested for a task ', ...
                'that is not recognized as SingleGeneratorDynamic/SMIB-AVR.']);
        end
        % Fixed machine/network denominators are constants and are absorbed into
        % fitted scalar coefficients. State-dependent division/inversion is not
        % physically admissible for this explicit fixed-parameter ODE.
        if logical(getfield_default_local(options.pysr, ...
                'forbidStateDependentDivision',true))
            srOpts.binaryOperators = srOpts.binaryOperators( ...
                ~strcmp(srOpts.binaryOperators,'/'));
            srOpts.unaryOperators = srOpts.unaryOperators( ...
                ~strcmpi(srOpts.unaryOperators,'inv'));
        end
        srOpts.typedPhysicalConstraints = typedPriorMode;
        srOpts.trigAllowedVariables = getfield_default_local( ...
            options.pysr,'trigAllowedVariables',{'x1'});
        srOpts.strictTrigAtomsOnly = logical(getfield_default_local( ...
            options.pysr,'strictTrigAtomsOnly',true));
    else
        srOpts.typedPhysicalConstraints = '';
    end
    srOpts.operatorComplexities = getfield_default_local( ...
        options.pysr, 'operatorComplexities', struct());
    srOpts.forbidNestedTrig = logical(getfield_default_local( ...
        options.pysr, 'forbidNestedTrig', false));
    srOpts.forbidNestedSquare = logical(getfield_default_local( ...
        options.pysr, 'forbidNestedSquare', false));
    srOpts.forbidNestedSqrt = logical(getfield_default_local( ...
        options.pysr, 'forbidNestedSqrt', false));
    srOpts.forbidMotifReverseNesting = logical(getfield_default_local( ...
        options.pysr, 'forbidMotifReverseNesting', false));
    srOpts.maxOperatorOccurrences = getfield_default_local( ...
        options.pysr, 'maxOperatorOccurrences', struct());
    srOpts.topKExpressionsToReport = options.pysr.topKExpressionsToReport;
    srOpts.displayCandidateRankings = options.pysr.displayCandidateRankings;
    srOpts.candidateRankingTopK = options.pysr.candidateRankingTopK;
    srOpts.equationLossMultiplier = options.pysr.equationLossMultiplier;
    srOpts.maxReportComplexity = options.pysr.maxReportComplexity;
    srOpts.semanticDedupTolerance = options.pysr.semanticDedupTolerance;
    srOpts.structureScoreEnable = options.pysr.structureScoreEnable;
    srOpts.structureValidationMultiplier = options.pysr.structureValidationMultiplier;
    srOpts.structureValidationWeight = options.pysr.structureValidationWeight;
    srOpts.structureMachineErrorAbsMSEFloor = options.pysr.structureMachineErrorAbsMSEFloor;
    srOpts.structureMachineErrorRelMSEFloor = options.pysr.structureMachineErrorRelMSEFloor;
    srOpts.structureNeighborhoodMaxDistance = options.pysr.structureNeighborhoodMaxDistance;
    srOpts.structureNeighborhoodMinDistance = options.pysr.structureNeighborhoodMinDistance;
    srOpts.structureNeighborhoodComplexityWindow = options.pysr.structureNeighborhoodComplexityWindow;
    srOpts.structureFrontierMaxAbs = options.pysr.structureFrontierMaxAbs;
    srOpts.verbose = options.verbose;

    innerNumRestarts = max(1, round(getfield_default_local(options.pysr, 'innerNumRestarts', 4)));
    innerNIterations = max(1, round(getfield_default_local(options.pysr, 'innerNIterations', 500)));
    innerPopulations = max(1, round(getfield_default_local(options.pysr, 'innerPopulations', 8)));
    if strcmpi(char(string(getfield_default_local(options.pysr,'populationBudgetMode','fixed_total'))), ...
            'fixed_total') && innerPopulations < numel(srOutputIndices)
        error(['Stage-0 fixed total population budget (%d) must be at least the ', ...
            'number of unresolved outputs (%d).'], innerPopulations, numel(srOutputIndices));
    end
    innerSeedStride = round(getfield_default_local(options.pysr, 'innerRandomStateStride', 100));

    % Optional validation-triggered targeted rescue. The base multi-output
    % search is always completed first. Each unresolved output is judged only
    % against its own validation scale and its own restart history, so the same
    % mechanism applies to heterogeneous multi-output systems as well as cyclic
    % systems. Two trigger paths are supported:
    %   A) poor best quality + large within-output restart dispersion;
    %   B) persistently poor best quality even when restart dispersion is small.
    % A/B both launch official single-output PySR rescue restarts. Once an
    % output is flagged, up to maxRestartsPerOutput rescue restarts are run
    % independently; the only early stop is reaching the soft quality threshold.
    % Rescue guess injection is controlled separately from the base multi-output
    % search so prior pressure can be reduced during recovery.
    adaptiveRescueEnable = logical(getfield_default_local( ...
        options.pysr,'adaptiveRescueEnable',true));
    adaptiveRescueSoftNormalizedMSE = getfield_default_local( ...
        options.pysr,'adaptiveRescueSoftNormalizedMSE',1e-6);
    adaptiveRescueHardNormalizedMSE = getfield_default_local( ...
        options.pysr,'adaptiveRescueHardNormalizedMSE',1e-4);
    adaptiveRescueInstabilityFactor = getfield_default_local( ...
        options.pysr,'adaptiveRescueInstabilityFactor',5);
    adaptiveRescueGuessFraction = getfield_default_local( ...
        options.pysr,'adaptiveRescueGuessFraction',0.05);
    adaptiveRescueNoiseFloorMultiplier = getfield_default_local( ...
        options.pysr,'adaptiveRescueNoiseFloorMultiplier',4);
    adaptiveRescueUseKnownNoiseFloor = logical(getfield_default_local( ...
        options.pysr,'adaptiveRescueUseKnownNoiseFloor',true));
    adaptiveRescueMaxRestarts = max(1,round(getfield_default_local( ...
        options.pysr,'adaptiveRescueMaxRestartsPerOutput',2)));
    adaptiveRescuePopulationMultiplier = getfield_default_local( ...
        options.pysr,'adaptiveRescuePopulationMultiplier',1.5);
    adaptiveRescueMinPopulations = max(1,round(getfield_default_local( ...
        options.pysr,'adaptiveRescueMinPopulations',8)));
    adaptiveRescueMaxPopulations = max(adaptiveRescueMinPopulations,round(getfield_default_local( ...
        options.pysr,'adaptiveRescueMaxPopulations',20)));
    adaptiveRescueExplicitPopulations = getfield_default_local( ...
        options.pysr,'adaptiveRescuePopulations',[]);
    adaptiveRescueNIterationsRaw = getfield_default_local( ...
        options.pysr,'adaptiveRescueNIterations',[]);
    if isempty(adaptiveRescueNIterationsRaw)
        adaptiveRescueNIterations = innerNIterations;
    else
        adaptiveRescueNIterations = max(1,round(adaptiveRescueNIterationsRaw));
    end
    adaptiveRescueMaxOutputs = getfield_default_local( ...
        options.pysr,'adaptiveRescueMaxOutputs',3);
    adaptiveRescueUseMatchedGuess = logical(getfield_default_local( ...
        options.pysr,'adaptiveRescueUseOutputSpecificInitialGuess', ...
        getfield_default_local(options.pysr,'adaptiveRescueUseMatchedInitialGuess',true)));
    adaptiveRescueSeedOffset = round(getfield_default_local( ...
        options.pysr,'adaptiveRescueSeedOffset',10000));
    adaptiveRescueOutputSeedStride = round(getfield_default_local( ...
        options.pysr,'adaptiveRescueOutputSeedStride',1000));
    adaptiveRescueRestartSeedStride = round(getfield_default_local( ...
        options.pysr,'adaptiveRescueRestartSeedStride',100));

    if ~isscalar(adaptiveRescueSoftNormalizedMSE) || ...
            ~isfinite(adaptiveRescueSoftNormalizedMSE) || adaptiveRescueSoftNormalizedMSE <= 0
        adaptiveRescueSoftNormalizedMSE = 1e-6;
    end
    if ~isscalar(adaptiveRescueHardNormalizedMSE) || ...
            ~isfinite(adaptiveRescueHardNormalizedMSE) || adaptiveRescueHardNormalizedMSE <= 0
        adaptiveRescueHardNormalizedMSE = 1e-4;
    end
    adaptiveRescueHardNormalizedMSE = max( ...
        adaptiveRescueHardNormalizedMSE,adaptiveRescueSoftNormalizedMSE);
    if ~isscalar(adaptiveRescueInstabilityFactor) || ...
            ~isfinite(adaptiveRescueInstabilityFactor) || adaptiveRescueInstabilityFactor <= 1
        adaptiveRescueInstabilityFactor = 5;
    end
    if ~isscalar(adaptiveRescueGuessFraction) || ...
            ~isfinite(adaptiveRescueGuessFraction) || ...
            adaptiveRescueGuessFraction < 0 || adaptiveRescueGuessFraction > 1
        adaptiveRescueGuessFraction = 0.05;
    end
    if ~isscalar(adaptiveRescueNoiseFloorMultiplier) || ...
            ~isfinite(adaptiveRescueNoiseFloorMultiplier) || adaptiveRescueNoiseFloorMultiplier < 1
        adaptiveRescueNoiseFloorMultiplier = 4;
    end
    if ~isscalar(adaptiveRescuePopulationMultiplier) || ...
            ~isfinite(adaptiveRescuePopulationMultiplier) || adaptiveRescuePopulationMultiplier < 1
        adaptiveRescuePopulationMultiplier = 1.5;
    end
    if ~isempty(adaptiveRescueExplicitPopulations)
        if ~isscalar(adaptiveRescueExplicitPopulations) || ...
                ~isfinite(adaptiveRescueExplicitPopulations) || adaptiveRescueExplicitPopulations < 1
            adaptiveRescueExplicitPopulations = [];
        else
            adaptiveRescueExplicitPopulations = round(adaptiveRescueExplicitPopulations);
        end
    end
    if isempty(adaptiveRescueMaxOutputs) || ~isscalar(adaptiveRescueMaxOutputs) || ...
            isnan(adaptiveRescueMaxOutputs) || adaptiveRescueMaxOutputs < 1
        adaptiveRescueMaxOutputs = 3;
    end

    % Strict deterministic repeat-check mode is a diagnostic of one identical
    % PySR call, so keep the adaptive controller out of that special test.
    if logical(getfield_default_local(options.pysr,'strictDeterministicTestMode',false)) && ...
            adaptiveRescueEnable
        adaptiveRescueEnable = false;
        if options.verbose
            fprintf(['Stage 0 general targeted rescue disabled for strict deterministic ', ...
                'repeat-check mode.\n']);
        end
    end

    srOpts.originalOutputIndices = srOutputIndices;
    srOpts.nIterations = innerNIterations;
    srOpts.populations = innerPopulations;

    if options.verbose
        populationsPerOutput = max(1, floor(srOpts.populations / numel(srOutputIndices)));
        effectiveTotalPopulations = populationsPerOutput * numel(srOutputIndices);
        fprintf(['Stage 0 native multi-output PySR: one 2-D-Y fit per restart for %d unresolved ', ...
            'output(s), with independent output losses/archives.\n'], numel(srOutputIndices));
        fprintf(['Stage 0 PySR fixed-total population budget: iterations/total populations/', ...
            'populationSize = %d/%d/%d; equal protected allocation = %d per output ', ...
            '(%d effective total).\n'], srOpts.nIterations, srOpts.populations, ...
            srOpts.populationSize, populationsPerOutput, effectiveTotalPopulations);
        fprintf('Stage 0 PySR operators: binary={%s}, unary={%s}.\n', ...
            strjoin(normalize_list_local(srOpts.binaryOperators), ','), ...
            strjoin(normalize_list_local(srOpts.unaryOperators), ','));
        fprintf(['Stage 0 PySR forbidden nesting: trig-in-trig=%d, ', ...
            'square-in-square=%d, sqrt-in-sqrt=%d.\n'], ...
            srOpts.forbidNestedTrig,srOpts.forbidNestedSquare, ...
            srOpts.forbidNestedSqrt);
        if srOpts.initialGuessesEnable
            periodicReinjection = srOpts.fractionReplacedGuesses > 0;
            fprintf(['Stage 0 PySR 2 shared initial guesses: enabled=1, ', ...
                'scope=%s, count=%d, fraction-replaced=%.3f, ', ...
                'periodic-reinjection=%d, policy=official-soft, ', ...
                'minimum-version=%s.\n'], ...
                srOpts.initialGuessScope,numel(srOpts.initialGuesses), ...
                srOpts.fractionReplacedGuesses,periodicReinjection, ...
                srOpts.minimumPySRVersion);
            if ~isempty(srOpts.initialGuessOutputMap)
                fprintf(['Stage 0 rescue-only initial-guess output map ', ...
                    '(0=shared, original output indices): [%s].\n'], ...
                    num2str(srOpts.initialGuessOutputMap));
            end
        else
            fprintf('Stage 0 PySR 2 shared initial guesses disabled.\n');
        end
        if srOpts.batching
            fprintf(['Stage 0 PySR mini-batch evolution enabled: batch size=%d; ', ...
                'Hall-of-Fame comparison remains full-data.\n'], srOpts.batchSize);
        else
            fprintf('Stage 0 PySR mini-batch evolution disabled; all mutations use full training data.\n');
        end
        growthControls = [srOpts.weightAddNode,srOpts.weightInsertNode, ...
            srOpts.weightDeleteNode,srOpts.weightDoNothing, ...
            srOpts.weightMutateConstant,srOpts.weightMutateOperator, ...
            srOpts.weightMutateFeature,srOpts.weightSwapOperands, ...
            srOpts.weightRotateTree,srOpts.weightRandomize, ...
            srOpts.weightSimplify,srOpts.weightOptimize];
        if ~isempty(srOpts.tournamentSelectionN) || ...
                ~isempty(srOpts.tournamentSelectionP) || ...
                ~isempty(srOpts.annealing) || ...
                ~isempty(srOpts.annealingAlpha) || ...
                ~isempty(srOpts.crossoverProbability) || ...
                any(~isnan(growthControls)) || ...
                ~isempty(srOpts.optimizeProbability) || ...
                ~isempty(srOpts.shouldSimplify)
            fprintf(['Stage 0 case-local evolution controls: ncycles_per_iteration=', ...
                'unchanged; tournament n/p=%s/%s; annealing=%s, alpha=%s; ', ...
                'crossover=%s; optimize-probability=%s; should-simplify=%s.\n'], ...
                value_text_local(srOpts.tournamentSelectionN), ...
                value_text_local(srOpts.tournamentSelectionP), ...
                value_text_local(srOpts.annealing), ...
                value_text_local(srOpts.annealingAlpha), ...
                value_text_local(srOpts.crossoverProbability), ...
                value_text_local(srOpts.optimizeProbability), ...
                value_text_local(srOpts.shouldSimplify));
            fprintf(['Stage 0 mutation weights: add=%s, insert=%s, delete=%s, ', ...
                'do-nothing=%s, mutate-constant=%s, mutate-operator=%s, ', ...
                'mutate-feature=%s, swap=%s, rotate=%s, randomize=%s, ', ...
                'simplify=%s, optimize=%s.\n'], ...
                value_text_local(srOpts.weightAddNode), ...
                value_text_local(srOpts.weightInsertNode), ...
                value_text_local(srOpts.weightDeleteNode), ...
                value_text_local(srOpts.weightDoNothing), ...
                value_text_local(srOpts.weightMutateConstant), ...
                value_text_local(srOpts.weightMutateOperator), ...
                value_text_local(srOpts.weightMutateFeature), ...
                value_text_local(srOpts.weightSwapOperands), ...
                value_text_local(srOpts.weightRotateTree), ...
                value_text_local(srOpts.weightRandomize), ...
                value_text_local(srOpts.weightSimplify), ...
                value_text_local(srOpts.weightOptimize));
        end
        if typedPriorEnable
            fprintf(['Stage 0 typed generator prior [%s]: trig variables={%s}; ', ...
                'no state-dependent denominator; fixed trig atoms only=%d.\n'], ...
                srOpts.typedPhysicalConstraints, ...
                strjoin(normalize_list_local(srOpts.trigAllowedVariables),','), ...
                srOpts.strictTrigAtomsOnly);
        end
        fprintf(['Stage 0 selection numerical floor: max(abs %.3e, rel %.3e ', ...
            '* max(1,mean(y_val.^2))); simplicity wins inside the floor.\n'], ...
            srOpts.structureMachineErrorAbsMSEFloor, ...
            srOpts.structureMachineErrorRelMSEFloor);
        if srOpts.machinePrecisionEarlyStopEnable
            fprintf(['Stage 0 machine-precision training early stop: all unresolved ', ...
                'outputs required; train-HOF threshold=max(abs %.3e, rel %.3e ', ...
                '* max(1,mean(y_train.^2))); check every %d iterations after ', ...
                'minimum %d; stop remaining restarts after external-validation ', ...
                'confirmation=%d.\n'], ...
                srOpts.machinePrecisionEarlyStopAbsMSE, ...
                srOpts.machinePrecisionEarlyStopRelMSE, ...
                srOpts.machinePrecisionEarlyStopCheckInterval, ...
                srOpts.machinePrecisionEarlyStopMinIterations, ...
                srOpts.machinePrecisionEarlyStopAcrossRestarts);
        else
            fprintf('Stage 0 machine-precision training early stop: disabled.\n');
        end
        if ~isempty(fieldnames(srOpts.operatorComplexities))
            complexityFields = fieldnames(srOpts.operatorComplexities);
            complexityParts = cell(size(complexityFields));
            for iComplexity = 1:numel(complexityFields)
                name = complexityFields{iComplexity};
                complexityParts{iComplexity} = sprintf('%s=%.6g', name, ...
                    srOpts.operatorComplexities.(name));
            end
            fprintf('Stage 0 PySR custom operator complexities: {%s}.\n', ...
                strjoin(complexityParts, ','));
        end
    end

    tSearch = tic;
    if logical(getfield_default_local(options.pysr, 'strictDeterministicTestMode', false)) && innerNumRestarts > 1
        error('Strict deterministic repeat-check mode requires Stage-0 innerNumRestarts = 1.');
    end
    if options.verbose
        fprintf(['Stage 0 restart search: %d independent native multi-output PySR fit(s), each with ', ...
            '%d iterations, total population budget %d, population size %d.\n'], ...
            innerNumRestarts, innerNIterations, innerPopulations, srOpts.populationSize);
        if adaptiveRescueEnable
            fprintf(['Stage 0 general targeted rescue: enabled=1; ', ...
                'A=(best q > soft threshold AND restart-instability > %.2fx), ', ...
                'B=(best q > hard threshold); clean soft/hard q=%.1e/%.1e, ', ...
                'known-noise floor multiplier=%.2f, rescue guess fraction=%.3f, ', ...
                'policy=run up to %d extra restart(s)/output unless soft threshold is reached, ', ...
                'population rule=max(%d,ceil(%.2fx base/output)) capped at %d, ', ...
                'iterations=%d, max outputs=%s, output-specific guess=%d.\n'], ...
                adaptiveRescueInstabilityFactor,adaptiveRescueSoftNormalizedMSE, ...
                adaptiveRescueHardNormalizedMSE,adaptiveRescueNoiseFloorMultiplier, ...
                adaptiveRescueGuessFraction,adaptiveRescueMaxRestarts, ...
                adaptiveRescueMinPopulations,adaptiveRescuePopulationMultiplier, ...
                adaptiveRescueMaxPopulations,adaptiveRescueNIterations, ...
                value_text_local(adaptiveRescueMaxOutputs),adaptiveRescueUseMatchedGuess);
        else
            fprintf('Stage 0 general targeted rescue: disabled.\n');
        end
    end
    restartResults = cell(innerNumRestarts, 1);
    restartSeeds = zeros(innerNumRestarts, 1);
    restartTimes = zeros(innerNumRestarts, 1);
    completedRestarts = 0;
    earlyStopAcrossRestartsTriggered = false;
    baseWorkRoot = srOpts.workRoot;
    for iRestart = 1:innerNumRestarts
        srOptsRestart = srOpts;
        srOptsRestart.randomState = srOpts.randomState + (iRestart - 1) * innerSeedStride;
        srOptsRestart.workRoot = fullfile(baseWorkRoot, ...
            sprintf('outer_seed_%d', srOpts.randomState), ...
            sprintf('inner_restart_%02d_seed_%d', iRestart, srOptsRestart.randomState));
        restartSeeds(iRestart) = srOptsRestart.randomState;
        if options.verbose
            fprintf('  Restart %d/%d | seed=%d | budget(iter/total-pop/popSize)=%d/%d/%d\n', ...
                iRestart, innerNumRestarts, srOptsRestart.randomState, ...
                srOptsRestart.nIterations, srOptsRestart.populations, srOptsRestart.populationSize);
        end
        tRestart = tic;
        restartResults{iRestart} = train_official_pysr_baseline( ...
            srData.Xtr, srData.Ytr, srData.Xval, srData.Yval, srData.Xte, srData.Yte, ...
            getfield_default_local(srData,'Xood',[]), getfield_default_local(srData,'Yood',[]), ...
            srTask, srOptsRestart, []);
        restartTimes(iRestart) = toc(tRestart);
        completedRestarts = iRestart;

        restartTriggered = logical(getfield_default_local( ...
            restartResults{iRestart},'machinePrecisionEarlyStopTriggered',false));
        [externalValReached,valLosses,valThresholds] = ...
            machine_precision_external_validation_reached_local( ...
                restartResults{iRestart},srData, ...
                srOpts.machinePrecisionEarlyStopAbsMSE, ...
                srOpts.machinePrecisionEarlyStopRelMSE);
        restartResults{iRestart}.machinePrecisionExternalValidationReached = externalValReached;
        restartResults{iRestart}.machinePrecisionExternalValidationLosses = valLosses;
        restartResults{iRestart}.machinePrecisionExternalValidationThresholds = valThresholds;
        if options.verbose && srOpts.machinePrecisionEarlyStopEnable
            fprintf(['    machine-precision status: train-HOF-triggered=%d, ', ...
                'iterations=%d/%d, external-val-all-output=%d, ', ...
                'val losses=[%s], thresholds=[%s].\n'], ...
                restartTriggered, ...
                getfield_default_local(restartResults{iRestart}, ...
                    'nIterationsCompleted',srOptsRestart.nIterations), ...
                getfield_default_local(restartResults{iRestart}, ...
                    'nIterationsRequested',srOptsRestart.nIterations), ...
                externalValReached,num2str(valLosses,'%.3e '), ...
                num2str(valThresholds,'%.3e '));
        end
        if srOpts.machinePrecisionEarlyStopEnable && ...
                srOpts.machinePrecisionEarlyStopAcrossRestarts && ...
                restartTriggered && externalValReached
            earlyStopAcrossRestartsTriggered = true;
            if options.verbose
                fprintf(['  Stage 0 machine-precision early stop confirmed on ', ...
                    'training Hall of Fame and external validation; skipping ', ...
                    '%d remaining restart(s).\n'],innerNumRestarts-iRestart);
            end
            break;
        end
    end
    restartResults = restartResults(1:completedRestarts);
    restartSeeds = restartSeeds(1:completedRestarts);
    restartTimes = restartTimes(1:completedRestarts);

    adaptiveRescueInfo = empty_adaptive_rescue_info_local();
    if adaptiveRescueEnable
        rescueConfig = struct();
        rescueConfig.softNormalizedMSE = adaptiveRescueSoftNormalizedMSE;
        rescueConfig.hardNormalizedMSE = adaptiveRescueHardNormalizedMSE;
        rescueConfig.instabilityFactor = adaptiveRescueInstabilityFactor;
        rescueConfig.guessFraction = adaptiveRescueGuessFraction;
        rescueConfig.noiseFloorMultiplier = adaptiveRescueNoiseFloorMultiplier;
        rescueConfig.useKnownNoiseFloor = adaptiveRescueUseKnownNoiseFloor;
        rescueConfig.maxRestartsPerOutput = adaptiveRescueMaxRestarts;
        rescueConfig.populationMultiplier = adaptiveRescuePopulationMultiplier;
        rescueConfig.minPopulations = adaptiveRescueMinPopulations;
        rescueConfig.maxPopulations = adaptiveRescueMaxPopulations;
        rescueConfig.explicitPopulations = adaptiveRescueExplicitPopulations;
        rescueConfig.nIterations = adaptiveRescueNIterations;
        rescueConfig.maxOutputs = adaptiveRescueMaxOutputs;
        rescueConfig.useOutputSpecificInitialGuess = adaptiveRescueUseMatchedGuess;
        rescueConfig.seedOffset = adaptiveRescueSeedOffset;
        rescueConfig.outputSeedStride = adaptiveRescueOutputSeedStride;
        rescueConfig.restartSeedStride = adaptiveRescueRestartSeedStride;
        [restartResults,restartSeeds,restartTimes,adaptiveRescueInfo] = ...
            run_adaptive_targeted_rescue_local( ...
                restartResults,restartSeeds,restartTimes,srData,srTask,srOpts, ...
                srOutputIndices,task.ny,rescueConfig,baseWorkRoot,options.verbose);
    end

    srResult = merge_inner_restart_results_local( ...
        restartResults, restartSeeds, restartTimes, srData, options.pysr);
    srResult.adaptiveRescue = adaptiveRescueInfo;
    srResult.machinePrecisionEarlyStop = struct( ...
        'enabled',srOpts.machinePrecisionEarlyStopEnable, ...
        'acrossRestartsEnabled',srOpts.machinePrecisionEarlyStopAcrossRestarts, ...
        'triggeredAcrossRestarts',earlyStopAcrossRestartsTriggered, ...
        'completedRestarts',completedRestarts, ...
        'requestedRestarts',innerNumRestarts, ...
        'absoluteMSEFloor',srOpts.machinePrecisionEarlyStopAbsMSE, ...
        'relativeMSEFloor',srOpts.machinePrecisionEarlyStopRelMSE, ...
        'checkInterval',srOpts.machinePrecisionEarlyStopCheckInterval, ...
        'minimumIterations',srOpts.machinePrecisionEarlyStopMinIterations);
    if logical(getfield_default_local(options.pysr, 'strictDeterministicTestMode', false))
        if ~logical(srOpts.deterministic) || ~strcmpi(strtrim(char(srOpts.parallelism)), 'serial')
            error('Strict Stage-0 deterministic test mode requires deterministic=true and parallelism=''serial''.');
        end
        if options.verbose
            fprintf('Stage 0 strict deterministic test: repeating the identical PySR search once for exact comparison.\n');
        end
        srResultRepeat = train_official_pysr_baseline(srData.Xtr, srData.Ytr, srData.Xval, srData.Yval, ...
            srData.Xte, srData.Yte, getfield_default_local(srData,'Xood',[]), ...
            getfield_default_local(srData,'Yood',[]), srTask, srOpts, []);
        repeatTol = getfield_default_local(options.pysr, 'repeatabilityPredictionTolerance', 1e-12);
        repeatReport = assert_stage0_repeatability_local(srResult, srResultRepeat, repeatTol);
        srResult.repeatabilityCheck = repeatReport;
        if options.verbose
            fprintf('Stage 0 strict deterministic test passed: expressions identical, max prediction difference %.3e.\n', ...
                repeatReport.maxPredictionDifference);
        end
    end
    searchTime = toc(tSearch);

    if any(bypassMask)
        srResult = combine_per_output_bypass_and_sr_local( ...
            baseFit, srResult, bypassMask, srOutputIndices, data);
    end

    coreExpressions = getfield_default_local(srResult, 'bestStructureExpressionsPerOutput', {});
    if isempty(coreExpressions)
        coreExpressions = getfield_default_local(srResult, 'bestScoreExpressionsPerOutput', {});
    end
    if isempty(coreExpressions)
        coreExpressions = getfield_default_local(srResult, 'coreExpressionsPerOutput', {});
    end
    if isempty(coreExpressions) || numel(coreExpressions) ~= task.ny
        error('Official PySR Stage 0 did not return one selected structure core per output.');
    end

    model = struct();
    model.outputExpressions = coreExpressions;
    model.coreExpressions = coreExpressions;
    model.bestScoreExpressions = coreExpressions;
    model.bestExpressions = coreExpressions; % backward-compatible alias
    model.trainMetrics = srResult.trainMetrics;
    model.valMetrics = srResult.valMetrics;
    model.testMetrics = srResult.testMetrics;
    model.oodMetrics = srResult.oodMetrics;
    model.YhatTrain = srResult.YhatTrain;
    model.YhatVal = srResult.YhatVal;
    model.YhatTest = srResult.YhatTest;
    model.YhatOod = srResult.YhatOod;
    model.outputSelections = getfield_default_local(srResult, 'outputSelections', struct([]));
    model.candidateRankingsByValidation = getfield_default_local(srResult, 'candidateRankingsByValidation', struct([]));
    model.candidateRankingsByComplexity = getfield_default_local(srResult, 'candidateRankingsByComplexity', struct([]));
    model.candidateRankingsByBestScore = getfield_default_local(srResult, 'candidateRankingsByBestScore', struct([]));
    model.candidateRankingsByStructureScore = getfield_default_local(srResult, 'candidateRankingsByStructureScore', struct([]));
    model.variableIndexBase = getfield_default_local(srResult, 'variableIndexBase', 1);
    model.complexity = total_selected_complexity_local(model.outputSelections, ...
        getfield_default_local(srResult, 'complexity', NaN));
    model.nActiveTerms = model.complexity;
    model.searchBackend = 'per_output_sindy_bypass_plus_native_multioutput_pysr';
    model.adaptiveRescue = getfield_default_local(srResult,'adaptiveRescue', ...
        empty_adaptive_rescue_info_local());
    if isstruct(model.outputSelections) && ~isempty(model.outputSelections)
        selectionModes = arrayfun(@(s) char(string(getfield_default_local( ...
            s,'selectionMode',''))), model.outputSelections, 'UniformOutput', false);
        selectionModes = unique(selectionModes(~cellfun(@isempty,selectionModes)), 'stable');
        if isempty(selectionModes)
            model.selectionMode = 'machine_floor_aware_soft_validation_structure_score';
        elseif numel(selectionModes) == 1
            model.selectionMode = selectionModes{1};
        else
            model.selectionMode = strjoin(selectionModes, '+');
        end
    else
        model.selectionMode = 'machine_floor_aware_soft_validation_structure_score';
    end

    candidate = make_candidate_local(model, 'per_output_sindy_or_native_pysr_structure_score');
    candidate.valLoss = model.valMetrics.mse;
    candidate.fitness = candidate.valLoss;
    candidate.complexity = model.complexity;

    result.searchResult = srResult;
    result.candidates = candidate;
    result.bestModel = model;
    result.coreExpressions = coreExpressions;
    result.bestScoreExpressions = coreExpressions;
    result.bestExpressions = coreExpressions; % backward-compatible alias
    result.skeletonSet = make_skeleton_set_local(coreExpressions, model.outputSelections);
    result.searchTime = searchTime;
    result.trainTime = toc(tTotal);
    result.reason = 'per_output_sindy_bypass_and_native_multioutput_pysr_structure_score_completed';
    result.workDir = getfield_default_local(srResult, 'workDir', '');
    result.pythonExecutable = getfield_default_local(srResult, 'pythonExe', options.pysr.pythonExe);

    if options.verbose
        fprintf('Stage 0 native multi-output PySR completed in %.3f s.\n', searchTime);
        for j = 1:task.ny
            coreMeta = selection_meta_local(model.outputSelections, j, 'core');
            if isempty(fieldnames(coreMeta)); coreMeta = selection_meta_local(model.outputSelections, j, 'bestScore'); end
            fprintf(['  y%d selected core | structure score %.6g | Sval %.4f | Sfront %.4g (norm %.4f) | ', ...
                'val MSE %.6e | complexity %.1f | %s\n'], j, ...
                getfield_default_local(coreMeta,'structure_score',NaN), ...
                getfield_default_local(coreMeta,'validation_soft_score',NaN), ...
                getfield_default_local(coreMeta,'frontier_score_raw',NaN), ...
                getfield_default_local(coreMeta,'frontier_score_normalized',NaN), ...
                getfield_default_local(coreMeta,'validation_mse',NaN), ...
                getfield_default_local(coreMeta,'complexity',NaN), coreExpressions{j});
        end
        if logical(getfield_default_local(options.pysr, 'displayCandidateRankings', false))
            print_candidate_rankings_local(model.outputSelections, ...
                getfield_default_local(options.pysr, 'candidateRankingTopK', 20));
        end
    end
end


function [restartResults,restartSeeds,restartTimes,info] = ...
        run_adaptive_targeted_rescue_local( ...
        restartResults,restartSeeds,restartTimes,srData,srTask,srOpts, ...
        srOutputIndices,nOriginalOutputs,cfg,baseWorkRoot,verbose)
%RUN_ADAPTIVE_TARGETED_RESCUE_LOCAL General per-output Stage-0 rescue.
%
% Each unresolved output is assessed against its own validation scale.  Let
% q_{j,r}=MSE_val(j,r)/scale_j, where scale_j is the validation variance with
% a robust mean-square fallback.  The ordinary restarts define
%   qBest_j   = min_r q_{j,r},
%   qMedian_j = median_r q_{j,r},
%   D_j       = qMedian_j / max(qBest_j,eps).
%
% Trigger A (stochastic instability):
%   qBest_j > softThreshold_j AND D_j > instabilityFactor.
% Trigger B (persistent poor quality):
%   not A AND qBest_j > hardThreshold_j.
%
% If the derivative-noise protocol is known, both thresholds are lifted above
% the expected validation-noise MSE floor. Once an output is flagged, two
% independent single-output rescue restarts are available by default. The
% second is skipped only when the first already reaches the soft threshold.
% Rescue guess injection is deliberately weaker than the base search.

    info = empty_adaptive_rescue_info_local();
    info.enabled = true;
    info.triggerMode = 'general_output_self_referenced_A_instability_B_persistent_poor';
    info.softNormalizedMSE = cfg.softNormalizedMSE;
    info.hardNormalizedMSE = cfg.hardNormalizedMSE;
    info.instabilityFactor = cfg.instabilityFactor;
    info.continueImprovementRatio = NaN; % legacy field; fixed-two policy
    info.guessFraction = cfg.guessFraction;
    info.noiseFloorMultiplier = cfg.noiseFloorMultiplier;
    info.useKnownNoiseFloor = cfg.useKnownNoiseFloor;
    info.maxRestartsPerOutput = cfg.maxRestartsPerOutput;
    info.populationMultiplier = cfg.populationMultiplier;
    info.minPopulations = cfg.minPopulations;
    info.maxPopulations = cfg.maxPopulations;
    info.explicitPopulations = cfg.explicitPopulations;
    info.nIterations = cfg.nIterations;
    info.maxOutputs = cfg.maxOutputs;
    info.useOutputSpecificInitialGuess = cfg.useOutputSpecificInitialGuess;

    [restartMSE,restartQ,bestMSE,bestQ,medianQ,instabilityRatio,targetScale] = ...
        validation_quality_by_restart_local(restartResults,srData);
    noiseFloorQ = known_validation_noise_floor_local( ...
        srData,srOutputIndices,targetScale,cfg.useKnownNoiseFloor);
    softThresholdQ = max(cfg.softNormalizedMSE, ...
        cfg.noiseFloorMultiplier.*noiseFloorQ);
    hardThresholdQ = max(cfg.hardNormalizedMSE, ...
        cfg.noiseFloorMultiplier.*noiseFloorQ);
    hardThresholdQ = max(hardThresholdQ,softThresholdQ);

    [flagMask,triggerA,triggerB,triggerReason,severity] = ...
        general_adaptive_rescue_trigger_local( ...
            bestQ,instabilityRatio,softThresholdQ,hardThresholdQ, ...
            cfg.instabilityFactor,cfg.maxOutputs);

    info.baseRestartValidationMSE = restartMSE;
    info.baseRestartNormalizedMSE = restartQ;
    info.preRescueBestValidationMSE = bestMSE;
    info.preRescueNormalizedMSE = bestQ;
    info.preRescueMedianNormalizedMSE = medianQ;
    info.restartInstabilityRatio = instabilityRatio;
    info.targetScale = targetScale;
    info.noiseFloorNormalizedMSE = noiseFloorQ;
    info.softTriggerNormalizedMSE = softThresholdQ;
    info.hardTriggerNormalizedMSE = hardThresholdQ;
    info.triggerA = triggerA;
    info.triggerB = triggerB;
    info.triggerReasonByLocalOutput = triggerReason;
    info.triggerSeverity = severity;
    info.flaggedLocalOutputIndices = find(flagMask);
    info.flaggedOriginalOutputIndices = srOutputIndices(flagMask);
    info.nFlaggedOutputs = nnz(flagMask);
    info.totalExtraRestarts = 0;
    info.outputs = struct([]);

    if verbose
        fprintf(['Stage 0 general targeted-rescue screening is output-self-referenced: ', ...
            'q=MSE_val/validation-variance (RMS fallback); A tests restart instability, ', ...
            'B tests persistently poor absolute quality.\n']);
        for j = 1:numel(bestQ)
            fprintf(['  local y%d (original y%d): base q=[%s], best=%.3e, median=%.3e, ', ...
                'D=%.3g, noise-floor=%.3e, soft/hard=%.3e/%.3e, trigger=%s, rescue=%d\n'], ...
                j,srOutputIndices(j),num2str(restartQ(:,j).','%.3e '), ...
                bestQ(j),medianQ(j),instabilityRatio(j),noiseFloorQ(j), ...
                softThresholdQ(j),hardThresholdQ(j),triggerReason{j},flagMask(j));
        end
    end

    if ~any(flagMask)
        info.status = 'not_triggered';
        return;
    end

    basePopulationsPerOutput = max(1,floor(srOpts.populations/max(1,numel(srOutputIndices))));
    if ~isempty(cfg.explicitPopulations)
        rescuePopulations = round(cfg.explicitPopulations);
    else
        rescuePopulations = max(cfg.minPopulations, ...
            ceil(cfg.populationMultiplier*basePopulationsPerOutput));
        rescuePopulations = min(cfg.maxPopulations,rescuePopulations);
    end
    rescuePopulations = max(1,rescuePopulations);
    info.effectivePopulations = rescuePopulations;
    info.basePopulationsPerOutput = basePopulationsPerOutput;

    flagged = find(flagMask);
    for iFlag = 1:numel(flagged)
        jLocal = flagged(iFlag);
        originalIndex = srOutputIndices(jLocal);
        rescueData = subset_output_data_local(srData,jLocal);
        rescueTask = srTask;
        rescueTask.ny = 1;
        if isfield(srTask,'outputNames') && numel(srTask.outputNames) >= jLocal
            rescueTask.outputNames = srTask.outputNames(jLocal);
        end

        runningBestMSE = bestMSE(jLocal);
        runningBestQ = bestQ(jLocal);
        softQ = softThresholdQ(jLocal);
        hardQ = hardThresholdQ(jLocal);
        outputRecord = struct( ...
            'localOutputIndex',jLocal, ...
            'originalOutputIndex',originalIndex, ...
            'triggerReason',triggerReason{jLocal}, ...
            'triggerA',triggerA(jLocal), ...
            'triggerB',triggerB(jLocal), ...
            'preRescueBestValidationMSE',runningBestMSE, ...
            'preRescueNormalizedMSE',runningBestQ, ...
            'baseRestartValidationMSE',restartMSE(:,jLocal).', ...
            'baseRestartNormalizedMSE',restartQ(:,jLocal).', ...
            'baseMedianNormalizedMSE',medianQ(jLocal), ...
            'restartInstabilityRatio',instabilityRatio(jLocal), ...
            'noiseFloorNormalizedMSE',noiseFloorQ(jLocal), ...
            'softTriggerNormalizedMSE',softQ, ...
            'hardTriggerNormalizedMSE',hardQ, ...
            'extraRestartsRun',0, ...
            'passedAfterRescue',false, ...
            'improvedAfterRescue',false, ...
            'stoppedBecauseNoUsefulImprovement',false, ... % legacy field; fixed-two policy no longer uses it
            'rescueGuessFraction',cfg.guessFraction, ...
            'restartSeeds',[], ...
            'restartValidationMSE',[], ...
            'restartNormalizedMSE',[], ...
            'runningBestNormalizedMSE',[], ...
            'improvementRatioAfterRestart',[], ...
            'usedOutputSpecificInitialGuess',false, ...
            'rescueInitialGuesses',{{}}, ...
            'postRescueBestValidationMSE',runningBestMSE, ...
            'postRescueNormalizedMSE',runningBestQ);

        for iRescue = 1:cfg.maxRestartsPerOutput
            preAttemptBestQ = runningBestQ;
            rescueOpts = srOpts;
            rescueOpts.reportRole = 'stage0_targeted_single_output_rescue';
            rescueOpts.reportTitle = sprintf( ...
                'Stage-0 targeted rescue original y%d',originalIndex);
            rescueOpts.nIterations = cfg.nIterations;
            rescueOpts.populations = rescuePopulations;
            rescueOpts.originalOutputIndices = originalIndex;
            rescueSeed = srOpts.randomState + cfg.seedOffset + ...
                (originalIndex-1)*cfg.outputSeedStride + ...
                (iRescue-1)*cfg.restartSeedStride;
            rescueOpts.randomState = rescueSeed;
            rescueOpts.workRoot = fullfile(baseWorkRoot,'adaptive_rescue', ...
                sprintf('original_y%02d',originalIndex), ...
                sprintf('rescue_restart_%02d_seed_%d',iRescue,rescueSeed));

            [rescueOpts,usedOutputSpecificGuess,rescueGuessLibrary] = ...
                configure_rescue_initial_guesses_local( ...
                    rescueOpts,nOriginalOutputs,originalIndex, ...
                    cfg.useOutputSpecificInitialGuess);
            % Use weaker prior injection in rescue than in the ordinary
            % multi-output search. This preserves the matched guess as a soft
            % exploitative cue while leaving more population mass for exploration.
            rescueOpts.fractionReplacedGuesses = cfg.guessFraction;
            outputRecord.usedOutputSpecificInitialGuess = usedOutputSpecificGuess;
            outputRecord.rescueInitialGuesses = rescueGuessLibrary;

            if verbose
                fprintf(['  Targeted rescue original y%d | trigger=%s | extra restart %d/%d | ', ...
                    'seed=%d | budget(iter/populations/popSize)=%d/%d/%d | ', ...
                    'output-specific-guess=%d | guess-fraction=%.3f\n'], ...
                    originalIndex,triggerReason{jLocal},iRescue,cfg.maxRestartsPerOutput, ...
                    rescueSeed,rescueOpts.nIterations,rescueOpts.populations, ...
                    rescueOpts.populationSize,usedOutputSpecificGuess, ...
                    rescueOpts.fractionReplacedGuesses);
            end

            tRescue = tic;
            rescueResult = train_official_pysr_baseline( ...
                rescueData.Xtr,rescueData.Ytr,rescueData.Xval,rescueData.Yval, ...
                rescueData.Xte,rescueData.Yte, ...
                getfield_default_local(rescueData,'Xood',[]), ...
                getfield_default_local(rescueData,'Yood',[]), ...
                rescueTask,rescueOpts,[]);
            rescueTime = toc(tRescue);

            rescueMSE = best_validation_quality_single_local(rescueResult);
            rescueQ = rescueMSE/max(targetScale(jLocal),realmin);
            runningBestMSE = min(runningBestMSE,rescueMSE);
            runningBestQ = runningBestMSE/max(targetScale(jLocal),realmin);
            if isfinite(preAttemptBestQ) && preAttemptBestQ > 0
                improvementRatio = runningBestQ/preAttemptBestQ;
            elseif isfinite(runningBestQ) && runningBestQ <= preAttemptBestQ
                improvementRatio = 0;
            else
                improvementRatio = Inf;
            end

            pseudoResult = expand_single_output_rescue_result_local( ...
                rescueResult,restartResults{1},jLocal,numel(srOutputIndices));
            restartResults{end+1,1} = pseudoResult; %#ok<AGROW>
            restartSeeds(end+1,1) = rescueSeed; %#ok<AGROW>
            restartTimes(end+1,1) = rescueTime; %#ok<AGROW>

            outputRecord.extraRestartsRun = iRescue;
            outputRecord.restartSeeds(end+1) = rescueSeed;
            outputRecord.restartValidationMSE(end+1) = rescueMSE;
            outputRecord.restartNormalizedMSE(end+1) = rescueQ;
            outputRecord.runningBestNormalizedMSE(end+1) = runningBestQ;
            outputRecord.improvementRatioAfterRestart(end+1) = improvementRatio;
            outputRecord.improvedAfterRescue = outputRecord.improvedAfterRescue || ...
                (isfinite(improvementRatio) && improvementRatio < 1);
            info.totalExtraRestarts = info.totalExtraRestarts + 1;

            if verbose
                fprintf(['    rescue result original y%d: restart q=%.3e; cumulative best q=%.3e; ', ...
                    'soft/hard=%.3e/%.3e; improvement ratio=%.3f.\n'], ...
                    originalIndex,rescueQ,runningBestQ,softQ,hardQ,improvementRatio);
            end

            if isfinite(runningBestQ) && runningBestQ <= softQ
                outputRecord.passedAfterRescue = true;
                if verbose
                    fprintf(['    targeted rescue original y%d reached the soft quality ', ...
                        'threshold after %d extra restart(s); stopping.\n'], ...
                        originalIndex,iRescue);
                end
                break;
            end

            if iRescue >= cfg.maxRestartsPerOutput
                break;
            end

            % Fixed-two rescue policy: once A/B flags an output, continue to
            % the next independent rescue restart whenever the soft threshold
            % has not yet been reached. A poor first rescue is not evidence that
            % another stochastic PySR restart is uninformative.
            if verbose
                fprintf(['    targeted rescue original y%d remains above the soft ', ...
                    'threshold; continuing to rescue restart %d/%d.\n'], ...
                    originalIndex,iRescue+1,cfg.maxRestartsPerOutput);
            end
        end

        outputRecord.postRescueBestValidationMSE = runningBestMSE;
        outputRecord.postRescueNormalizedMSE = runningBestQ;
        if isempty(info.outputs)
            info.outputs = outputRecord;
        else
            info.outputs(end+1) = outputRecord; %#ok<AGROW>
        end
    end

    info.status = 'completed';
    info.allFlaggedOutputsPassed = isempty(info.outputs) || ...
        all([info.outputs.passedAfterRescue]);
    if verbose
        fprintf(['Stage 0 general targeted rescue completed: flagged outputs=%d, ', ...
            'extra single-output restarts=%d, all reached soft threshold=%d.\n'], ...
            info.nFlaggedOutputs,info.totalExtraRestarts, ...
            info.allFlaggedOutputsPassed);
    end
end


function [restartMSE,restartQ,bestMSE,bestQ,medianQ,instabilityRatio,targetScale] = ...
        validation_quality_by_restart_local(restartResults,data)
%VALIDATION_QUALITY_BY_RESTART_LOCAL Per-output quality for each base restart.
    ny = size(data.Yval,2);
    nr = numel(restartResults);
    restartMSE = Inf(nr,ny);
    for r = 1:nr
        rr = restartResults{r};
        if ~isfield(rr,'outputSelections') || ~isstruct(rr.outputSelections)
            continue;
        end
        for j = 1:min(ny,numel(rr.outputSelections))
            restartMSE(r,j) = best_validation_from_selection_local( ...
                rr.outputSelections(j));
        end
    end

    targetScale = validation_target_scale_local(data.Yval);
    restartQ = restartMSE./targetScale;
    bestMSE = min(restartMSE,[],1);
    bestQ = min(restartQ,[],1);
    medianQ = NaN(1,ny);
    instabilityRatio = NaN(1,ny);
    for j = 1:ny
        vals = restartQ(:,j);
        vals = vals(isfinite(vals) & vals >= 0);
        if isempty(vals)
            medianQ(j) = Inf;
            instabilityRatio(j) = Inf;
        else
            medianQ(j) = median(vals);
            if isfinite(bestQ(j))
                instabilityRatio(j) = medianQ(j)/max(bestQ(j),realmin);
            else
                instabilityRatio(j) = Inf;
            end
        end
    end
end


function targetScale = validation_target_scale_local(Yval)
%VALIDATION_TARGET_SCALE_LOCAL Output-self-referenced normalization scale.
% Prefer validation variance because it measures the dynamic range that must be
% explained rather than the absolute offset.  Fall back to mean square for a
% nearly constant output, then to one only for a truly degenerate zero target.
    if isempty(Yval)
        targetScale = 1;
        return;
    end
    targetScale = var(Yval,0,1,'omitnan');
    rms2 = mean(Yval.^2,1,'omitnan');
    finiteRms2 = rms2(isfinite(rms2) & rms2 > 0);
    if isempty(finiteRms2)
        globalReference = 1;
    else
        globalReference = median(finiteRms2);
    end
    varianceFloor = max(realmin,1e-12*globalReference);
    bad = ~isfinite(targetScale) | targetScale <= varianceFloor;
    targetScale(bad) = rms2(bad);
    bad = ~isfinite(targetScale) | targetScale <= realmin;
    targetScale(bad) = 1;
end


function noiseFloorQ = known_validation_noise_floor_local( ...
        data,originalOutputIndices,targetScale,useKnownNoiseFloor)
%KNOWN_VALIDATION_NOISE_FLOOR_LOCAL Expected normalized validation-noise MSE.
    noiseFloorQ = zeros(1,numel(originalOutputIndices));
    if ~useKnownNoiseFloor || ~isfield(data,'derivativeLabelNoise') || ...
            ~isstruct(data.derivativeLabelNoise)
        return;
    end
    info = data.derivativeLabelNoise;
    applyToValidation = logical(getfield_default_local(info,'applyToValidation',false));
    rho = double(getfield_default_local(info,'relativeStd',0));
    scale = getfield_default_local(info,'perCoordinateScale',[]);
    if ~applyToValidation || ~isscalar(rho) || ~isfinite(rho) || rho <= 0 || ...
            isempty(scale)
        return;
    end
    scale = reshape(double(scale),1,[]);
    for j = 1:numel(originalOutputIndices)
        idx = originalOutputIndices(j);
        if idx >= 1 && idx <= numel(scale) && isfinite(scale(idx))
            noiseVar = (rho*scale(idx))^2;
            noiseFloorQ(j) = noiseVar/max(targetScale(j),realmin);
        end
    end
end


function [flagMask,triggerA,triggerB,triggerReason,severity] = ...
        general_adaptive_rescue_trigger_local( ...
        bestQ,instabilityRatio,softThresholdQ,hardThresholdQ, ...
        instabilityFactor,maxOutputs)
%GENERAL_ADAPTIVE_RESCUE_TRIGGER_LOCAL General A/B rescue detector.
    bestQ = reshape(double(bestQ),1,[]);
    instabilityRatio = reshape(double(instabilityRatio),1,[]);
    softThresholdQ = reshape(double(softThresholdQ),1,[]);
    hardThresholdQ = reshape(double(hardThresholdQ),1,[]);

    triggerA = (~isfinite(bestQ)) | ...
        ((bestQ > softThresholdQ) & ...
        (instabilityRatio > instabilityFactor));
    triggerB = ~triggerA & (bestQ > hardThresholdQ);
    flagMask = triggerA | triggerB;

    n = numel(bestQ);
    triggerReason = repmat({'none'},1,n);
    triggerReason(triggerA) = {'A_restart_instability'};
    triggerReason(triggerB) = {'B_persistent_poor_quality'};

    severity = zeros(1,n);
    aIdx = find(triggerA);
    if ~isempty(aIdx)
        severity(aIdx) = max( ...
            bestQ(aIdx)./max(softThresholdQ(aIdx),realmin), ...
            instabilityRatio(aIdx)./max(instabilityFactor,realmin));
    end
    bIdx = find(triggerB);
    if ~isempty(bIdx)
        severity(bIdx) = bestQ(bIdx)./max(hardThresholdQ(bIdx),realmin);
    end
    severity(~isfinite(severity)) = realmax;

    flagged = find(flagMask);
    if isfinite(maxOutputs) && numel(flagged) > maxOutputs
        [~,ord] = sort(severity(flagged),'descend');
        nKeep = max(1,min(numel(flagged),floor(maxOutputs)));
        keep = flagged(ord(1:nKeep));
        dropped = setdiff(flagged,keep);
        flagMask(:) = false;
        flagMask(keep) = true;
        for j = dropped
            triggerReason{j} = [triggerReason{j} '_not_run_max_outputs_cap'];
        end
    end
end


function bestMSE = best_validation_quality_single_local(resultSR)
    bestMSE = Inf;
    if isfield(resultSR,'outputSelections') && isstruct(resultSR.outputSelections) && ...
            ~isempty(resultSR.outputSelections)
        bestMSE = best_validation_from_selection_local(resultSR.outputSelections(1));
    end
    if ~isfinite(bestMSE)
        valMetrics = getfield_default_local(resultSR,'valMetrics',struct());
        bestMSE = getfield_default_local(valMetrics,'mse',Inf);
    end
end


function bestMSE = best_validation_from_selection_local(selection)
    bestMSE = Inf;
    candidates = getfield_default_local(selection,'candidates',struct([]));
    if isstruct(candidates) && ~isempty(candidates)
        vals = arrayfun(@(c) scalar_or_inf_local( ...
            getfield_default_local(c,'validation_mse',Inf)),candidates);
        if ~isempty(vals); bestMSE = min(bestMSE,min(vals)); end
    end
    core = getfield_default_local(selection,'core',struct());
    if isstruct(core) && ~isempty(fieldnames(core))
        bestMSE = min(bestMSE,scalar_or_inf_local( ...
            getfield_default_local(core,'validation_mse',Inf)));
    end
    best = getfield_default_local(selection,'best',struct());
    if isstruct(best) && ~isempty(fieldnames(best))
        bestMSE = min(bestMSE,scalar_or_inf_local( ...
            getfield_default_local(best,'validation_mse',Inf)));
    end
end


function [rescueOpts,usedOutputSpecificGuess,guessLibrary] = ...
        configure_rescue_initial_guesses_local( ...
        rescueOpts,nOriginalOutputs,originalIndex,useOutputSpecificGuess)
%CONFIGURE_RESCUE_INITIAL_GUESSES_LOCAL Prefer explicit output mappings.
% The ordinary multi-output fit still receives the shared official guess library.
% During a single-output rescue we can remove the ambiguity: if
% initialGuessOutputMap explicitly maps one or more guesses to this original
% output, only those mapped guesses (plus any map==0 shared guesses) are used.
% For backward compatibility, when no explicit map exists and the number of
% guesses equals the number of original outputs, the one-to-one positional map
% is used. Otherwise the full shared library is retained.
    usedOutputSpecificGuess = false;
    guessLibrary = normalize_initial_guesses_local(getfield_default_local( ...
        rescueOpts,'initialGuesses',{}));
    if isempty(guessLibrary)
        rescueOpts.initialGuesses = {};
        rescueOpts.initialGuessesEnable = false;
        return;
    end

    map = normalize_initial_guess_output_map_local(getfield_default_local( ...
        rescueOpts,'initialGuessOutputMap',[]),numel(guessLibrary));
    if useOutputSpecificGuess
        if ~isempty(map)
            hasMatched = any(map==originalIndex);
            selected = (map==0) | (map==originalIndex);
            % An explicit map is authoritative.  Guesses mapped to other
            % outputs are never injected into this single-output rescue.  If
            % this output has no mapped prior, retain only map==0 genuinely
            % shared guesses (possibly none).
            guessLibrary = guessLibrary(selected);
            usedOutputSpecificGuess = hasMatched;
        elseif numel(guessLibrary)==nOriginalOutputs && ...
                originalIndex>=1 && originalIndex<=numel(guessLibrary)
            guessLibrary = {guessLibrary{originalIndex}};
            usedOutputSpecificGuess = true;
        end
    end
    rescueOpts.initialGuesses = guessLibrary;
    rescueOpts.initialGuessesEnable = logical(getfield_default_local( ...
        rescueOpts,'initialGuessesEnable',false)) && ~isempty(guessLibrary);

    % The official adapter has one canonical scope name for PySR guesses:
    % shared_all_unresolved_outputs.  A targeted rescue is already a true
    % single-output fit, so after subsetting the library above, applying that
    % (possibly output-specific) library to "all unresolved outputs" means
    % applying it to this one rescued output only.  Do NOT invent a second
    % scope token here; stale versions used 'single_output_rescue', which the
    % official adapter correctly rejected before the rescue fit started.
    rescueOpts.initialGuessScope = 'shared_all_unresolved_outputs';
end


function map = normalize_initial_guess_output_map_local(rawMap,nGuesses)
%NORMALIZE_INITIAL_GUESS_OUTPUT_MAP_LOCAL 0 denotes a shared/unmapped guess.
    if nargin < 2; nGuesses = []; end
    if isempty(rawMap)
        map = [];
        return;
    end
    if iscell(rawMap)
        try
            rawMap = cellfun(@double,rawMap);
        catch
            map = [];
            return;
        end
    end
    map = reshape(double(rawMap),1,[]);
    if ~isempty(nGuesses) && numel(map) ~= nGuesses
        warning(['Stage-0 initialGuessOutputMap length (%d) does not match ', ...
            'the initial-guess library size (%d); treating the library as shared.'], ...
            numel(map),nGuesses);
        map = [];
        return;
    end
    if any(~isfinite(map)) || any(map < 0) || any(abs(map-round(map)) > 0)
        warning(['Stage-0 initialGuessOutputMap must contain nonnegative integer ', ...
            'original-output indices (0 means shared); treating the library as shared.']);
        map = [];
        return;
    end
    map = round(map);
end


function pseudo = expand_single_output_rescue_result_local( ...
        rescueResult,templateResult,targetLocalIndex,ny)
%EXPAND_SINGLE_OUTPUT_RESCUE_RESULT_LOCAL Adapt one-output result for merge.
    if ~isfield(templateResult,'outputSelections') || ...
            numel(templateResult.outputSelections) < ny
        error('Cannot expand targeted-rescue result: base outputSelections are incomplete.');
    end
    if ~isfield(rescueResult,'outputSelections') || ...
            isempty(rescueResult.outputSelections)
        error('Targeted-rescue PySR result does not contain outputSelections.');
    end

    pseudo = templateResult;
    for j = 1:ny
        sel = pseudo.outputSelections(j);
        sel.candidates = struct([]);
        sel.core = struct();
        sel.best = struct();
        sel.bestScore = struct();
        if isfield(sel,'validationRanking'); sel.validationRanking = struct([]); end
        if isfield(sel,'complexityRanking'); sel.complexityRanking = struct([]); end
        if isfield(sel,'bestScoreRanking'); sel.bestScoreRanking = struct([]); end
        if isfield(sel,'structureScoreRanking'); sel.structureScoreRanking = struct([]); end
        pseudo.outputSelections(j) = sel;
    end

    rescueSource = rescueResult.outputSelections(1);
    rescueSelection = pseudo.outputSelections(targetLocalIndex);
    fieldNames = fieldnames(rescueSelection);
    for iField = 1:numel(fieldNames)
        name = fieldNames{iField};
        if isfield(rescueSource,name)
            rescueSelection.(name) = rescueSource.(name);
        end
    end
    if isfield(rescueSelection,'outputIndex')
        rescueSelection.outputIndex = targetLocalIndex;
    end
    pseudo.outputSelections(targetLocalIndex) = rescueSelection;
end


function info = empty_adaptive_rescue_info_local()
    info = struct( ...
        'enabled',false, ...
        'status','disabled', ...
        'triggerMode','', ...
        'softNormalizedMSE',NaN, ...
        'hardNormalizedMSE',NaN, ...
        'instabilityFactor',NaN, ...
        'continueImprovementRatio',NaN, ... % legacy compatibility
        'guessFraction',NaN, ...
        'noiseFloorMultiplier',NaN, ...
        'useKnownNoiseFloor',false, ...
        'maxRestartsPerOutput',0, ...
        'populationMultiplier',NaN, ...
        'minPopulations',0, ...
        'maxPopulations',0, ...
        'explicitPopulations',[], ...
        'requestedPopulations',0, ...
        'effectivePopulations',0, ...
        'basePopulationsPerOutput',0, ...
        'nIterations',0, ...
        'maxOutputs',0, ...
        'useOutputSpecificInitialGuess',false, ...
        'baseRestartValidationMSE',[], ...
        'baseRestartNormalizedMSE',[], ...
        'preRescueBestValidationMSE',[], ...
        'preRescueNormalizedMSE',[], ...
        'preRescueMedianNormalizedMSE',[], ...
        'restartInstabilityRatio',[], ...
        'targetScale',[], ...
        'noiseFloorNormalizedMSE',[], ...
        'softTriggerNormalizedMSE',[], ...
        'hardTriggerNormalizedMSE',[], ...
        'triggerA',[], ...
        'triggerB',[], ...
        'triggerReasonByLocalOutput',{{}}, ...
        'triggerSeverity',[], ...
        'flaggedLocalOutputIndices',[], ...
        'flaggedOriginalOutputIndices',[], ...
        'nFlaggedOutputs',0, ...
        'totalExtraRestarts',0, ...
        'outputs',struct([]), ...
        'allFlaggedOutputsPassed',true);
end


function print_candidate_rankings_local(selections, requestedTopK)
    if ~isstruct(selections) || isempty(selections)
        fprintf('Stage-0 candidate ranking report requested, but no exported Pareto candidates are available.\n');
        return;
    end
    requestedTopK = max(0, round(requestedTopK));
    fprintf('\n========================================\n');
    fprintf('Stage-0 exported Pareto candidate rankings\n');
    fprintf(['All tables use the external ID validation set. The structure-score table ', ...
        'also defines the selected Stage-0 core.\n']);
    fprintf('========================================\n');
    for j = 1:numel(selections)
        outputIndex = getfield_default_local(selections(j), 'outputIndex', j);
        nAvailable = getfield_default_local(selections(j), 'candidateCount', 0);
        valRows = getfield_default_local(selections(j), 'validationRanking', struct([]));
        complexityRows = getfield_default_local(selections(j), 'complexityRanking', struct([]));
        bestScoreRows = getfield_default_local(selections(j), 'bestScoreRanking', struct([]));
        structureScoreRows = getfield_default_local(selections(j), 'structureScoreRanking', struct([]));
        fprintf('\ny%d: %d semantically unique finite Pareto candidate(s); requested top K = %d.\n', ...
            outputIndex, nAvailable, requestedTopK);
        print_candidate_table_local(valRows, ...
            'Selected-restart external validation MSE ranking (restart-local rel. best)', requestedTopK);
        print_candidate_table_local(complexityRows, ...
            'Selected-restart structural complexity ranking (restart-local rel. best)', requestedTopK);
        print_candidate_table_local(bestScoreRows, ...
            'Selected-restart native PySR score ranking (restart-local rel. best)', requestedTopK);
        print_structure_candidate_table_local(structureScoreRows, ...
            'Cross-restart global soft-validation structure-score ranking (global rel. best)', requestedTopK);
    end
end

function print_structure_candidate_table_local(rows, titleText, requestedTopK)
    fprintf('  %s\n', titleText);
    if ~isstruct(rows) || isempty(rows)
        fprintf('    <no candidates exported for this ranking>\n');
        return;
    end
    fprintf(['    rank | structure score | Sval | Sfront(raw/norm) | ', ...
        'validation MSE | rel. global best | complexity | PySR score | role           | expression\n']);
    nPrint = min(numel(rows), requestedTopK);
    for i = 1:nPrint
        fprintf(['    %4d | %15.5f | %5.3f | %10.4g/%5.3f | %14.6e | %9.4g | ', ...
            '%10.1f | %10.4g | %-14s | %s\n'], ...
            getfield_default_local(rows(i),'rank',i), ...
            getfield_default_local(rows(i),'structure_score',NaN), ...
            getfield_default_local(rows(i),'validation_soft_score',NaN), ...
            getfield_default_local(rows(i),'frontier_score_raw',NaN), ...
            getfield_default_local(rows(i),'frontier_score_normalized',NaN), ...
            getfield_default_local(rows(i),'validation_mse',NaN), ...
            getfield_default_local(rows(i),'relative_to_best_validation_mse',NaN), ...
            getfield_default_local(rows(i),'complexity',NaN), ...
            getfield_default_local(rows(i),'score',NaN), ...
            char(string(getfield_default_local(rows(i),'selection_role','none'))), ...
            char(string(getfield_default_local(rows(i),'expression','<expression unavailable>'))));
    end
end

function print_candidate_table_local(rows, titleText, requestedTopK)
    fprintf('  %s\n', titleText);
    if ~isstruct(rows) || isempty(rows)
        fprintf('    <no candidates exported for this ranking>\n');
        return;
    end
    fprintf('    rank | validation MSE | rel. restart best | complexity | PySR score | role        | expression\n');
    nPrint = min(numel(rows), requestedTopK);
    for i = 1:nPrint
        rankValue = getfield_default_local(rows(i), 'rank', i);
        valMSE = getfield_default_local(rows(i), 'validation_mse', NaN);
        relBest = getfield_default_local(rows(i), 'relative_to_best_validation_mse', NaN);
        complexity = getfield_default_local(rows(i), 'complexity', NaN);
        pysrScore = getfield_default_local(rows(i), 'score', NaN);
        role = char(string(getfield_default_local(rows(i), 'selection_role', 'none')));
        expression = char(string(getfield_default_local(rows(i), 'expression', '<expression unavailable>')));
        fprintf('    %4d | %14.6e | %12.4g | %10.1f | %10.4g | %-11s | %s\n', ...
            rankValue, valMSE, relBest, complexity, pysrScore, role, expression);
    end
end

function [reached,losses,thresholds] = ...
        machine_precision_external_validation_reached_local(resultSR,data,absFloor,relFloor)
% Confirm that a training-HOF early stop also generalizes to the fixed external
% validation set before suppressing later independent restarts.
    Yhat = getfield_default_local(resultSR,'YhatVal',[]);
    Yval = getfield_default_local(data,'Yval',[]);
    if isempty(Yhat) || isempty(Yval) || ~isequal(size(Yhat),size(Yval))
        losses = Inf(1,size(Yval,2));
        thresholds = NaN(size(losses));
        reached = false;
        return;
    end
    losses = mean((Yhat-Yval).^2,1);
    scales = max(1,mean(Yval.^2,1));
    thresholds = max(absFloor,relFloor.*scales);
    reached = all(isfinite(losses) & losses <= thresholds);
end

function options = fill_defaults_local(options)
    defaults = struct();
    defaults.verbose = true;
    defaults.singleLayerBypassEnable = true;
    defaults.singleLayerBypassThreshold = 1e-12;
    defaults.worstOutputWeight = 0.1;
    defaults.baseDictionary = struct('polyOrder',2, ...
        'unaryOperators',{{'inv','sqrt','exp','sin','cos','log'}}, ...
        'includeUnaryOnMonomials',true,'includeOperatorCrossTerms',true, ...
        'includeSinCosPair',false,'maxLibraryTerms',Inf);
    defaults.fit = struct('thresholdList',[0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3], ...
        'maxSTLSQIter',10,'ridgeLambda',1e-10,'scaleFloor',1e-12, ...
        'coefficientZeroTolerance',1e-12,'complexityTieWeight',1e-12, ...
        'computeHoldoutMetrics',true,'storePredictions',true,'buildExpressions',true);
    defaults.pysr = struct('pythonExe','python','pysrPaperRoot','', ...
        'workRoot',fullfile(tempdir,'phdn_stage0_pysr_runs'),'keepWorkDir',true, ...
        'minimumPySRVersion','2.0.0a2','requirePySR2',true, ...
        'initialGuessesEnable',false,'initialGuesses',{{}}, ...
        'fractionReplacedGuesses',0.05, ...
        'initialGuessScope','shared_all_unresolved_outputs', ...
        'initialGuessOutputMap',[], ...
        'adaptiveRescueEnable',true, ...
        'adaptiveRescueSoftNormalizedMSE',1e-6, ...
        'adaptiveRescueHardNormalizedMSE',1e-4, ...
        'adaptiveRescueInstabilityFactor',5, ...
        'adaptiveRescueContinueImprovementRatio',0.5, ... % legacy compatibility; ignored
        'adaptiveRescueGuessFraction',0.05, ...
        'adaptiveRescueUseKnownNoiseFloor',true, ...
        'adaptiveRescueNoiseFloorMultiplier',4, ...
        'adaptiveRescueMaxRestartsPerOutput',2, ...
        'adaptiveRescuePopulationMultiplier',1.5, ...
        'adaptiveRescueMinPopulations',8, ...
        'adaptiveRescueMaxPopulations',20, ...
        'adaptiveRescuePopulations',[], ...
        'adaptiveRescueNIterations',[], ...
        'adaptiveRescueMaxOutputs',3, ...
        'adaptiveRescueUseOutputSpecificInitialGuess',true, ...
        'adaptiveRescueSeedOffset',10000, ...
        'adaptiveRescueOutputSeedStride',1000, ...
        'adaptiveRescueRestartSeedStride',100, ...
        'machinePrecisionEarlyStopEnable',false, ...
        'machinePrecisionEarlyStopAbsMSE',1e-12, ...
        'machinePrecisionEarlyStopRelMSE',1e-12, ...
        'machinePrecisionEarlyStopCheckInterval',50, ...
        'machinePrecisionEarlyStopMinIterations',50, ...
        'machinePrecisionEarlyStopAcrossRestarts',true, ...
        'grammarCasemode','general','populationSize',50,'maxSize',30,'maxDepth',10,'parsimony',1e-6, ...
        'modelSelection','best','randomState',1,'deterministic',true,'parallelism','serial', ...
        'batching',false,'batchSize',50, ...
        'tournamentSelectionN',[],'tournamentSelectionP',[], ...
        'annealing',[],'annealingAlpha',[],'crossoverProbability',[], ...
        'weightAddNode',[],'weightInsertNode',[],'weightDeleteNode',[], ...
        'weightDoNothing',[],'weightMutateConstant',[], ...
        'weightMutateOperator',[],'weightMutateFeature',[], ...
        'weightSwapOperands',[],'weightRotateTree',[], ...
        'weightRandomize',[],'weightSimplify',[],'weightOptimize',[], ...
        'optimizeProbability',[],'shouldSimplify',[], ...
        'strictDeterministicTestMode',false,'repeatabilityPredictionTolerance',1e-12, ...
        'verbosity',1,'progress',false, ...
        'binaryOperators',{{'+','-','*','/'}}, ...
        'unaryOperators',{{'square','cube','inv','sqrt','exp','sin','cos','log'}}, ...
        'forbidNestedTrig',false,'forbidNestedSquare',false,'forbidNestedSqrt',false, ...
        'topKExpressionsToReport',10,'displayCandidateRankings',false, ...
        'candidateRankingTopK',10,'equationLossMultiplier',Inf, ...
        'maxReportComplexity',Inf,'semanticDedupTolerance',1e-8, ...
        'structureScoreEnable',true,'structureValidationMultiplier',4.0, ...
        'structureValidationWeight',0.20, ...
        'structureMachineErrorAbsMSEFloor',0, ...
        'structureMachineErrorRelMSEFloor',0, ...
        'structureNeighborhoodMaxDistance',0.55,'structureNeighborhoodMinDistance',0.10, ...
        'structureNeighborhoodComplexityWindow',8,'structureFrontierMaxAbs',20, ...
        'multiOutputMode','native_single_fit_independent_output_archives', ...
        'populationBudgetMode','fixed_total', ...
        'innerNumRestarts',4, ...
        'innerNIterations',500,'innerPopulations',8, ...
        'innerRandomStateStride',100);
    options = merge_struct_local(defaults, options);
end

function candidate = make_candidate_local(model, source)
    candidate = struct('rank',1,'generation',0,'fitness',model.valMetrics.mse, ...
        'valLoss',model.valMetrics.mse,'complexity',getfield_default_local(model,'complexity',NaN), ...
        'model',model,'label',source,'skeletonSource',source);
end

function S = make_skeleton_set_local(coreExpr, selections)
    S = struct();
    S.coreExpressions = coreExpr;
    S.bestScoreExpressions = coreExpr;
    S.bestExpressions = coreExpr; % backward-compatible alias
    S.outputSelections = selections;
    S.source = 'per_output_sindy_or_native_multioutput_soft_validation_structure_score_selection';
end

function meta = selection_meta_local(selections, idx, fieldName)
    meta = struct();
    if isstruct(selections) && numel(selections) >= idx && isfield(selections(idx), fieldName)
        meta = selections(idx).(fieldName);
    end
end

function out = merge_struct_local(defaults, supplied)
    out = defaults;
    if ~isstruct(supplied); return; end
    names = fieldnames(supplied);
    for i = 1:numel(names)
        name = names{i};
        if isfield(out,name) && isstruct(out.(name)) && isstruct(supplied.(name))
            out.(name) = merge_struct_local(out.(name), supplied.(name));
        else
            out.(name) = supplied.(name);
        end
    end
end


function guesses = normalize_initial_guesses_local(value)
%NORMALIZE_INITIAL_GUESSES_LOCAL Return a flat, nonempty, stable cellstr.
    guesses = {};
    if isempty(value)
        return;
    end
    if ischar(value)
        items = {value};
    elseif isstring(value)
        items = cellstr(value(:));
    elseif iscell(value)
        items = value(:);
    else
        error('Stage-0 initial guesses must be char, string, or a cell array of strings.');
    end
    for iItem = 1:numel(items)
        item = items{iItem};
        if iscell(item)
            nested = normalize_initial_guesses_local(item);
            guesses = [guesses,nested]; %#ok<AGROW>
            continue;
        end
        if isstring(item) && ~isscalar(item)
            nested = normalize_initial_guesses_local(item);
            guesses = [guesses,nested]; %#ok<AGROW>
            continue;
        end
        text = strtrim(char(string(item)));
        if ~isempty(text) && ~any(strcmp(guesses,text))
            guesses{end+1} = text; %#ok<AGROW>
        end
    end
end

function list = normalize_list_local(value)
    if ischar(value) || (isstring(value) && isscalar(value)); list = {char(value)};
    elseif isstring(value); list = cellstr(value(:));
    elseif iscell(value); list = cellfun(@char,value,'UniformOutput',false);
    else; list = cellstr(string(value(:)));
    end
end

function text = value_text_local(value)
    if isempty(value)
        text = '<default>';
    elseif islogical(value)
        text = sprintf('%d',logical(value));
    elseif isnumeric(value) && isscalar(value)
        text = sprintf('%.6g',double(value));
    else
        text = char(string(value));
    end
end

function value = getfield_default_local(s, name, defaultValue)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name)); value = s.(name);
    else; value = defaultValue;
    end
end

function out = subset_output_data_local(data, outputIndices)
    out = data;
    fields = {'Ytr','Yval','Yte','Yood'};
    for i = 1:numel(fields)
        name = fields{i};
        if isfield(out,name) && ~isempty(out.(name))
            out.(name) = out.(name)(:,outputIndices);
        end
    end
end

function model = fit_sindy_per_output_local(dictionary,data,fitOptions)
%FIT_SINDY_PER_OUTPUT_LOCAL Select the SINDy threshold independently per output.
    ny = size(data.Ytr,2);
    fits = cell(1,ny);
    for j = 1:ny
        fits{j} = fit_stage0_dictionary(dictionary,subset_output_data_local(data,j),fitOptions);
    end
    model = fits{1};
    nTerms = size(fits{1}.Xi,1);
    model.Xi = zeros(nTerms,ny);
    model.XiScaled = zeros(nTerms,ny);
    model.activeMask = false(nTerms,ny);
    model.threshold = zeros(1,ny);
    model.outputExpressions = cell(1,ny);
    model.prediction.Ytr = zeros(size(data.Ytr));
    model.prediction.Yval = zeros(size(data.Yval));
    model.prediction.Yte = zeros(size(data.Yte));
    hasOod = isfield(data,'Yood') && ~isempty(data.Yood);
    if hasOod; model.prediction.Yood = zeros(size(data.Yood)); else; model.prediction.Yood = []; end
    for j = 1:ny
        fit = fits{j};
        model.Xi(:,j) = fit.Xi;
        model.XiScaled(:,j) = fit.XiScaled;
        model.activeMask(:,j) = fit.activeMask;
        model.threshold(j) = fit.threshold;
        model.outputExpressions{j} = fit.outputExpressions{1};
        model.prediction.Ytr(:,j) = fit.prediction.Ytr;
        model.prediction.Yval(:,j) = fit.prediction.Yval;
        model.prediction.Yte(:,j) = fit.prediction.Yte;
        if hasOod; model.prediction.Yood(:,j) = fit.prediction.Yood; end
    end
    model.nActiveCoefficients = nnz(model.activeMask);
    model.nActiveTerms = nnz(any(model.activeMask,2));
    model.complexity = model.nActiveCoefficients;
    model.trainMetrics = compute_regression_metrics(model.prediction.Ytr,data.Ytr);
    model.valMetrics = compute_regression_metrics(model.prediction.Yval,data.Yval);
    model.testMetrics = compute_regression_metrics(model.prediction.Yte,data.Yte);
    model.trainPerOutputMSE = mean((model.prediction.Ytr-data.Ytr).^2,1);
    model.valPerOutputMSE = mean((model.prediction.Yval-data.Yval).^2,1);
    model.testPerOutputMSE = mean((model.prediction.Yte-data.Yte).^2,1);
    model.trainPerOutputNormalizedMSE = normalized_output_mse(data.Ytr,model.prediction.Ytr);
    model.valPerOutputNormalizedMSE = normalized_output_mse(data.Yval,model.prediction.Yval);
    model.testPerOutputNormalizedMSE = normalized_output_mse(data.Yte,model.prediction.Yte);
    model.valLossMean = mean(model.valPerOutputNormalizedMSE);
    model.valLossWorst = max(model.valPerOutputNormalizedMSE);
    model.valLoss = model.valLossMean + getfield_default_local(fitOptions,'worstOutputWeight',0)*model.valLossWorst;
    model.fitness = model.valLoss;
    if hasOod
        model.oodMetrics = compute_regression_metrics(model.prediction.Yood,data.Yood);
        model.oodPerOutputMSE = mean((model.prediction.Yood-data.Yood).^2,1);
        model.oodPerOutputNormalizedMSE = normalized_output_mse(data.Yood,model.prediction.Yood);
    else
        model.oodMetrics = struct('mse',NaN,'rmse',NaN,'mae',NaN,'r2',NaN);
        model.oodPerOutputMSE = [];
        model.oodPerOutputNormalizedMSE = [];
    end
end

function merged = combine_per_output_bypass_and_sr_local(baseFit, sr, bypassMask, srIndices, data)
%COMBINE_PER_OUTPUT_BYPASS_AND_SR_LOCAL Restore original output ordering.
    ny = numel(bypassMask);
    expressions = baseFit.outputExpressions;
    srExpressions = getfield_default_local(sr,'bestStructureExpressionsPerOutput', ...
        getfield_default_local(sr,'coreExpressionsPerOutput',{}));
    expressions(srIndices) = srExpressions;

    YhatTrain = baseFit.prediction.Ytr;
    YhatVal = baseFit.prediction.Yval;
    YhatTest = baseFit.prediction.Yte;
    YhatOod = baseFit.prediction.Yood;
    YhatTrain(:,srIndices) = sr.YhatTrain;
    YhatVal(:,srIndices) = sr.YhatVal;
    YhatTest(:,srIndices) = sr.YhatTest;
    if ~isempty(YhatOod) && isfield(sr,'YhatOod') && ~isempty(sr.YhatOod)
        YhatOod(:,srIndices) = sr.YhatOod;
    end

    template = struct('outputIndex',NaN,'core',struct(),'bestScore',struct(), ...
        'best',struct(),'candidates',struct([]),'validationRanking',struct([]), ...
        'complexityRanking',struct([]),'bestScoreRanking',struct([]), ...
        'structureScoreRanking',struct([]),'candidateCount',0,'selectionMode','', ...
        'selectedInnerRestart',NaN,'selectedInnerSeed',NaN);
    selections = repmat(template,1,ny);
    srSelections = getfield_default_local(sr,'outputSelections',struct([]));
    srPosition = zeros(1,ny);
    srPosition(srIndices) = 1:numel(srIndices);
    for j = 1:ny
        if bypassMask(j)
            complexity = nnz(baseFit.activeMask(:,j));
            core = struct('rank',1,'rank_mode','per_output_sindy_bypass', ...
                'expression',baseFit.outputExpressions{j},'complexity',complexity, ...
                'train_mse',baseFit.trainPerOutputMSE(j), ...
                'validation_mse',baseFit.valPerOutputMSE(j), ...
                'loss',baseFit.valPerOutputMSE(j),'score',NaN, ...
                'structure_score',NaN,'frontier_score_raw',NaN, ...
                'frontier_score_normalized',NaN, ...
                'validation_soft_score',NaN, ...
                'structure_eligible',false,'frontier_score_valid',false, ...
                'selection_role','sindy-bypass', ...
                'selection_rule','per_output_fixed_sindy_validation_bypass', ...
                'ranking_scope','per_output_sindy_bypass', ...
                'relative_error_scope','per_output_sindy_bypass', ...
                'structure_signature',structure_signature_local(baseFit.outputExpressions{j}));
            core.relative_to_best_validation_mse = 1;
            selections(j).outputIndex = j;
            selections(j).core = core;
            selections(j).bestScore = core;
            selections(j).best = core;
            selections(j).candidates = core;
            selections(j).validationRanking = core;
            selections(j).complexityRanking = core;
            selections(j).bestScoreRanking = core;
            selections(j).candidateCount = 1;
            selections(j).selectionMode = 'per_output_fixed_sindy_bypass';
        else
            source = srSelections(srPosition(j));
            selections(j) = normalize_selection_local(source,template,j);
        end
    end

    merged = sr;
    merged.bestStructureExpressionsPerOutput = expressions;
    merged.bestScoreExpressionsPerOutput = expressions; % backward-compatible alias
    merged.coreExpressionsPerOutput = expressions;
    merged.bestExpressionsPerOutput = expressions;
    merged.outputSelections = selections;
    merged.YhatTrain = YhatTrain;
    merged.YhatVal = YhatVal;
    merged.YhatTest = YhatTest;
    merged.YhatOod = YhatOod;
    merged.trainMetrics = compute_regression_metrics(YhatTrain,data.Ytr);
    merged.valMetrics = compute_regression_metrics(YhatVal,data.Yval);
    merged.validationMSE = merged.valMetrics.mse;
    merged.testMetrics = compute_regression_metrics(YhatTest,data.Yte);
    if isfield(data,'Yood') && ~isempty(data.Yood)
        merged.oodMetrics = compute_regression_metrics(YhatOod,data.Yood);
    else
        merged.oodMetrics = struct('mse',NaN,'rmse',NaN);
    end
    merged.complexity = total_selected_complexity_local(selections,NaN);
    merged.nActiveCoefficients = merged.complexity;
    merged.selectionMode = 'per_output_sindy_bypass_plus_soft_validation_structure_score';
    merged.bypassOutputMask = bypassMask;
    merged.pysrOutputIndices = srIndices;
    merged.candidateRankingsByValidation = remap_ranking_outputs_local( ...
        getfield_default_local(sr,'candidateRankingsByValidation',struct([])),srIndices);
    merged.candidateRankingsByComplexity = remap_ranking_outputs_local( ...
        getfield_default_local(sr,'candidateRankingsByComplexity',struct([])),srIndices);
    merged.candidateRankingsByBestScore = remap_ranking_outputs_local( ...
        getfield_default_local(sr,'candidateRankingsByBestScore',struct([])),srIndices);
    merged.candidateRankingsByStructureScore = collect_structure_rankings_local(selections);
end

function selection = normalize_selection_local(source,template,outputIndex)
    selection = template;
    names = fieldnames(template);
    for i = 1:numel(names)
        if isfield(source,names{i}); selection.(names{i}) = source.(names{i}); end
    end
    selection.outputIndex = outputIndex;
end

function rows = remap_ranking_outputs_local(rows,originalIndices)
    if ~isstruct(rows); rows = struct([]); return; end
    for i = 1:numel(rows)
        localIndex = getfield_default_local(rows(i),'outputIndex',1);
        if isfinite(localIndex) && localIndex >= 1 && localIndex <= numel(originalIndices)
            rows(i).outputIndex = originalIndices(localIndex);
        end
    end
end



function merged = merge_inner_restart_results_local(restartResults, restartSeeds, restartTimes, data, pysrOptions)
%MERGE_INNER_RESTART_RESULTS_LOCAL Merge validation-ratio-qualified pools.
% One robust, two-sided, multi-scale frontier-prominence score is computed in
% each Python run from already evaluated Pareto candidates. This merge
% de-duplicates exact constant-free skeletons, aggregates repeated evidence,
% applies robust sigmoid normalization (not tiny-pool percentiles), adds
% continuous validation evidence inside the hard rho pool, and selects the
% largest score without any new model fits.

    nRestarts = numel(restartResults);
    if nRestarts < 1
        error('No Stage-0 restart results were supplied.');
    end
    merged = restartResults{1};
    ny = size(data.Ytr, 2);
    useStructureScore = logical(getfield_default_local(pysrOptions, 'structureScoreEnable', true));
    validationMultiplier = getfield_default_local(pysrOptions, 'structureValidationMultiplier', 4.0);
    if ~isscalar(validationMultiplier) || ~isfinite(validationMultiplier) || validationMultiplier < 1
        validationMultiplier = 4.0;
    end
    validationWeight = getfield_default_local(pysrOptions, 'structureValidationWeight', 0.20);
    if ~isscalar(validationWeight) || ~isfinite(validationWeight)
        validationWeight = 0.20;
    end
    validationWeight = min(max(validationWeight, 0), 1);
    machineAbsFloor = getfield_default_local( ...
        pysrOptions, 'structureMachineErrorAbsMSEFloor', 0);
    machineRelFloor = getfield_default_local( ...
        pysrOptions, 'structureMachineErrorRelMSEFloor', 0);
    if ~isscalar(machineAbsFloor) || ~isfinite(machineAbsFloor) || machineAbsFloor < 0
        machineAbsFloor = 0;
    end
    if ~isscalar(machineRelFloor) || ~isfinite(machineRelFloor) || machineRelFloor < 0
        machineRelFloor = 0;
    end

    selectedRestart = ones(1, ny);
    selectedExpressions = cell(1, ny);
    selectedSelections = struct([]);
    YhatTrain = zeros(size(data.Ytr));
    YhatVal = zeros(size(data.Yval));
    YhatTest = zeros(size(data.Yte));
    hasOod = isfield(data,'Yood') && ~isempty(data.Yood);
    if hasOod; YhatOod = zeros(size(data.Yood)); else; YhatOod = []; end

    for j = 1:ny
        pool = cell(1,0);
        for r = 1:nRestarts
            rr = restartResults{r};
            candidates = struct([]);
            if isfield(rr,'outputSelections') && numel(rr.outputSelections) >= j
                candidates = getfield_default_local(rr.outputSelections(j), 'candidates', struct([]));
            end
            if ~isstruct(candidates) || isempty(candidates)
                core = getfield_default_local(rr.outputSelections(j), 'core', struct());
                if ~isempty(fieldnames(core)); candidates = core; end
            end
            for k = 1:numel(candidates)
                candidate = candidates(k);
                candidate.inner_restart = r;
                candidate.inner_seed = restartSeeds(r);
                signature = char(string(getfield_default_local(candidate, 'structure_signature', '')));
                if isempty(signature)
                    signature = structure_signature_local(getfield_default_local(candidate, 'expression', ''));
                    candidate.structure_signature = signature;
                end
                pool{end+1} = candidate; %#ok<AGROW>
            end
        end
        if isempty(pool)
            error('Stage-0 restart merge found no exported candidates for output y%d.', j);
        end

        % Constant-free exact skeleton de-duplication. Keep the lowest-validation
        % representative and aggregate its local structural components.
        signatures = cellfun(@(c) char(string(c.structure_signature)), pool, 'UniformOutput', false);
        uniqueSignatures = unique(signatures, 'stable');
        representatives = cell(1, numel(uniqueSignatures));
        frontierSamples = cell(1, numel(uniqueSignatures));
        for u = 1:numel(uniqueSignatures)
            members = find(strcmp(signatures, uniqueSignatures{u}));
            vals = cellfun(@(c) scalar_or_inf_local(getfield_default_local(c,'validation_mse',Inf)), pool(members));
            [~,bestLocal] = min(vals);
            representatives{u} = pool{members(bestLocal)};
            frontierSamples{u} = cellfun(@(c) candidate_structure_component_local(c,'frontier_score_raw'), pool(members));
        end

        validationValues = cellfun(@(c) scalar_or_inf_local(getfield_default_local(c,'validation_mse',Inf)), representatives);
        complexityValues = cellfun(@(c) scalar_or_inf_local(getfield_default_local(c,'complexity',Inf)), representatives);
        targetEnergy = mean(data.Yval(:,j).^2, 'omitnan');
        if ~isfinite(targetEnergy); targetEnergy = 0; end
        selectionMSEFloor = max([machineAbsFloor, ...
            machineRelFloor*max(1,targetEnergy), realmin]);
        effectiveValidationValues = max(validationValues, selectionMSEFloor);
        [~,validationOrder] = sortrows([effectiveValidationValues(:), ...
            complexityValues(:), validationValues(:)], [1 2 3]);
        bestValidationRaw = min(validationValues);
        bestValidation = effectiveValidationValues(validationOrder(1));
        validationLimit = max(bestValidation * validationMultiplier, selectionMSEFloor);
        eligible = validationOrder(effectiveValidationValues(validationOrder) <= validationLimit);
        if isempty(eligible); eligible = validationOrder(1); end

        frontierRaw = NaN(1,numel(eligible));
        for q = 1:numel(eligible)
            u = eligible(q);
            frontierRaw(q) = finite_median_or_nan_local(frontierSamples{u});
        end
        frontierNormalized = robust_unit_scores_local(frontierRaw);
        validationSoft = reshape(arrayfun(@(u) soft_validation_score_local( ...
            effectiveValidationValues(u), bestValidation, validationMultiplier), eligible), 1, []);
        combined = (1-validationWeight)*frontierNormalized + ...
            validationWeight*validationSoft;
        if any(isfinite(frontierRaw))
            combined(~isfinite(frontierRaw)) = -realmax;
        end
        validScoreMask = combined > -realmax/2;
        if any(validScoreMask)
            eligible = eligible(validScoreMask);
            combined = combined(validScoreMask);
            frontierRaw = frontierRaw(validScoreMask);
            frontierNormalized = frontierNormalized(validScoreMask);
            validationSoft = validationSoft(validScoreMask);
        end
        orderMatrix = [-combined(:), ...
            reshape(effectiveValidationValues(eligible),[],1), ...
            reshape(complexityValues(eligible),[],1), ...
            reshape(validationValues(eligible),[],1)];
        [~,scoreOrder] = sortrows(orderMatrix, [1 2 3 4]);
        eligible = eligible(scoreOrder);
        combined = combined(scoreOrder);
        frontierRaw = frontierRaw(scoreOrder);
        frontierNormalized = frontierNormalized(scoreOrder);
        validationSoft = validationSoft(scoreOrder);

        structureRows = struct([]);
        for rankValue = 1:numel(eligible)
            candidate = representatives{eligible(rankValue)};
            candidate.rank = rankValue;
            candidate.rank_mode = 'structure_score';
            if bestValidationRaw <= realmin
                candidate.relative_to_best_validation_mse = double(validationValues(eligible(rankValue)) > realmin) * Inf;
                if validationValues(eligible(rankValue)) <= realmin
                    candidate.relative_to_best_validation_mse = 1;
                end
            else
                candidate.relative_to_best_validation_mse = validationValues(eligible(rankValue)) / bestValidation;
            end
            candidate.frontier_score_raw = frontierRaw(rankValue);
            candidate.frontier_score_valid = isfinite(frontierRaw(rankValue));
            candidate.frontier_score_normalized = frontierNormalized(rankValue);
            candidate.validation_soft_score = validationSoft(rankValue);
            candidate.structure_score = combined(rankValue);
            candidate.selection_mse_floor = selectionMSEFloor;
            candidate.validation_mse_effective = ...
                effectiveValidationValues(eligible(rankValue));
            candidate.validation_floor_tied = ...
                validationValues(eligible(rankValue)) <= selectionMSEFloor;
            candidate.selection_prefers_simplicity_within_floor = true;
            candidate.structure_eligible = true;
            candidate.selection_role = 'none';
            candidate.ranking_scope = 'cross_restart_global';
            candidate.relative_error_scope = 'cross_restart_global_validation_best';
            candidate = remove_fields_if_present_local(candidate, {'prediction_paths','compile_fallback_rank','inner_seed'});
            if isempty(structureRows); structureRows = candidate; else; structureRows(end+1) = candidate; end %#ok<AGROW>
        end

        floorCandidates = find(validationValues <= selectionMSEFloor);
        usedMachineFloorTie = ~isempty(floorCandidates);
        if usedMachineFloorTie
            % Validation differences below the scale-aware numerical floor are
            % not scientific evidence for a larger tree.  Across restarts,
            % select the simplest semantically distinct candidate; raw MSE is
            % only the last deterministic tie-breaker.
            [~,floorOrder] = sortrows([complexityValues(floorCandidates).', ...
                validationValues(floorCandidates).'], [1 2]);
            selectedRepresentativeIndex = floorCandidates(floorOrder(1));
            selectedCandidate = representatives{selectedRepresentativeIndex};
            selectedCandidate.frontier_score_raw = ...
                candidate_structure_component_local(selectedCandidate,'frontier_score_raw');
            selectedCandidate.frontier_score_valid = ...
                isfinite(selectedCandidate.frontier_score_raw);
            selectedCandidate.frontier_score_normalized = 0.5;
            selectedCandidate.validation_soft_score = 1.0;
            selectedCandidate.structure_score = 1.0;
            selectedCandidate.selection_rule = 'machine_floor_simplicity_tie';
            selectedCandidate.selection_mse_floor = selectionMSEFloor;
            selectedCandidate.validation_mse_effective = selectionMSEFloor;
            selectedCandidate.validation_floor_tied = true;
            selectedCandidate.selection_prefers_simplicity_within_floor = true;
            selectedCandidate.selection_role = 'final-structure-core';
            selectedCandidate.ranking_scope = 'cross_restart_global';
            selectedCandidate.relative_error_scope = 'cross_restart_global_validation_best';

            % Rebuild the exported selection table from every floor-tied
            % candidate, including frontier endpoints that the ordinary
            % two-sided structure score may exclude.
            floorCandidates = floorCandidates(floorOrder);
            structureRows = struct([]);
            for q = 1:numel(floorCandidates)
                u = floorCandidates(q);
                candidate = representatives{u};
                candidate.rank = q;
                candidate.rank_mode = 'machine_floor_simplicity_tie';
                candidate.frontier_score_raw = ...
                    candidate_structure_component_local(candidate,'frontier_score_raw');
                candidate.frontier_score_valid = isfinite(candidate.frontier_score_raw);
                candidate.frontier_score_normalized = 0.5;
                candidate.validation_soft_score = 1.0;
                candidate.structure_score = 1.0;
                candidate.selection_rule = 'machine_floor_simplicity_tie';
                candidate.selection_mse_floor = selectionMSEFloor;
                candidate.validation_mse_effective = selectionMSEFloor;
                candidate.validation_floor_tied = true;
                candidate.selection_prefers_simplicity_within_floor = true;
                candidate.structure_eligible = true;
                candidate.selection_role = 'none';
                if q == 1; candidate.selection_role = 'final-structure-core'; end
                candidate.ranking_scope = 'cross_restart_global';
                candidate.relative_error_scope = 'cross_restart_global_validation_best';
                candidate = remove_fields_if_present_local(candidate, ...
                    {'prediction_paths','compile_fallback_rank','inner_seed'});
                if isempty(structureRows)
                    structureRows = candidate;
                else
                    structureRows(end+1) = candidate; %#ok<AGROW>
                end
            end
        elseif useStructureScore
            selectedCandidate = representatives{eligible(1)};
            selectedCandidate.frontier_score_raw = frontierRaw(1);
            selectedCandidate.frontier_score_valid = isfinite(frontierRaw(1));
            selectedCandidate.frontier_score_normalized = frontierNormalized(1);
            selectedCandidate.validation_soft_score = validationSoft(1);
            selectedCandidate.structure_score = combined(1);
            selectedCandidate.selection_rule = 'machine_floor_aware_soft_validation_structure_score';
            selectedCandidate.selection_mse_floor = selectionMSEFloor;
            selectedCandidate.validation_mse_effective = ...
                effectiveValidationValues(eligible(1));
            selectedCandidate.validation_floor_tied = false;
            selectedCandidate.selection_prefers_simplicity_within_floor = true;
            selectedCandidate.selection_role = 'final-structure-core';
            selectedCandidate.ranking_scope = 'cross_restart_global';
            selectedCandidate.relative_error_scope = 'cross_restart_global_validation_best';
        else
            % Compatibility mode: choose the per-restart core with the lowest
            % external validation MSE, matching the former wrapper behavior.
            coreCandidates = cellfun(@(c) strcmp(char(string( ...
                getfield_default_local(c,'selection_role',''))), 'restart-local-core'), pool);
            candidateIndices = find(coreCandidates);
            if isempty(candidateIndices); candidateIndices = 1:numel(pool); end
            vals = cellfun(@(c) scalar_or_inf_local(getfield_default_local(c,'validation_mse',Inf)), pool(candidateIndices));
            [~,ii] = min(vals);
            selectedCandidate = pool{candidateIndices(ii)};
        end

        selectedRestart(j) = selectedCandidate.inner_restart;
        rr = restartResults{selectedRestart(j)};
        % Compile the full-precision SymPy serialization when available, while
        % retaining the native searched expression for ranking/reporting.  The
        % latter may round constants in PySR's display string and can lose
        % exact numerical reproduction in large, ill-conditioned trees.
        selectedExpressions{j} = char(string(getfield_default_local( ...
            selectedCandidate, 'compiler_expression', selectedCandidate.expression)));
        selectedSelection = rr.outputSelections(j);
        selectedSelection.core = remove_fields_if_present_local(selectedCandidate, ...
            {'prediction_paths','compile_fallback_rank','inner_restart','inner_seed'});
        selectedSelection.core.selection_role = 'final-structure-core';
        selectedSelection.core.ranking_scope = 'cross_restart_global';
        selectedSelection.core.relative_error_scope = 'cross_restart_global_validation_best';
        selectedSelection.bestScore = selectedSelection.core; % backward-compatible field name
        selectedSelection.best = selectedSelection.core;
        if usedMachineFloorTie
            selectedSelection.selectionMode = 'machine_floor_simplicity_tie';
        elseif useStructureScore
            selectedSelection.selectionMode = 'machine_floor_aware_soft_validation_structure_score';
        else
            selectedSelection.selectionMode = 'legacy_restart_core_validation_mse';
        end
        selectedSelection.selectedInnerRestart = selectedRestart(j);
        selectedSelection.selectedInnerSeed = restartSeeds(selectedRestart(j));
        selectedSelection.validationRanking = relabel_restart_local_rows_local( ...
            getfield_default_local(selectedSelection,'validationRanking',struct([])), selectedCandidate);
        selectedSelection.complexityRanking = relabel_restart_local_rows_local( ...
            getfield_default_local(selectedSelection,'complexityRanking',struct([])), selectedCandidate);
        selectedSelection.bestScoreRanking = relabel_restart_local_rows_local( ...
            getfield_default_local(selectedSelection,'bestScoreRanking',struct([])), selectedCandidate);
        selectedSelection.candidates = relabel_restart_local_rows_local( ...
            getfield_default_local(selectedSelection,'candidates',struct([])), selectedCandidate);
        selectedSelection.structureScoreRanking = structureRows;
        selectedSelection.candidateCount = numel(uniqueSignatures);
        if ~isempty(structureRows)
            selectedSelection.structureScoreRanking(1).selection_role = 'final-structure-core';
        end
        if j == 1
            selectedSelections = repmat(selectedSelection, 1, ny);
        else
            selectedSelections(j) = selectedSelection;
        end

        paths = getfield_default_local(selectedCandidate, 'prediction_paths', struct());
        if isempty(fieldnames(paths))
            error('Selected Stage-0 structure candidate for y%d has no exported prediction paths.', j);
        end
        YhatTrain(:,j) = read_prediction_vector_local(paths.train, size(data.Ytr,1));
        YhatVal(:,j) = read_prediction_vector_local(paths.validation, size(data.Yval,1));
        YhatTest(:,j) = read_prediction_vector_local(paths.test, size(data.Yte,1));
        if hasOod
            YhatOod(:,j) = read_prediction_vector_local(paths.ood, size(data.Yood,1));
        end
    end

    merged.bestStructureExpressionsPerOutput = selectedExpressions;
    merged.bestScoreExpressionsPerOutput = selectedExpressions; % backward-compatible alias
    merged.coreExpressionsPerOutput = selectedExpressions;
    merged.bestExpressionsPerOutput = selectedExpressions;
    merged.outputSelections = selectedSelections;
    merged.candidateRankingsByStructureScore = collect_structure_rankings_local(selectedSelections);
    merged.YhatTrain = YhatTrain;
    merged.YhatVal = YhatVal;
    merged.YhatTest = YhatTest;
    merged.YhatOod = YhatOod;
    merged.trainMetrics = compute_regression_metrics(YhatTrain, data.Ytr);
    merged.valMetrics = compute_regression_metrics(YhatVal, data.Yval);
    merged.validationMSE = merged.valMetrics.mse;
    merged.testMetrics = compute_regression_metrics(YhatTest, data.Yte);
    if hasOod
        merged.oodMetrics = compute_regression_metrics(YhatOod, data.Yood);
    else
        merged.oodMetrics = struct('mse',NaN,'rmse',NaN);
    end
    merged.complexity = total_selected_complexity_local(selectedSelections, NaN);
    merged.nActiveCoefficients = merged.complexity;
    if useStructureScore
        merged.selectionMode = 'machine_floor_aware_soft_validation_structure_score';
    else
        merged.selectionMode = 'legacy_restart_core_validation_mse';
    end
    merged.innerRestart = struct('enabled',true,'numRestarts',nRestarts, ...
        'seeds',restartSeeds(:).','times',restartTimes(:).', ...
        'selectedRestartPerOutput',selectedRestart);
    merged.innerRestartResults = restartResults;
    merged.trainTime = sum(restartTimes);
    if ~isfield(merged,'timeStats') || ~isstruct(merged.timeStats)
        merged.timeStats = struct();
    end
    merged.timeStats.innerRestartTimes = restartTimes(:).';
    merged.timeStats.innerRestartTotal = sum(restartTimes);
    merged.timeStats.total = sum(restartTimes);
end

function signature = structure_signature_local(expression)
    signature = lower(regexprep(char(string(expression)), ...
        '(?<![A-Za-z_])(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', '#'));
    signature = regexprep(signature, '\s+', '');
end

function value = scalar_or_inf_local(value)
    if ~(isnumeric(value) && isscalar(value) && isfinite(value)); value = Inf; end
end

function value = scalar_or_nan_local(value)
    if ~(isnumeric(value) && isscalar(value) && isfinite(value)); value = NaN; end
end

function value = candidate_structure_component_local(candidate, fieldName)
    if ~logical(getfield_default_local(candidate, 'structure_eligible', false))
        value = NaN;
        return;
    end
    if strcmp(fieldName,'frontier_score_raw') && ~logical(getfield_default_local(candidate,'frontier_score_valid',true))
        value = NaN;
        return;
    end
    value = scalar_or_nan_local(getfield_default_local(candidate, fieldName, NaN));
end

function value = finite_median_or_nan_local(values)
    values = values(isfinite(values));
    if isempty(values); value = NaN; else; value = median(values); end
end

function score = robust_unit_scores_local(values)
%ROBUST_UNIT_SCORES_LOCAL Robust sigmoid scaling for small merged pools.
% A singleton or a tied pool is neutral (0.5); unlike empirical percentiles,
% two candidates are not forced toward artificial endpoint confidence.
    score = zeros(size(values));
    finiteMask = isfinite(values);
    finiteValues = values(finiteMask);
    nFinite = numel(finiteValues);
    if nFinite == 0; return; end
    if nFinite == 1
        score(finiteMask) = 0.5;
        return;
    end
    center = median(finiteValues);
    scale = 1.4826 * median(abs(finiteValues - center));
    if ~isfinite(scale) || scale <= eps
        scale = std(finiteValues);
    end
    if ~isfinite(scale) || scale <= eps
        score(finiteMask) = 0.5;
        return;
    end
    z = max(min((finiteValues-center)/scale, 8), -8);
    score(finiteMask) = 1 ./ (1 + exp(-z));
end

function score = soft_validation_score_local(validationMSE, bestValidationMSE, validationMultiplier)
%SOFT_VALIDATION_SCORE_LOCAL Continuous validation evidence within rho.
% The best validation MSE maps to 1 and rho times best maps to 0, with a
% linear transition in log-error ratio. The hard rho eligibility gate remains.
    if ~isscalar(validationMSE) || ~isfinite(validationMSE) || validationMSE < 0
        score = 0;
        return;
    end
    if bestValidationMSE <= realmin
        score = double(validationMSE <= realmin);
        return;
    end
    if validationMultiplier <= 1
        score = double(validationMSE <= bestValidationMSE);
        return;
    end
    ratio = max(1, validationMSE / bestValidationMSE);
    score = min(max(1 - log(ratio)/log(validationMultiplier), 0), 1);
end

function out = remove_fields_if_present_local(in, names)
    out = in;
    for i = 1:numel(names)
        if isfield(out,names{i}); out = rmfield(out,names{i}); end
    end
end

function rows = relabel_restart_local_rows_local(rows, finalCandidate)
%RELABEL_RESTART_LOCAL_ROWS_LOCAL Remove stale pre-merge winner labels.
% These rows belong to the selected restart and keep their restart-local
% relative errors. Only the exact cross-restart winner may be labeled final.
    if ~isstruct(rows) || isempty(rows); return; end
    finalExpression = char(string(getfield_default_local(finalCandidate,'expression','')));
    for i = 1:numel(rows)
        rows(i).selection_role = 'none';
        rows(i).ranking_scope = 'selected_restart_local';
        rows(i).relative_error_scope = 'restart_local_validation_best';
        expression = char(string(getfield_default_local(rows(i),'expression','')));
        if ~isempty(finalExpression) && strcmp(expression, finalExpression)
            rows(i).selection_role = 'final-structure-core';
        end
    end
end

function vector = read_prediction_vector_local(path, expectedRows)
    if exist(path,'file') ~= 2
        error('Stage-0 candidate prediction file is missing: %s', path);
    end
    if exist('readmatrix','file') == 2; vector = readmatrix(path); else; vector = csvread(path); end
    vector = vector(:);
    if numel(vector) ~= expectedRows
        error('Stage-0 candidate prediction size mismatch for %s: expected %d, found %d.', ...
            path, expectedRows, numel(vector));
    end
end

function report = assert_stage0_repeatability_local(a, b, tol)
    exprA = getfield_default_local(a, 'bestScoreExpressionsPerOutput', {});
    exprB = getfield_default_local(b, 'bestScoreExpressionsPerOutput', {});
    if ~isequal(exprA, exprB)
        error(['Strict deterministic Stage-0 repeatability check failed: selected ', ...
            'structure-score core expressions differ between two identical searches.']);
    end
    fields = {'YhatTrain','YhatVal','YhatTest','YhatOod'};
    maxDiff = 0;
    for i = 1:numel(fields)
        A = getfield_default_local(a, fields{i}, []);
        B = getfield_default_local(b, fields{i}, []);
        if ~isequal(size(A), size(B))
            error('Strict deterministic Stage-0 repeatability check failed: %s size differs.', fields{i});
        end
        if ~isempty(A)
            d = max(abs(A(:) - B(:)));
            if isempty(d); d = 0; end
            maxDiff = max(maxDiff, d);
        end
    end
    if ~isfinite(maxDiff) || maxDiff > tol
        error(['Strict deterministic Stage-0 repeatability check failed: max prediction ', ...
            'difference %.3e exceeds tolerance %.3e.'], maxDiff, tol);
    end
    report = struct('passed',true,'maxPredictionDifference',maxDiff, ...
        'predictionTolerance',tol,'coreExpressions',{exprA});
end

function rankings = collect_structure_rankings_local(selections)
%COLLECT_STRUCTURE_RANKINGS_LOCAL Merge per-output ranking rows safely.
% Candidate records exported by PySR can contain output-dependent optional
% metadata fields. MATLAB requires identical field names and field order for
% indexed struct-array assignment, so harmonize every incoming row before
% appending it to the cross-output table.
    rankings = struct([]);
    for j = 1:numel(selections)
        rows = getfield_default_local(selections(j), 'structureScoreRanking', struct([]));
        if ~isstruct(rows) || isempty(rows); continue; end
        for i = 1:numel(rows)
            row = rows(i);
            row.outputIndex = getfield_default_local(selections(j), 'outputIndex', j);
            if isempty(rankings)
                rankings = row;
            else
                [rankings,row] = harmonize_struct_fields_local(rankings,row);
                rankings(end+1) = row; %#ok<AGROW>
            end
        end
    end
end

function [left,right] = harmonize_struct_fields_local(left,right)
%HARMONIZE_STRUCT_FIELDS_LOCAL Add missing optional fields before concatenation.
    leftNames = fieldnames(left);
    rightNames = fieldnames(right);
    missingFromLeft = setdiff(rightNames,leftNames,'stable');
    missingFromRight = setdiff(leftNames,rightNames,'stable');

    for i = 1:numel(missingFromLeft)
        [left.(missingFromLeft{i})] = deal([]);
    end
    for i = 1:numel(missingFromRight)
        right.(missingFromRight{i}) = [];
    end

    % After both structs have the same fields, normalize field order as well.
    left = orderfields(left);
    right = orderfields(right);
end

function total = total_selected_complexity_local(selections, fallback)
    total = 0;
    found = false;
    if isstruct(selections)
        for i = 1:numel(selections)
            core = getfield_default_local(selections(i), 'core', struct());
            if isempty(fieldnames(core)); core = getfield_default_local(selections(i), 'bestScore', struct()); end
            c = getfield_default_local(core, 'complexity', NaN);
            if isnumeric(c) && isscalar(c) && isfinite(c)
                total = total + c;
                found = true;
            end
        end
    end
    if ~found; total = fallback; end
end

function tf = is_single_generator_dynamic_case_local(task)
    tf = false;
    if ~isstruct(task); return; end
    names = {};
    for fieldName = {'name','caseId','modelVariant'}
        f = fieldName{1};
        if isfield(task,f) && ~isempty(task.(f))
            names{end+1} = lower(char(string(task.(f)))); %#ok<AGROW>
        end
    end
    if isfield(task,'parameters') && isstruct(task.parameters) && ...
            isfield(task.parameters,'modelVariant')
        names{end+1} = lower(char(string(task.parameters.modelVariant))); %#ok<AGROW>
    end
    joined = strjoin(names,' ');
    tf = contains(joined,'singlegeneratordynamic') || ...
        contains(joined,'single_generator_dynamic') || ...
        contains(joined,'smib_avr') || contains(joined,'salient_pole');
end
