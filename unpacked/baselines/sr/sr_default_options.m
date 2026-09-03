function opts = sr_default_options()
%SR_DEFAULT_OPTIONS Internal options for the official PySR engine.
%
% PhDN Stage 0 calls the official Python PySRRegressor package through this
% MATLAB/Python backend.  The demos no longer launch a second independent SR
% baseline; Stage0-SR ablation metrics are collected from the existing search.

    opts = struct();

    % Generic backend labels. Stage 0 overrides these with its own role/title.
    opts.reportRole = 'official_pysr_engine';
    opts.reportTitle = 'Official PySR engine result';

    % Prior/operator source label.  This does not give PySR PhDN's layerwise
    % architecture; it only records which prior level the operator choice is
    % intended to be comparable with.
    opts.grammarCasemode = 'weak_prior_lv1';

    % Only supported symbolic-regression backend.
    opts.searchBackend = 'official_pysr';

    % Python/PySR execution controls.  Set pythonExe to the Python executable
    % in the conda environment where `pysr` is installed.
    opts.pythonExe = 'python';
    opts.pysrPaperRoot = '';
    opts.workRoot = fullfile(tempdir, 'phdnn_official_pysr_runs');
    opts.keepWorkDir = true;

    % PySR 2 initial-equation support.  The framework pins the current 2.0
    % pre-release because equation guesses are not available in PySR 1.x.
    opts.minimumPySRVersion = '2.0.0a2';
    opts.requirePySR2 = true;
    opts.initialGuessesEnable = false;
    opts.initialGuesses = {};
    opts.fractionReplacedGuesses = 0.05;
    opts.initialGuessScope = 'shared_all_unresolved_outputs';

    % Optional official PySR growth/selection controls. Empty values preserve
    % PySR defaults. Case demos may expose these as user-editable settings.
    opts.tournamentSelectionN = [];
    opts.tournamentSelectionP = [];
    opts.crossoverProbability = [];
    opts.weightAddNode = [];
    opts.weightInsertNode = [];
    opts.weightDeleteNode = [];
    opts.weightDoNothing = [];
    opts.weightMutateConstant = [];
    opts.weightMutateOperator = [];
    opts.weightMutateFeature = [];
    opts.weightSwapOperands = [];
    opts.weightRotateTree = [];
    opts.weightRandomize = [];
    opts.weightSimplify = [];
    opts.weightOptimize = [];
    opts.optimizeProbability = [];
    opts.shouldSimplify = [];

    % Scale-aware machine-precision early stopping.  Native multi-output PySR
    % is advanced in warm-start chunks, and a restart stops only when every
    % unresolved output reaches its full-training Hall-of-Fame MSE threshold.
    % The Stage-0 wrapper may then stop the remaining independent restarts only
    % after the same threshold is also verified on the external validation set.
    opts.machinePrecisionEarlyStopEnable = false;
    opts.machinePrecisionEarlyStopAbsMSE = 1e-12;
    opts.machinePrecisionEarlyStopRelMSE = 1e-12;
    opts.machinePrecisionEarlyStopCheckInterval = 50;
    opts.machinePrecisionEarlyStopMinIterations = 50;
    opts.machinePrecisionEarlyStopAcrossRestarts = true;

    % Official PySRRegressor options.
    opts.nIterations = 100;
    opts.populations = 8;
    % A 2-D Y is sent through one native PySR fit. Each output still owns an
    % independent equation archive and loss; only the Julia worker scheduler is
    % shared. Stage 0 may replace this with a fixed-total population budget.
    opts.multiOutputMode = 'native_single_fit_independent_output_archives';
    opts.populationBudgetMode = 'per_output';
    opts.populationSize = 50;
    opts.maxSize = 30;
    opts.maxDepth = 10;
    opts.parsimony = 1e-6;
    opts.modelSelection = 'best';
    opts.randomState = 1;
    opts.deterministic = false;
    opts.parallelism = 'multithreading';
    opts.verbosity = 1;
    opts.progress = false;

    % Operator family.  The adapter maps 'inv' and 'sqrt' to protected custom
    % operators for numerical robustness when input/expression values cross
    % singular or negative domains.
    opts.binaryOperators = {'+', '-', '*', '/'};
    opts.unaryOperators = {'inv', 'sqrt', 'log'};
    opts.operatorComplexities = struct();

    % Equation-table reporting.  These options only affect what is reported
    % back to MATLAB; PySR still performs its own model selection internally.
    opts.topKExpressionsToReport = 10;
    % Display-only Pareto rankings. candidateRankingTopK limits table rows only;
    % it does not limit the structure-score candidate pool.
    opts.displayCandidateRankings = false;
    opts.candidateRankingTopK = 20;
    opts.equationLossMultiplier = 10;
    opts.maxReportComplexity = 1000;

    % Semantic de-duplication before core selection.
    opts.semanticDedupTolerance = 1e-8;

    % Optional PhDN Stage-0 post-search structure selection. Independent SR
    % baselines keep this disabled; the PhDN wrapper enables it explicitly.
    opts.structureScoreEnable = false;
    opts.structureValidationMultiplier = 4.0;
    % Final share of continuous validation evidence inside the hard rho pool.
    % The remaining weight is assigned to one robust two-sided frontier score.
    opts.structureValidationWeight = 0.20;
    % Validation errors below this scale-aware MSE floor are treated as tied;
    % the simpler expression is selected instead of chasing roundoff-level gain.
    opts.structureMachineErrorAbsMSEFloor = 0;
    opts.structureMachineErrorRelMSEFloor = 0;
    opts.structureNeighborhoodMaxDistance = 0.55;
    opts.structureNeighborhoodMinDistance = 0.10;
    opts.structureNeighborhoodComplexityWindow = 8;
    opts.structureFrontierMaxAbs = 20;

    % Display.
    opts.verbose = true;
end
