%% Demo: SINDy bypass -> native multi-output PySR -> augmented PhDN DAG -> refinement
% Stage 0: Test the fixed general-SINDy dictionary for a mechanical-precision
%          bypass. If rejected, send all unresolved outputs through one native
%          multi-output PySR fit per restart while retaining independent output
%          archives, losses, rankings, and final structure cores.
% Stage 1: Recursively compile these core trees into a shared compact PhDN DAG
%          and append a bounded SINDy/general-operator augmentation family.
% Stage 2: Run BP/LSQ refinement, limited densification, and one-step pruning.
%
% KAN-paper MLP, official EQL-Div, pruned-KAN, and SINDy baselines are retained
% below. The PhDN Stage-0 SR ablation is collected automatically from the
% already-completed Stage-0 search. An independent PySR-original-score ablation is also available.

clear; clc;
rng(1);

casemode = 'general';       % general | weak_prior_lv1 | weak_prior_lv2 | weak_prior_lv3 | strong_prior
caseToRun = 'I_9_18';
runAllCases = true;
% caseList = { ...
%     'I_9_18', ...
% };
caseList = { ...
    'I_6_2', 'I_6_2b', 'I_9_18', 'I_12_11', 'I_13_12', 'I_15_3x', ...
    'I_16_6', 'I_18_4', 'I_26_2', 'I_27_6', 'I_29_16', ...
    'I_30_3', 'I_30_5', 'I_37_4', 'I_40_1', 'I_44_4', 'I_50_26', ...
    'II_2_42', 'II_6_15a', 'II_11_7', 'II_11_27', 'II_35_18', ...
    'II_36_38', 'II_38_3', 'III_9_52', 'III_10_19', 'III_17_37'};

projectRoot = pwd;
if ~exist(fullfile(projectRoot, 'core'), 'dir') || ~exist(fullfile(projectRoot, 'tasks'), 'dir')
    try
        activeFile = matlab.desktop.editor.getActiveFilename;
        projectRoot = fileparts(activeFile);
    catch
        error('Cannot locate the project root. Please set MATLAB current folder to the framework root.');
    end
end
addpath(genpath(projectRoot));
rehash;

% -------------------------------------------------------------------------
% Feynman result persistence / replay
% -------------------------------------------------------------------------
OutputCaseRoot = fullfile(projectRoot,'outputs','Feynman_Dimless');
if ~exist(OutputCaseRoot,'dir'); mkdir(OutputCaseRoot); end
ResultsFile = fullfile(OutputCaseRoot,'feynman_dimless_results.mat');
SaveResults = true;
RecordedReportCompactMode = true;

% -------------------------------------------------------------------------
% Stage 0: fixed-SINDy bypass, then protected-output native multi-output PySR
% -------------------------------------------------------------------------
Stage0SingleLayerBypassEnable = true;
Stage0SingleLayerBypassThreshold = 1e-18;

% Fixed Phi_SINDy used only by the preliminary Stage-0 bypass.
Stage0BasePolyOrder = 2;
Stage0BaseUnaryOperators = {'inv','sqrt','exp','sin','cos','log'};
Stage0BaseIncludeUnaryOnMonomials = true;
Stage0BaseIncludeOperatorCrossTerms = false;
Stage0BaseIncludeSinCosPair = false;
Stage0BaseMaxLibraryTerms = Inf;
Stage0STLSQThresholdList = [0, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3];
Stage0STLSQMaxIter = 10;
Stage0RidgeLambda = 0;
Stage0WorstOutputWeight = 0.10;

% Official PySR searches all unresolved outputs in one native multi-output fit.
% Every output still owns an independent population archive and scalar loss.
Stage0PythonExe = 'C:\Users\hhy\miniconda3\envs\pysr_sr\python.exe';
Stage0PySRPaperRoot = fullfile(projectRoot, 'baselines', 'pysr_paper_main');
Stage0WorkRoot = fullfile(projectRoot, 'tmp', 'stage0_pysr_runs');
Stage0GrammarCasemode = casemode;
Stage0PopulationSize = 64;

% Independent standard PySR restarts inside each outer round. All exported
% candidates share the same validation set. Their existing predictions are
% reused to score one robust two-sided multi-scale frontier score Sfront; this
% structural ranking launches no extra PySR or Stage-1 evaluations.
Stage0InnerNumRestarts = 3;
Stage0InnerNIterations = 500;
Stage0InnerPopulations = 18; % fixed total logical-population budget across unresolved outputs
Stage0PopulationBudgetMode = 'fixed_total';
Stage0InnerRandomStateStride = 100;
Stage0MaxDepth = 10;
Stage0MaxSize = 30;
Stage0Parsimony = 1e-6;
Stage0ModelSelection = 'best';
Stage0RandomState = 1;
Stage0Deterministic = false;% false - multithreading; true - serial
Stage0Parallelism = 'multithreading'; % multithreading; serial
Stage0StrictDeterministicTestMode = false; % runs the identical Stage-0 search twice and asserts exact repeatability
Stage0RepeatabilityPredictionTolerance = 1e-12;
Stage0Verbosity = 1;
Stage0Progress = false;
Stage0TopKExpressionsToReport = 10;
Stage0DisplayCandidateRankings = true; % controls printing of all candidate ranking tables
Stage0CandidateRankingTopK = 10;       % maximum displayed rows per ranking only
Stage0SemanticDedupTolerance = 1e-8;
Stage0StructureScoreEnable = true;
Stage0StructureValidationMultiplier = 4.0;
Stage0StructureValidationWeight = 0.30; % soft-validation share; Sfront share is 1 minus this value
Stage0StructureFrontierMaxAbs = 20;
% Slocal and K are merged into Sfront.
Stage0StructureNeighborhoodMaxDistance = 0.55;
Stage0StructureNeighborhoodMinDistance = 0.10;
Stage0StructureNeighborhoodComplexityWindow = 8;
% Sfront bound is Stage0StructureFrontierMaxAbs.

% Outer repeated-run controls. Each round keeps the same MATLAB data split
% (rng(1)) but advances the PySR random state to sample an independent
% multithreaded evolutionary trajectory. Set the stride to 0 to reuse
% the same PySR random state in every round.
NumRounds = 3;
RoundRandomStateStride = 1;

% -------------------------------------------------------------------------
% Stage 1: SR-to-PhDN augmented-DAG initialization
% -------------------------------------------------------------------------
Stage1EnableAugmentation = true;
Stage1AugmentationPolyOrder = Stage0BasePolyOrder;       % uniform constant + total-degree Poly_2
Stage1AugmentationIncludeCrossTerms = true;
Stage1ForceStage0SeedOnly = true;
Stage1ExpandBoundsToIncludeStage0Seed = true;
Stage1Stage0SeedBoundMargin = 1e-6;
Stage1RequireExactStage0Reproduction = true;
Stage1Stage0ReproductionRelTolerance = 1e-6;
Stage1Stage0ReproductionAbsTolerance = 1e-10;

% -------------------------------------------------------------------------
% Stage 2: differentiable refinement and limited densification
% -------------------------------------------------------------------------
Stage2Enable = true;
FinalLSQMaxIter = 500;
FinalLSQMaxFunEvals = 5e4;
FinalLSQMaxRelValIncrease = 1e-10;
PostBPPruneEnable = true;
PostBPPruneNumIterations = 1;
PostBPPruneAbsThreshold = 1e-4;
PostBPPruneRelThreshold = 0;
PostBPPruneMaxRelValIncrease = 1e-4;

% -------------------------------------------------------------------------
% Data/report controls
% -------------------------------------------------------------------------
InitializationMode = 'skip';
NSamples = 2500;
EnableOOD = true;
NOODSamples = 500;
DisplayDictionary = true;
DictionaryMaxRowsToPrint = 80;
PrintFinalXiMatrices = false;
FinalXiPrintPrecision = 4;
FinalXiPrintOnlyActive = false;

% -------------------------------------------------------------------------
% Method switches
% -------------------------------------------------------------------------
% MLP, EQL, KAN, and SINDy can be enabled independently. If RunPhDNMainModel=false
% and any external baseline is enabled, one shared task-generated split is
% created and reused. Stage0-SR is recorded only when PhDN runs because it is
% an ablation of PhDN's existing Stage-0 search, not an independent baseline.
RunPhDNMainModel = true;
RunOriginalPySRScoreBaseline = true;  % independent PySR search; native/original score selects expression
RunMLPBaseline = true;
RunEQLBaseline = true;
RunKANBaseline = true;
RunSINDyBaseline = true;

% Replay controls. Run*=true always takes priority over the matching replay flag.
displayRecordReport_PhDN = false;
displayRecordReport_Stage0SR = false;
displayRecordReport_PySROriginalScore = false;
displayRecordReport_MLP = false;
displayRecordReport_EQL = false;
displayRecordReport_KAN = false;
displayRecordReport_SINDy = false;

% -------------------------------------------------------------------------
% KAN-paper Feynman MLP baseline controls
% -------------------------------------------------------------------------
% Paper structure: fixed width 20; affine-layer depth 2:6; Tanh/ReLU/SiLU.
% Within each outer round, all 15 candidates use the same round seed and the
% validation MSE selects one model. ID-test/OOD never participate in selection.
MLPPythonExe = Stage0PythonExe;
KANPythonExe = Stage0PythonExe;
PyKANRoot = fullfile(projectRoot, 'third_party', 'pykan');
MLPSeed = 1;
MLPProtocol = 'kan_feynman_sweep';
MLPWidth = 20;
MLPDepthList = 2:6;
MLPActivationList = {'tanh','relu','silu'};
MLPOptimizer = 'LBFGS';
MLPTrainSteps = 500;
MLPLearningRate = 1.0;
MLPWorkRoot = fullfile(projectRoot, 'tmp', 'kan_paper_mlp_runs');
MLPVerbose = true;
MLPDisplaySweepTable = true;

% -------------------------------------------------------------------------
% Official EQL-Div baseline controls
% -------------------------------------------------------------------------
% The network/loss/training/model-selection core is the unchanged Theano
% implementation bundled from martius-lab/EQL. MATLAB only supplies the raw
% shared data split and collects the official candidate scan.
EQLPythonExe = 'C:\Users\hhy\miniconda3\envs\eql_official\python.exe';
EQLOfficialRoot = fullfile(projectRoot, 'baselines', 'eql', 'official_eql');
EQLSeed = 1;
EQLDepthList = [2,3,4];              % paper depth L; upstream hidden layers=L-1
EQLLambdaList = [1e-5,1e-3];  % complete ICML-2018 grid
EQLUnitsPerUnaryType = 10;           % also 10 multiplication units upstream
EQLStepsPerHiddenLayer = 3000;
EQLBatchSize = 20;
EQLLearningRate = 1e-3;
EQLGradient = 'adam';
EQLLambdaL2 = 0;
EQLPenaltyEvery = 50;
EQLValidateEvery = 10;
EQLCandidateWorkers = 0;             % auto: min(candidate count, logical CPU count)
EQLNormalizeInputs = true;            % external training-box map to [-1,1]
EQLNormalizeOutputs = true;           % external training mean/std scaling
EQLTheanoFlags = 'device=cpu,floatX=float64,optimizer=fast_run,exception_verbosity=high';
EQLWorkRoot = fullfile(projectRoot, 'tmp', 'eql_official_runs');
EQLVerbose = true;
EQLDisplaySweepTable = true;

% -------------------------------------------------------------------------
% Official-pyKAN Feynman pruned-refinement baseline controls
% -------------------------------------------------------------------------
% Appendix-P protocol: initialize at G=3, train exactly once for 200 LBFGS
% steps at every paper grid with one fixed sparsification lambda, carry the
% model forward through the official pykan refine() operation, and call the
% official prune() once after G=200. No lamb=0 warm-up, post-prune recovery,
% validation-knee early stop, or other extra training stage is added.
KANSeed = 1;
KANWidth = 5;
KANDepthList = 2:6;
KANGridList = [3,5,10,20,50,100,200];
KANSplineOrder = 3;
KANSparsificationLambdaList = [1e-2,1e-3];
KANStepsPerGrid = 200;
KANOptimizer = 'LBFGS';
KANLearningRate = 1.0;
KANPruneNodeThreshold = 1e-2;
KANPruneEdgeThreshold = 3e-2;
KANWorkRoot = fullfile(projectRoot, 'tmp', 'kan_official_feynman_runs');
KANVerbose = true;
KANDisplaySweepTable = true;

% -------------------------------------------------------------------------
% SINDy baseline controls
% -------------------------------------------------------------------------
SINDyThresholdList = [0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3];
SINDyMaxSTLSQIter = 10;
SINDyRidgeLambda = 0;
SINDyDictionaryMode = 'general';       % independent broad flat dictionary used only as a baseline
SINDyPolyOrder = 2;
SINDyUnaryOperators = {'inv','sqrt','exp','sin','cos','log'};
SINDyIncludeUnaryOnMonomials = true;
SINDyIncludeOperatorCrossTerms = true;
SINDyUsePhdnDictionarySupport = false;
SINDyCenterScaleLibrary = false;
SINDyRemoveNearConstantRows = false;
SINDyVerbose = true;
SINDyMaxTermsToPrint = 30;

% Stage0-SR ablation reporting has no separate controls. It reuses the
% Stage-0 expressions, ID/OOD predictions, and timing already stored by PhDN.
% If every output is accepted by the Stage-0 SINDy bypass, that accepted
% single-layer SINDy model is still the final PhDN result; only Stage0-SR is N/A
% because no PySR search was executed.

if runAllCases
    selectedCases = unique(caseList, 'stable');
    if numel(selectedCases) < numel(caseList)
        warning('Duplicate case names were removed because NumRounds now controls repeated runs.');
    end
else
    selectedCases = {caseToRun};
end

allResults = struct();
summaryRows = struct([]);
iSummary = 0;

validateattributes(NumRounds, {'numeric'}, {'scalar','integer','positive','finite'}, mfilename, 'NumRounds');
validateattributes(RoundRandomStateStride, {'numeric'}, {'scalar','integer','finite'}, mfilename, 'RoundRandomStateStride');

for iRound = 1:NumRounds
    roundStage0RandomState = Stage0RandomState + (iRound - 1) * RoundRandomStateStride;
    roundMlpSeed = MLPSeed + (iRound - 1) * RoundRandomStateStride;
    roundEqlSeed = EQLSeed + (iRound - 1) * RoundRandomStateStride;
    roundKanSeed = KANSeed + (iRound - 1) * RoundRandomStateStride;
    fprintf('\n####################################################\n');
    fprintf('Outer complete-run round %d/%d | Stage-0 PySR state = %d | MLP/EQL/KAN seeds = %d/%d/%d\n', ...
        iRound, NumRounds, roundStage0RandomState, roundMlpSeed, roundEqlSeed, roundKanSeed);
    fprintf('MATLAB data split seed is reset to rng(1) for controlled comparison.\n');
    fprintf('####################################################\n');

    for iCase = 1:numel(selectedCases)
        caseName = selectedCases{iCase};
        rng(1);

        fprintf('\n====================================================\n');
        fprintf('Running KAN dimensionless Feynman case: %s | round %d/%d\n', caseName, iRound, NumRounds);
        fprintf('casemode = %s\n', casemode);
        fprintf('InitializationMode = %s\n', InitializationMode);
        fprintf('====================================================\n');

    task = task_kan_feynman_dimless(caseName, casemode);
    task.operatorMode = 'true';
    task.arch.operatorMode = 'true';
    task.training.operatorMode = 'true';
    task.DisplaySymbolic = strcmpi(casemode, 'strong_prior');

    opts = phdnn_default_options(task);
    opts.init.mode = lower(strtrim(InitializationMode));
    opts.data.nSamples = NSamples;
    opts.ood.enable = EnableOOD;
    opts.ood.nSamples = NOODSamples;
    opts.ood.autoMode = 'upper';
    opts.ood.autoGapRatio = 0.05;
    opts.ood.autoWidthRatio = 0.25;
    opts.ood.useTaskOodDomain = true;

    % Stage 0 controls.
    opts.stage0.enable = true;
    opts.stage0.method = 'sindy_bypass_then_native_multioutput_pysr';
    opts.stage0.singleLayerBypassEnable = Stage0SingleLayerBypassEnable;
    opts.stage0.singleLayerBypassThreshold = Stage0SingleLayerBypassThreshold;
    opts.stage0.baseDictionary.polyOrder = Stage0BasePolyOrder;
    opts.stage0.baseDictionary.unaryOperators = Stage0BaseUnaryOperators;
    opts.stage0.baseDictionary.includeUnaryOnMonomials = Stage0BaseIncludeUnaryOnMonomials;
    opts.stage0.baseDictionary.includeOperatorCrossTerms = Stage0BaseIncludeOperatorCrossTerms;
    opts.stage0.baseDictionary.includeSinCosPair = Stage0BaseIncludeSinCosPair;
    opts.stage0.baseDictionary.maxLibraryTerms = Stage0BaseMaxLibraryTerms;
    opts.stage0.fit.thresholdList = Stage0STLSQThresholdList;
    opts.stage0.fit.maxSTLSQIter = Stage0STLSQMaxIter;
    opts.stage0.fit.ridgeLambda = Stage0RidgeLambda;
    opts.stage0.worstOutputWeight = Stage0WorstOutputWeight;
    opts.stage0.pysr.pythonExe = Stage0PythonExe;
    opts.stage0.pysr.pysrPaperRoot = Stage0PySRPaperRoot;
    opts.stage0.pysr.workRoot = Stage0WorkRoot;
    opts.stage0.pysr.grammarCasemode = Stage0GrammarCasemode;
    opts.stage0.pysr.populationSize = Stage0PopulationSize;
    opts.stage0.pysr.innerNumRestarts = Stage0InnerNumRestarts;
    opts.stage0.pysr.innerNIterations = Stage0InnerNIterations;
    opts.stage0.pysr.innerPopulations = Stage0InnerPopulations;
    opts.stage0.pysr.multiOutputMode = 'native_single_fit_independent_output_archives';
    opts.stage0.pysr.populationBudgetMode = Stage0PopulationBudgetMode;
    opts.stage0.pysr.innerRandomStateStride = Stage0InnerRandomStateStride;
    opts.stage0.pysr.maxDepth = Stage0MaxDepth;
    opts.stage0.pysr.maxSize = Stage0MaxSize;
    opts.stage0.pysr.parsimony = Stage0Parsimony;
    % Binary/unary operators and operator-complexity overrides remain those
    % installed by phdnn_default_options(task), so each Feynman weak-prior
    % case keeps its task-defined compact grammar instead of being overwritten
    % by an old demo-level 'auto' placeholder.
    opts.stage0.pysr.unaryOperators = unique([reshape(opts.stage0.pysr.unaryOperators, 1, []), {'log'}], 'stable');
    opts.stage0.pysr.modelSelection = Stage0ModelSelection;
    opts.stage0.pysr.selectionPolicy = 'phdn_structure_score'; % proposed PhDN selector
    opts.stage0.pysr.randomState = roundStage0RandomState;
    opts.stage0.pysr.deterministic = Stage0Deterministic;
    opts.stage0.pysr.parallelism = Stage0Parallelism;
    opts.stage0.pysr.strictDeterministicTestMode = Stage0StrictDeterministicTestMode;
    opts.stage0.pysr.repeatabilityPredictionTolerance = Stage0RepeatabilityPredictionTolerance;
    opts.stage0.pysr.verbosity = Stage0Verbosity;
    opts.stage0.pysr.progress = Stage0Progress;
    opts.stage0.pysr.topKExpressionsToReport = Stage0TopKExpressionsToReport;
    opts.stage0.pysr.displayCandidateRankings = Stage0DisplayCandidateRankings;
    opts.stage0.pysr.candidateRankingTopK = Stage0CandidateRankingTopK;
    opts.stage0.pysr.semanticDedupTolerance = Stage0SemanticDedupTolerance;
    opts.stage0.pysr.structureScoreEnable = Stage0StructureScoreEnable;
    opts.stage0.pysr.structureValidationMultiplier = Stage0StructureValidationMultiplier;
    opts.stage0.pysr.structureValidationWeight = Stage0StructureValidationWeight;
    opts.stage0.pysr.structureFrontierMaxAbs = Stage0StructureFrontierMaxAbs;
    % Slocal/K weights removed in v73c.
    opts.stage0.pysr.structureNeighborhoodMaxDistance = Stage0StructureNeighborhoodMaxDistance;
    opts.stage0.pysr.structureNeighborhoodMinDistance = Stage0StructureNeighborhoodMinDistance;
    opts.stage0.pysr.structureNeighborhoodComplexityWindow = Stage0StructureNeighborhoodComplexityWindow;
    % Sfront maximum is passed above.
    opts.stage0.verbose = true;

    % Stage 1 controls.
    opts.stage1.method = 'sr_to_phdn_augmented_dag_initialization';
    opts.stage1.dictionaryMode = sprintf('sr_structural_dag_plus_uniform_poly%d_augmentation', Stage1AugmentationPolyOrder);
    opts.stage1.enableAugmentation = Stage1EnableAugmentation;
    opts.stage1.includeBestExpressionPath = true;
    opts.stage1.augmentationPolyOrder = Stage1AugmentationPolyOrder;
    opts.stage1.augmentationIncludeCrossTerms = Stage1AugmentationIncludeCrossTerms;
    opts.stage1.forceStage0SeedOnly = Stage1ForceStage0SeedOnly;
    opts.stage1.expandBoundsToIncludeStage0Seed = Stage1ExpandBoundsToIncludeStage0Seed;
    opts.stage1.stage0SeedBoundMargin = Stage1Stage0SeedBoundMargin;
    opts.stage1.requireExactStage0Reproduction = Stage1RequireExactStage0Reproduction;
    opts.stage1.stage0ReproductionRelTolerance = Stage1Stage0ReproductionRelTolerance;
    opts.stage1.stage0ReproductionAbsTolerance = Stage1Stage0ReproductionAbsTolerance;
    opts.stage1.verbose = true;

    % Stage 2 controls.
    opts.stage2.enable = Stage2Enable;
    opts.init.lsq.enable = Stage2Enable;
    opts.init.lsq.maxIter = FinalLSQMaxIter;
    opts.init.lsq.maxFunEvals = FinalLSQMaxFunEvals;
    opts.init.lsq.maxRelValIncrease = FinalLSQMaxRelValIncrease;
    opts.init.postBPPrune.enable = Stage2Enable && PostBPPruneEnable;
    opts.init.postBPPrune.numIterations = PostBPPruneNumIterations;
    opts.init.postBPPrune.absThreshold = PostBPPruneAbsThreshold;
    opts.init.postBPPrune.relThreshold = PostBPPruneRelThreshold;
    opts.init.postBPPrune.contributionAbsThreshold = PostBPPruneAbsThreshold;
    opts.init.postBPPrune.contributionRelThreshold = PostBPPruneRelThreshold;
    opts.init.postBPPrune.maxRelValIncrease = PostBPPruneMaxRelValIncrease;
    opts.init.postBPPrune.acceptByValidation = true;
    opts.init.postBPPrune.verbose = true;

    opts.output.skipSymbolicDisplay = ~strcmpi(casemode, 'strong_prior');
    opts.output.printFinalXiMatrices = PrintFinalXiMatrices;
    opts.output.finalXiPrintPrecision = FinalXiPrintPrecision;
    opts.output.finalXiPrintOnlyActive = PrintFinalXiMatrices && FinalXiPrintOnlyActive;

    if DisplayDictionary
        fprintf('\nSINDy-bypass / native multi-output PySR / augmented-PhDN route:\n');
        fprintf('Stage 0 restarts: count/iterations/total-population-budget/populationSize = %d/%d/%d/%d\n', ...
            opts.stage0.pysr.innerNumRestarts, ...
            opts.stage0.pysr.innerNIterations, opts.stage0.pysr.innerPopulations, ...
            opts.stage0.pysr.populationSize);
        fprintf('Outer round/random state = %d/%d; PySR random state = %d\n', ...
            iRound, NumRounds, opts.stage0.pysr.randomState);
        fprintf('Stage 0 PySR deterministic/parallelism/repeat-check = %d/%s/%d\n', ...
            opts.stage0.pysr.deterministic, opts.stage0.pysr.parallelism, opts.stage0.pysr.strictDeterministicTestMode);
        fprintf('Stage 0 candidate ranking report/top K = %d/%d (new structure score, validation MSE, complexity, and PySR score)\n', ...
            opts.stage0.pysr.displayCandidateRankings, opts.stage0.pysr.candidateRankingTopK);
        fprintf('Stage 0 PySR maxDepth/maxSize = %d/%d; core selection = <= %.1fx best validation MSE, soft-val weight %.2f + Sfront\n', ...
            opts.stage0.pysr.maxDepth, opts.stage0.pysr.maxSize, ...
            opts.stage0.pysr.structureValidationMultiplier, opts.stage0.pysr.structureValidationWeight);
        fprintf('Stage 0 SINDy-bypass library: polyOrder=%d, unary={%s}, STLSQ thresholds=%d\n', ...
            opts.stage0.baseDictionary.polyOrder, strjoin(opts.stage0.baseDictionary.unaryOperators, ','), ...
            numel(opts.stage0.fit.thresholdList));
        fprintf('Stage 1: selected structure-score core trees -> shared compact DAG + uniform constant+Poly_%d augmentation; coefficient hard mask = off\n', ...
            opts.stage1.augmentationPolyOrder);
        fprintf('Stage 2: BP/LSQ refinement = %d, post-BP pruning = %d, prune iterations = %d\n', ...
            opts.init.lsq.enable, opts.init.postBPPrune.enable, opts.init.postBPPrune.numIterations);
        fprintf('Route summary: Stage0(SINDy bypass; otherwise native multi-output PySR with protected output archives) -> Stage1(SR structural DAG + uniform Poly_%d augmentation) -> Stage2(BP/LSQ refinement and pruning)\n', ...
            opts.stage1.augmentationPolyOrder);
        fprintf('RunPhDNMainModel                     = %d\n', RunPhDNMainModel);
        fprintf('RunMLPBaseline / RunEQLBaseline / RunKANBaseline / RunSINDyBaseline = %d / %d / %d / %d\n', ...
            RunMLPBaseline, RunEQLBaseline, RunKANBaseline, RunSINDyBaseline);
        if RunPhDNMainModel
            fprintf('Stage0-SR ablation recording  = automatic (collected from PhDN Stage 0; no extra run)\n');
        end
        if RunMLPBaseline
            fprintf('MLP sweep: width=%d, depths={%s}, activations={%s}, optimizer=%s, steps=%d, seed=%d\n', ...
                MLPWidth, num2str(MLPDepthList, '%d '), strjoin(MLPActivationList, ','), ...
                MLPOptimizer, MLPTrainSteps, roundMlpSeed);
        end
        if RunEQLBaseline
            if EQLCandidateWorkers <= 0
                eqlWorkerLabel = 'auto';
            else
                eqlWorkerLabel = sprintf('%d', EQLCandidateWorkers);
            end
            fprintf('Official EQL-Div sweep: L={%s}, lambda count/range=%d/[%.1e,%.1e], units/type=%d, epochs=(L-1)*%d, optimizer=%s, lr=%.1e, workers=%s, seed=%d\n', ...
                num2str(EQLDepthList, '%d '), numel(EQLLambdaList), min(EQLLambdaList), max(EQLLambdaList), ...
                EQLUnitsPerUnaryType, EQLStepsPerHiddenLayer, EQLGradient, ...
                EQLLearningRate, eqlWorkerLabel, roundEqlSeed);
            fprintf('Official EQL external scaling        = input:%d / output:%d\n', EQLNormalizeInputs, EQLNormalizeOutputs);
            fprintf('Official EQL source root             = %s\n', EQLOfficialRoot);
        end
        if RunKANBaseline
            fprintf('Official pyKAN Feynman sweep: width=%d, depths={%s}, grids={%s}, lambdas={%s}, steps/grid=%d, initial-sparse-fit-and-prune=1, later-lambda=0, seed=%d\n', ...
                KANWidth, num2str(KANDepthList, '%d '), num2str(KANGridList, '%d '), ...
                num2str(KANSparsificationLambdaList, '%.0e '), KANStepsPerGrid, roundKanSeed);
        end
        if RunSINDyBaseline
            fprintf('SINDy baseline controls: mode=%s, polyOrder=%d, unaryOps=%d, thresholds=%d, maxSTLSQIter=%d, ridge=%.1e\n', ...
                SINDyDictionaryMode, SINDyPolyOrder, numel(SINDyUnaryOperators), ...
                numel(SINDyThresholdList), SINDyMaxSTLSQIter, SINDyRidgeLambda);
        end
    end

    runAnyBaseline = RunOriginalPySRScoreBaseline || RunMLPBaseline || RunEQLBaseline || RunKANBaseline || RunSINDyBaseline;
    runAnyMethod = RunPhDNMainModel || runAnyBaseline;
    replayAnyMethod = displayRecordReport_PhDN || displayRecordReport_Stage0SR || ...
        displayRecordReport_PySROriginalScore || displayRecordReport_MLP || ...
        displayRecordReport_EQL || displayRecordReport_KAN || displayRecordReport_SINDy;
    result = [];
    resultStage0SR = [];
    baselineDataResult = [];
    resultPack = struct();
    context = struct('caseName',task.name,'round',iRound,'casemode',casemode, ...
        'stage0RandomState',roundStage0RandomState,'matlabDataSeed',1, ...
        'selectionPolicyProposed','phdn_structure_score', ...
        'selectionPolicyOriginalPySR','pysr_native_score');

    % ---------------- PhDN + collected proposed Stage-0 SR ----------------
    if RunPhDNMainModel
        result = phdnn_identify(task, opts);
        print_demo_output(task, result);
        if isfield(result,'ablations') && isstruct(result.ablations) && isfield(result.ablations,'stage0SR')
            resultStage0SR = result.ablations.stage0SR;
        else
            resultStage0SR = make_stage0_sr_ablation_result(result);
            result.ablations.stage0SR = resultStage0SR;
        end
        print_stage0_sr_ablation_result(resultStage0SR);
        resultPack.phdn = result;
        resultPack.stage0sr = resultStage0SR;
        baselineDataResult = result;
        if SaveResults
            save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'phdn','PhDN',result,context);
            save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'stage0sr','Stage0-SR (proposed score)',resultStage0SR,context);
        end
    else
        if displayRecordReport_PhDN
            [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'phdn','PhDN',RecordedReportCompactMode);
            if ok; resultPack.phdn=r; end
        end
        if displayRecordReport_Stage0SR
            [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'stage0sr','Stage0-SR (proposed score)',RecordedReportCompactMode);
            if ok; resultPack.stage0sr=r; end
        end
    end

    % Generate exactly one shared split only when a currently-run independent
    % baseline needs it and PhDN did not already provide that split.
    if runAnyBaseline && isempty(baselineDataResult)
        baselineDataResult = make_baseline_data_result_from_task(task, opts);
        fprintf('PhDN main model skipped; generated one shared data split for enabled baselines.\n');
    end

    % ---------------- Original/native PySR-score ablation ----------------
    if RunOriginalPySRScoreBaseline
        nativeOpts = opts;
        nativeOpts.stage0.pysr.selectionPolicy = 'pysr_native_score';
        nativeOpts.stage0.pysr.structureScoreEnable = false;
        nativeOpts.stage0.pysr.workRoot = fullfile(Stage0WorkRoot,'original_pysr_score');
        resultPySROriginal = run_feynman_original_pysr_score_baseline(task,nativeOpts,baselineDataResult);
        resultPack.pysrOriginalScore = resultPySROriginal;
        if SaveResults
            save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'pysrOriginalScore','PySR-original-score',resultPySROriginal,context);
        end
    elseif displayRecordReport_PySROriginalScore
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'pysrOriginalScore','PySR-original-score',RecordedReportCompactMode);
        if ok; resultPack.pysrOriginalScore=r; end
    end

    % ---------------- MLP ----------------
    if RunMLPBaseline
        mlpOpts = make_default_mlp_options_for_demo(roundMlpSeed);
        mlpOpts.protocol = MLPProtocol; mlpOpts.pythonExe = MLPPythonExe; mlpOpts.pykanRoot = PyKANRoot;
        mlpOpts.workRoot = MLPWorkRoot; mlpOpts.width = MLPWidth; mlpOpts.depthList = MLPDepthList;
        mlpOpts.activationList = MLPActivationList; mlpOpts.optimizer = MLPOptimizer; mlpOpts.steps = MLPTrainSteps;
        mlpOpts.learningRate = MLPLearningRate; mlpOpts.verbose = MLPVerbose; mlpOpts.displaySweepTable = MLPDisplaySweepTable;
        resultMlp = run_mlp_baseline_from_phdn_result(baselineDataResult, mlpOpts); resultPack.mlp = resultMlp;
        if RunPhDNMainModel; result.baselines.mlp = resultMlp; end
        if SaveResults; save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'mlp','MLP',resultMlp,context); end
    elseif displayRecordReport_MLP
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'mlp','MLP',RecordedReportCompactMode); if ok; resultPack.mlp=r; end
    end

    % ---------------- EQL-Div ----------------
    if RunEQLBaseline
        eqlOpts = make_default_eql_options_for_demo(roundEqlSeed);
        eqlOpts.pythonExe=EQLPythonExe; eqlOpts.officialRoot=EQLOfficialRoot; eqlOpts.workRoot=EQLWorkRoot;
        eqlOpts.depthList=EQLDepthList; eqlOpts.lambdaList=EQLLambdaList; eqlOpts.unitsPerUnaryType=EQLUnitsPerUnaryType;
        eqlOpts.multiplicationUnits=EQLUnitsPerUnaryType; eqlOpts.stepsPerHiddenLayer=EQLStepsPerHiddenLayer;
        eqlOpts.batchSize=EQLBatchSize; eqlOpts.learningRate=EQLLearningRate; eqlOpts.gradient=EQLGradient;
        eqlOpts.lambdaL2=EQLLambdaL2; eqlOpts.penaltyEvery=EQLPenaltyEvery; eqlOpts.validateEvery=EQLValidateEvery;
        eqlOpts.candidateWorkers=EQLCandidateWorkers; eqlOpts.normalizeInputs=EQLNormalizeInputs; eqlOpts.normalizeOutputs=EQLNormalizeOutputs;
        eqlOpts.theanoFlags=EQLTheanoFlags; eqlOpts.verbose=EQLVerbose; eqlOpts.displaySweepTable=EQLDisplaySweepTable;
        resultEql=run_eql_baseline_from_phdn_result(baselineDataResult,eqlOpts); resultPack.eql=resultEql;
        if RunPhDNMainModel; result.baselines.eql=resultEql; end
        if SaveResults; save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'eql','EQL-Div',resultEql,context); end
    elseif displayRecordReport_EQL
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'eql','EQL-Div',RecordedReportCompactMode); if ok; resultPack.eql=r; end
    end

    % ---------------- KAN ----------------
    if RunKANBaseline
        kanOpts=make_default_kan_options_for_demo(roundKanSeed);
        kanOpts.pythonExe=KANPythonExe; kanOpts.pykanRoot=PyKANRoot; kanOpts.workRoot=KANWorkRoot;
        kanOpts.width=KANWidth; kanOpts.depthList=KANDepthList; kanOpts.gridList=KANGridList; kanOpts.splineOrder=KANSplineOrder;
        kanOpts.sparsificationLambdaList=KANSparsificationLambdaList; kanOpts.stepsPerGrid=KANStepsPerGrid;
        kanOpts.optimizer=KANOptimizer; kanOpts.learningRate=KANLearningRate; kanOpts.pruneNodeThreshold=KANPruneNodeThreshold;
        kanOpts.pruneEdgeThreshold=KANPruneEdgeThreshold; kanOpts.verbose=KANVerbose; kanOpts.displaySweepTable=KANDisplaySweepTable;
        resultKan=run_kan_baseline_from_phdn_result(baselineDataResult,kanOpts); resultPack.kan=resultKan;
        if RunPhDNMainModel; result.baselines.kan=resultKan; end
        if SaveResults; save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'kan','KAN-pruned',resultKan,context); end
    elseif displayRecordReport_KAN
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'kan','KAN-pruned',RecordedReportCompactMode); if ok; resultPack.kan=r; end
    end

    % ---------------- SINDy ----------------
    if RunSINDyBaseline
        sindyOpts=make_default_sindy_options_for_demo(); sindyOpts.thresholdList=SINDyThresholdList;
        sindyOpts.maxSTLSQIter=SINDyMaxSTLSQIter; sindyOpts.ridgeLambda=SINDyRidgeLambda; sindyOpts.dictionaryMode=SINDyDictionaryMode;
        sindyOpts.polyOrder=SINDyPolyOrder; sindyOpts.unaryOperators=SINDyUnaryOperators; sindyOpts.includeUnaryOnMonomials=SINDyIncludeUnaryOnMonomials;
        sindyOpts.includeOperatorCrossTerms=SINDyIncludeOperatorCrossTerms; sindyOpts.usePhdnDictionarySupport=SINDyUsePhdnDictionarySupport;
        sindyOpts.centerScaleLibrary=SINDyCenterScaleLibrary; sindyOpts.removeNearConstantRows=SINDyRemoveNearConstantRows;
        sindyOpts.verbose=SINDyVerbose; sindyOpts.maxTermsToPrint=SINDyMaxTermsToPrint;
        resultSindy=run_sindy_baseline_from_phdn_result(baselineDataResult,task,sindyOpts,opts); resultPack.sindy=resultSindy;
        if RunPhDNMainModel; result.baselines.sindy=resultSindy; end
        if SaveResults; save_feynman_dimless_method_result(ResultsFile,iRound,task.name,'sindy','SINDy',resultSindy,context); end
    elseif displayRecordReport_SINDy
        [r,ok]=load_feynman_dimless_method_result(ResultsFile,iRound,task.name,'sindy','SINDy',RecordedReportCompactMode); if ok; resultPack.sindy=r; end
    end

    if ~runAnyMethod && ~replayAnyMethod
        fprintf('No run or replay method is enabled; skip this case.\n');
        continue;
    end

    % Refresh the stored PhDN object after attaching baseline and ablation records.
    if RunPhDNMainModel
        resultPack.phdn = result;
    end

        roundKey = sprintf('round_%02d', iRound);
        caseKey = matlab.lang.makeValidName(strrep(task.name, '.', 'p'));
        if ~isfield(allResults, roundKey)
            allResults.(roundKey) = struct();
        end
        allResults.(roundKey).(caseKey) = resultPack;

        if isfield(resultPack, 'phdn')
            [summaryRows, iSummary] = append_method_summary_row(summaryRows, iSummary, task.name, 'PhDN', resultPack.phdn, iRound);
        end
        if isfield(resultPack, 'stage0sr')
            [summaryRows, iSummary] = append_method_summary_row(summaryRows, iSummary, task.name, 'Stage0-SR', resultPack.stage0sr, iRound);
        end
        if isfield(resultPack, 'pysrOriginalScore')
            [summaryRows, iSummary] = append_method_summary_row(summaryRows, iSummary, task.name, 'PySR-original-score', resultPack.pysrOriginalScore, iRound);
        end
        if isfield(resultPack, 'mlp')
            [summaryRows, iSummary] = append_method_summary_row(summaryRows, iSummary, task.name, 'MLP', resultPack.mlp, iRound);
        end
        if isfield(resultPack, 'eql')
            [summaryRows, iSummary] = append_method_summary_row(summaryRows, iSummary, task.name, 'EQL-Div', resultPack.eql, iRound);
        end
        if isfield(resultPack, 'kan')
            [summaryRows, iSummary] = append_method_summary_row(summaryRows, iSummary, task.name, 'KAN-pruned', resultPack.kan, iRound);
        end
        if isfield(resultPack, 'sindy')
            [summaryRows, iSummary] = append_method_summary_row(summaryRows, iSummary, task.name, 'SINDy', resultPack.sindy, iRound);
        end
    end
end

% Preserve the per-run summary table, then aggregate all completed rounds.
% Stage0-SR rows are derived from PhDN Stage 0 and therefore add no training cost.
print_method_comparison_summary_local(summaryRows);
print_round_statistics_summary(summaryRows);

if SaveResults
    meta = struct('NumRounds',NumRounds,'selectedCases',{selectedCases},'casemode',casemode, ...
        'methodPersistenceSchema','independent_method_record_v1', ...
        'policy','per-method immediate checkpoint + non-destructive aggregate merge');
    save_feynman_dimless_summary_outputs(ResultsFile,allResults,summaryRows,meta);
end









