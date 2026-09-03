%% Demo: Soft-saturated Lorenz--96 system identification
% Second system-identification case for the PhDN manuscript.
% PhDN-G1/G2/G3 are independent prior-level methods in the same demo.
%
% The default K=10 cyclic system uses the demo parameters F and kappa:
%   z_i=x_{i-1}(x_{i+1}-x_{i-2}),
%   xdot_i=z_i/sqrt(1+(z_i/kappa)^2)-x_i+F.
% Stage 0 discovers one symbolic core per output. Stage 1 compiles the cores
% into a shared PhDN DAG and augments every active branch with K fixed
% tanh neural-ridge bases. Stage 2 jointly refines the structural and neural
% outer coefficients and then prunes weak contributions.
%
% The data sweep, baselines, persistence, rollout evaluation, paper figures,
% and output hierarchy clone run_demo_single_generator_dynamic_stage012.
% A Neural-SINDy comparator is also included. It replaces standalone
% degree-two polynomial block by the same K fixed raw-input neural-ridge
% bases used by the PhDN input branch; both baselines keep 1, x_i, and sqrt(x_i).

clear; clc;
rng(1);

casemode = 'general';
LorenzDimension = 10;
SaturationKappa = 1;
LorenzForcing = 8;
% Three independent Stage-0 prior levels, matching the original PhDN-G1--G3
% experiment structure. They share the same true system and data split:
%   G1 (theory level 0): no initial guesses.
%   G2 (theory level 2): ideal unsaturated cyclic transport+damping;
%                         the forcing F and smooth saturation are unknown.
%   G3 (theory level 3): the same unsaturated topology plus the true forcing F;
%                         only the smooth saturation is unknown.
% No training-sample coefficient calibration is used for the Stage-0 guesses.
caseToRun = sprintf('SS_L96_K%d',LorenzDimension);
TrainingSampleList = [250,500,1000,2000];

assert(all(diff(TrainingSampleList)>0), ...
    'TrainingSampleList must be strictly increasing for nested sample-model comparisons.');
NValidationSamples = 500;
NIDTestSamples = 1000;
NumRounds = 1;
RoundRandomStateStride = 1;

% Every Ntrain uses a prefix of one scrambled-Sobol master pool. Validation,
% ID-test, and OOD pools use independent fixed scrambles within each round.
SamplingMethod = 'scrambled_sobol';
SobolScrambleMethod = 'MatousekAffineOwen';
SobolSkip = 1024;
MaxTrainingSamples = max(TrainingSampleList);
BaseTrainPoolSeed = 7301;
BaseValidationPoolSeed = 7401;
BaseTestPoolSeed = 7501;
BaseOODPoolSeed = 8501;

projectRoot = pwd;
if ~exist(fullfile(projectRoot,'core'),'dir') || ~exist(fullfile(projectRoot,'tasks'),'dir')
    try
        activeFile = matlab.desktop.editor.getActiveFilename;
        projectRoot = fileparts(activeFile);
    catch
        error('Cannot locate the project root. Set the current folder to the framework root.');
    end
end
addpath(genpath(projectRoot));
addpath(fullfile(projectRoot,'system_identification'),'-begin');
addpath(fullfile(projectRoot,'baselines','eql'),'-begin');
clear plot_soft_saturated_lorenz96_trajectory ...
    plot_system_identification_sample_efficiency;
rehash;

% Build the parameterized task once so the output hierarchy, true RHS,
% reference model, data domains, and metadata all use the same F and kappa.
caseDefinitionTask = task_soft_saturated_lorenz96( ...
    caseToRun,casemode,LorenzForcing,SaturationKappa);
assert(caseDefinitionTask.parameters.F == LorenzForcing, ...
    'The task forcing does not match LorenzForcing.');
assert(caseDefinitionTask.parameters.kappa == SaturationKappa, ...
    'The task saturation parameter does not match SaturationKappa.');

% One physical-case output tree, matching the previous PhDN-G1--G3 layout.
% Each G level is persisted as an independent method under method_results/.
OutputCaseRoot = fullfile(projectRoot,'outputs',caseDefinitionTask.name);
OutputSummaryDir = fullfile(OutputCaseRoot,'summary');
if exist(OutputSummaryDir,'dir') ~= 7; mkdir(OutputSummaryDir); end

% Plot-only fast path: load the slim public summary and regenerate paper figures.
RegeneratePaperFiguresOnly = false;
PaperFigureSourceResultsFile = fullfile(OutputSummaryDir,'public_summary.mat');
LegacyPaperFigureSourceResultsFile = fullfile(OutputSummaryDir,'soft_saturated_lorenz96_results.mat');
if exist(PaperFigureSourceResultsFile,'file') ~= 2 && ...
        exist(LegacyPaperFigureSourceResultsFile,'file') == 2
    migrate_system_identification_summary_to_public( ...
        LegacyPaperFigureSourceResultsFile,PaperFigureSourceResultsFile, ...
        'DeleteLegacy',true);
end
PaperFigureExportDir = OutputSummaryDir;
PaperFigureVisible = true;
PaperFigureExportPDF = true;

if RegeneratePaperFiguresOnly
    paperFigureInfo = regenerate_soft_saturated_lorenz96_paper_figures( ...
        PaperFigureSourceResultsFile,PaperFigureExportDir, ...
        'Visible',PaperFigureVisible,'ExportPDF',PaperFigureExportPDF);
    fprintf('\nSoftSaturatedLorenz96 plot-only regeneration finished.\n');
    fprintf('Source summary MAT: %s\n',paperFigureInfo.sourceResultsFile);
    fprintf('Export directory   : %s\n',paperFigureInfo.exportDir);
    for iPaperFigure = 1:numel(paperFigureInfo.figures)
        fprintf('Generated figure %d: %s\n',iPaperFigure, ...
            paperFigureInfo.figures(iPaperFigure).pdfPath);
    end
    return;
end

CaseSimulationWallTimer = tic;
verify_soft_saturated_lorenz96_case_setup(caseDefinitionTask);

%% Stage 0: trivial-output bypass plus native multi-output PySR
% A generic flat SINDy dictionary is accepted only near machine precision.
% All unresolved outputs are searched by PySR with the fine-grained operators
% needed to assemble the soft-saturation law, but without the completed target
% composite feature.
Stage0SingleLayerBypassEnable = true;
Stage0SingleLayerBypassThreshold = 1e-24;

Stage0BasePolyOrder = 2;
Stage0BaseUnaryOperators = {'sqrt'};
Stage0BaseIncludeUnaryOnMonomials = false;
Stage0BaseIncludeOperatorCrossTerms = false;
Stage0BaseIncludeSinCosPair = false;
Stage0BaseMaxLibraryTerms = Inf;
Stage0STLSQThresholdList = [0,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3];
Stage0STLSQMaxIter = 10;
Stage0RidgeLambda = 0;
Stage0WorstOutputWeight = 0.10;

Stage0PythonExe = 'C:\Users\hhy\miniconda3\envs\pysr_sr\python.exe';
Stage0PySRPaperRoot = fullfile(projectRoot,'baselines','pysr_paper_main');
Stage0WorkRoot = fullfile(projectRoot,'tmp', ...
    'stage0_pysr_soft_saturated_l96_prior_ablation_runs');
Stage0GrammarCasemode = casemode;

Stage0UseCustomSRGrammar = true;
Stage0SRBinaryOperators = {'+','-','*','/'};
Stage0SRUnaryOperators = {'square','sqrt'};
Stage0TypedPhysicalPriorEnable = false;
Stage0TypedPhysicalPriorMode = '';
Stage0SRTrigAllowedVariables = {};
Stage0SRForbidStateDependentDivision = false;
Stage0SROperatorComplexities = struct();
Stage0SRStrictTrigAtomsOnly = false;
Stage0SRForbidNestedTrig = true;
Stage0SRForbidNestedSquare = true;
Stage0SRForbidNestedSqrt = true;

% Optional Stage-0 SR initial-guess port. As in the single-generator case,
% the manuscript demo keeps the actual guess expressions explicit and manual.
% For this Lorenz--96 case, the original uncalibrated structural priors are used:
% no training-sample LS coefficient fitting is applied to the guess expressions.
%
% Prior hierarchy:
%   G1: no Stage-0 initial guess.
%   G2: ideal unsaturated cyclic transport+damping,
%         x_{i-1}(x_{i+1}-x_{i-2}) - x_i;
%       forcing and smooth saturation are unknown.
%   G3: the same ideal unsaturated topology plus the prior-known forcing F=8;
%       only smooth saturation is unknown.
%
% The official PySR multi-output port remains a shared guess library across all
% unresolved outputs. The ten cyclic trees therefore compete under the normal
% data loss; no tree is hard-assigned to one output.
Stage0SRMinimumPySRVersion = '2.0.0a2';
Stage0SRRequirePySR2 = true;
Stage0SRInitialGuessFraction = 0.10;

assert(LorenzDimension==10 && abs(LorenzForcing-8)<1e-12 && ...
    abs(SaturationKappa-1)<1e-12, ...
    ['The manually entered Lorenz--96 prior block below is written for ', ...
     'K=10, F=8, kappa=1. Update the manual cyclic expressions if these ', ...
     'case parameters change.']);

% G1: no initial guess.
Stage0SRInitialGuesses_G1 = {};

% G2: ideal unsaturated cyclic transport+damping; forcing is not supplied.
Stage0SRInitialGuesses_G2 = { ...
    'x10*(x2-x9)-x1', ...
    'x1*(x3-x10)-x2', ...
    'x2*(x4-x1)-x3', ...
    'x3*(x5-x2)-x4', ...
    'x4*(x6-x3)-x5', ...
    'x5*(x7-x4)-x6', ...
    'x6*(x8-x5)-x7', ...
    'x7*(x9-x6)-x8', ...
    'x8*(x10-x7)-x9', ...
    'x9*(x1-x8)-x10'};

% G3: same unsaturated topology with the prior-known forcing F=8.
Stage0SRInitialGuesses_G3 = { ...
    'x10*(x2-x9)-x1+8', ...
    'x1*(x3-x10)-x2+8', ...
    'x2*(x4-x1)-x3+8', ...
    'x3*(x5-x2)-x4+8', ...
    'x4*(x6-x3)-x5+8', ...
    'x5*(x7-x4)-x6+8', ...
    'x6*(x8-x5)-x7+8', ...
    'x7*(x9-x6)-x8+8', ...
    'x8*(x10-x7)-x9+8', ...
    'x9*(x1-x8)-x10+8'};

% Keep the metadata helper for the common prior-level fields, but explicitly
% override the saved metadata so it matches the manual raw expressions above.
[~,Stage0PriorInfo_G1] = make_lorenz96_unsaturated_initial_guesses( ...
    LorenzDimension,LorenzForcing,'G1');
[~,Stage0PriorInfo_G2] = make_lorenz96_unsaturated_initial_guesses( ...
    LorenzDimension,LorenzForcing,'G2');
[~,Stage0PriorInfo_G3] = make_lorenz96_unsaturated_initial_guesses( ...
    LorenzDimension,LorenzForcing,'G3');

Stage0PriorInfo_G2.name = 'unsaturated_transport_and_damping_unknown_forcing';
Stage0PriorInfo_G2.description = [ ...
    'Ideal unsaturated cyclic transport and -x_i damping; the forcing ', ...
    'and smooth saturation are unknown. No guess-coefficient fitting is used.'];
Stage0PriorInfo_G2.includesTransport = true;
Stage0PriorInfo_G2.includesDamping = true;
Stage0PriorInfo_G2.includesForcing = false;
Stage0PriorInfo_G2.priorForcing = NaN;
Stage0PriorInfo_G2.coefficientCalibration = 'none';
Stage0PriorInfo_G2.outerCoefficientSource = 'manual_theory_unit_coefficients';
Stage0PriorInfo_G2.manualExpressions = Stage0SRInitialGuesses_G2;

Stage0PriorInfo_G3.name = 'full_unsaturated_lorenz96';
Stage0PriorInfo_G3.description = [ ...
    'Complete ideal unsaturated Lorenz--96 law with the prior-known true ', ...
    'forcing F; only smooth saturation is unknown. No guess-coefficient fitting is used.'];
Stage0PriorInfo_G3.includesTransport = true;
Stage0PriorInfo_G3.includesDamping = true;
Stage0PriorInfo_G3.includesForcing = true;
Stage0PriorInfo_G3.priorForcing = LorenzForcing;
Stage0PriorInfo_G3.coefficientCalibration = 'none';
Stage0PriorInfo_G3.forcingInitialization = 'prior-known F=8';
Stage0PriorInfo_G3.outerCoefficientSource = 'manual_theory_unit_coefficients_plus_known_forcing';
Stage0PriorInfo_G3.manualExpressions = Stage0SRInitialGuesses_G3;

Stage0SRInitialGuessFraction_G1 = 0;
Stage0SRInitialGuessFraction_G2 = Stage0SRInitialGuessFraction;
Stage0SRInitialGuessFraction_G3 = Stage0SRInitialGuessFraction;
assert(isempty(Stage0SRInitialGuesses_G1), ...
    'G1 must contain no Stage-0 initial guesses.');
assert(numel(Stage0SRInitialGuesses_G2)==LorenzDimension && ...
    numel(Stage0SRInitialGuesses_G3)==LorenzDimension, ...
    'G2 and G3 must each contain ten manually entered cyclic expressions.');

Stage0PriorAblation = struct();
Stage0PriorAblation.G1 = Stage0PriorInfo_G1;
Stage0PriorAblation.G2 = Stage0PriorInfo_G2;
Stage0PriorAblation.G3 = Stage0PriorInfo_G3;
Stage0PriorAblation.G1.initialGuesses = Stage0SRInitialGuesses_G1;
Stage0PriorAblation.G2.initialGuesses = Stage0SRInitialGuesses_G2;
Stage0PriorAblation.G3.initialGuesses = Stage0SRInitialGuesses_G3;
Stage0PriorAblation.G1.initialGuessFraction = Stage0SRInitialGuessFraction_G1;
Stage0PriorAblation.G2.initialGuessFraction = Stage0SRInitialGuessFraction_G2;
Stage0PriorAblation.G3.initialGuessFraction = Stage0SRInitialGuessFraction_G3;
caseDefinitionTask.prior.stage0PriorAblation = Stage0PriorAblation;

Stage0PopulationSize = 100;
Stage0InnerNumRestarts = 3;
Stage0InnerNIterations = 700;
Stage0InnerPopulations = 3*LorenzDimension; % preserve three protected populations per output
Stage0PopulationBudgetMode = 'fixed_total';
Stage0InnerRandomStateStride = 100;

% General output-self-referenced Stage-0 targeted rescue.
% Trigger A: best normalized validation MSE remains above the soft threshold
% AND the ordinary restarts are unstable for that same output.
% Trigger B: the ordinary restarts are relatively consistent, but even their
% best normalized validation MSE remains above the hard threshold.
% A/B both add one official single-output PySR restart. A second rescue restart
% is allowed only if the first reduces the running best q by at least 50% and
% the soft threshold is still not reached. Known derivative-label noise raises
% the soft/hard thresholds above 4x the expected validation-noise MSE floor.
Stage0AdaptiveRescueEnable = true;
Stage0AdaptiveRescueSoftNormalizedMSE = 1e-6;
Stage0AdaptiveRescueHardNormalizedMSE = 1e-3;
Stage0AdaptiveRescueInstabilityFactor = 5;
Stage0AdaptiveRescueContinueImprovementRatio = 0.5;
Stage0AdaptiveRescueUseKnownNoiseFloor = true;
Stage0AdaptiveRescueNoiseFloorMultiplier = 4;
Stage0AdaptiveRescueMaxRestartsPerOutput = 2;
Stage0AdaptiveRescuePopulationMultiplier = 1.5;
Stage0AdaptiveRescueMinPopulations = 8;
Stage0AdaptiveRescueMaxPopulations = 20;
Stage0AdaptiveRescueExplicitPopulations = []; % [] -> adaptive population rule
Stage0AdaptiveRescueNIterations = Stage0InnerNIterations;
Stage0AdaptiveRescueMaxOutputs = 3;
Stage0AdaptiveRescueUseOutputSpecificInitialGuess = true;
Stage0AdaptiveRescueSeedOffset = 10000;
Stage0AdaptiveRescueOutputSeedStride = 1000;
Stage0AdaptiveRescueRestartSeedStride = 100;
Stage0TournamentSelectionN = 8;
Stage0TournamentSelectionP = 0.85;
Stage0CrossoverProbability = [];
Stage0WeightAddNode = [];
Stage0WeightInsertNode = [];
Stage0WeightDeleteNode = [];
Stage0WeightDoNothing = [];
Stage0WeightMutateConstant = [];
Stage0WeightMutateOperator = [];
Stage0WeightMutateFeature = [];
Stage0WeightSwapOperands = [];
Stage0WeightRotateTree = [];
Stage0WeightRandomize = [];
Stage0WeightSimplify = [];
Stage0WeightOptimize = [];
Stage0OptimizeProbability = [];
Stage0ShouldSimplify = [];
Stage0MaxDepth = 12;
Stage0MaxSize = 40;
Stage0Parsimony = 1e-6;
Stage0ModelSelection = 'best';
Stage0RandomState = 1;
Stage0Deterministic = false;
Stage0Parallelism = 'multithreading';
Stage0Batching = false;
Stage0BatchSizeList = TrainingSampleList;
assert(numel(Stage0BatchSizeList)==numel(TrainingSampleList), ...
    'Stage0BatchSizeList must match TrainingSampleList.');
assert(all(Stage0BatchSizeList>=1 & Stage0BatchSizeList<=TrainingSampleList), ...
    'Every Stage-0 batch size must be positive and no larger than Ntrain.');
Stage0StrictDeterministicTestMode = false;
Stage0RepeatabilityPredictionTolerance = 1e-12;
Stage0MachinePrecisionEarlyStopEnable = true;
Stage0MachinePrecisionEarlyStopAbsMSE = 1e-14;
Stage0MachinePrecisionEarlyStopRelMSE = 1e-14;
Stage0MachinePrecisionEarlyStopCheckInterval = 50;
Stage0MachinePrecisionEarlyStopMinIterations = 50;
Stage0MachinePrecisionEarlyStopAcrossRestarts = true;
Stage0Verbosity = 1;
Stage0Progress = false;
Stage0TopKExpressionsToReport = 10;
Stage0DisplayCandidateRankings = true;
Stage0CandidateRankingTopK = 10;
Stage0SemanticDedupTolerance = 1e-8;
Stage0StructureScoreEnable = true;
Stage0StructureValidationMultiplier = 4.0;
Stage0StructureValidationWeight = 0.30;
Stage0StructureMachineErrorAbsMSEFloor = 1e-14;
Stage0StructureMachineErrorRelMSEFloor = 1e-14;
Stage0StructureFrontierMaxAbs = 20;
Stage0StructureNeighborhoodMaxDistance = 0.55;
Stage0StructureNeighborhoodMinDistance = 0.10;
Stage0StructureNeighborhoodComplexityWindow = 8;

%% Stage 1 and Stage 2
% Unlike the generator case, the degree-two polynomial block is replaced by
% K fixed neural ridges. The constant and all first-order branch coordinates
% are retained, so only the nonlinear augmentation family is changed.
Stage1EnableAugmentation = true;
Stage1AugmentationMode = 'fixed_neural_ridge';
Stage1AugmentationNeuralCount = LorenzDimension;
Stage1AugmentationNeuralActivation = 'tanh';
Stage1AugmentationNeuralQuantiles = [0.20,0.40,0.60,0.80]; % 4 locations x 3 scales = 12 shapes per 1-D branch
Stage1AugmentationNeuralScales = [0.5,1,2];
Stage1AugmentationNeuralPoolRatio = 3;
Stage1AugmentationNeuralSeed = 1701;
Stage1AugmentationNeuralStdFloor = 1e-10;
Stage1AugmentationNeuralVarianceThreshold = 1e-8;
Stage1AugmentationNeuralCorrelationThreshold = 0.995;
Stage1AugmentationNeuralEnsureFullDirectionalSpan = true;
Stage1AugmentationNeuralIncludeLinearTerms = true;
Stage1ForceStage0SeedOnly = true;
Stage1ExpandBoundsToIncludeStage0Seed = true;
Stage1Stage0SeedBoundMargin = 1e-6;
Stage1RequireExactStage0Reproduction = true;
Stage1Stage0ReproductionRelTolerance = 1e-6;
Stage1Stage0ReproductionAbsTolerance = 1e-10;

Stage2Enable = true;
% Reporting only: inspect whether zero-initialized neural/linear augmentation
% coefficients move, are validation-accepted, and survive pruning.
Stage2AugmentationDiagnosticsEnable = true;
FinalLSQMaxIter = 500;
FinalLSQMaxFunEvals = 5e4;
FinalLSQMaxRelValIncrease = 1e-10;
PostBPPruneEnable = true;
PostBPPruneNumIterations = 1;
PostBPPruneAbsThreshold = 1e-4;
PostBPPruneRelThreshold = 0;
PostBPPruneMaxRelValIncrease = 1e-4;

%% Dynamic rollout simulation controls
RolloutSolver = 'ode4';
RolloutHorizon = 5.0;
RolloutFixedStep = 0.01;
RolloutNOutputTimes = round(RolloutHorizon/RolloutFixedStep)+1;
RolloutNInitialConditions = 5;
RolloutInitialConditionLowerBound = repmat( ...
    [LorenzForcing-2.3,LorenzForcing+1.8],1,LorenzDimension/2);
RolloutInitialConditionUpperBound = repmat( ...
    [LorenzForcing-1.8,LorenzForcing+2.3],1,LorenzDimension/2);
RolloutReferenceInitialCondition = repmat( ...
    [LorenzForcing-2,LorenzForcing+2],1,LorenzDimension/2);
assert(mod(LorenzDimension,2)==0, ...
    'The alternating OOD/rollout design requires an even LorenzDimension.');
RolloutInitialConditionSeed = 9201;
RolloutMaxStateAbs = max(20,abs(LorenzForcing)+12)*ones(1,LorenzDimension);
RolloutMaxDerivativeAbs = 1e4;
RolloutMaxRhsEvaluationsPerIntegration = 4*(RolloutNOutputTimes-1)+4;
RolloutMaxWallTimePerIntegration = 600;
RolloutMaxWallTimePerMethod = 600;
RolloutAbortAfterConsecutiveFailures = 3;
RolloutProgressEveryIC = 1;
RolloutFailureMetricValue = Inf;

%% Methods and output controls
InitializationMode = 'skip';
EnableOOD = true;
DisplayDictionary = true;
PrintFinalXiMatrices = false;
FinalXiPrintPrecision = 4;
FinalXiPrintOnlyActive = false;
SaveResults = true;
ResultsFile = fullfile(OutputSummaryDir,'soft_saturated_lorenz96_results.mat');

% Targeted prior rerun: retrain only PhDN-G2 and PhDN-G3 using the restored
% raw manual structural priors. G1 and all conventional baselines remain disabled;
% their previously saved results are left untouched and can be shown later by
% the plot-only replot script.
RunPhDNMainModel_G1 = false;
RunPhDNMainModel_G2 = true;
RunPhDNMainModel_G3 = true;
RunMLPBaseline = false;
RunEQLBaseline = false;
RunKANBaseline = false;
RunSINDyBaseline = false;
RunNeuralSINDyBaseline = false;

% Include Stage-0 SR ablations generated by the PhDN methods that are trained
% in this run. With G1 disabled and all recorded-report switches false below,
% this rerun evaluates/saves only the newly generated SR-G2 and SR-G3 together
% with PhDN-G2/G3; no old G1/baseline result is replayed for display.
IncludeStage0SRAblationsInComparison = true;

% Training takes priority. When a Run* switch is false, the matching saved
% G-level result is replayed only when its displayRecordReport switch is true.
RecordedBaselineSourceRoot = OutputCaseRoot;
RecordedBaselineReplayStrict = false;
% Minimal recorded-report mode for replay-only MLX runs. When true, every
% method prints only final architecture/complexity, key ID/OOD metrics, and
% timing needed for scientific comparison. Historical sweep/candidate/term
% tables are omitted; saved result structures remain unchanged.
RecordedReportCompactMode = true;
displayRecordReport_PhDN_G1 = false;
displayRecordReport_PhDN_G2 = false;
displayRecordReport_PhDN_G3 = false;
displayRecordReport_SR1 = false;
displayRecordReport_SR2 = false;
displayRecordReport_SR3 = false;
displayRecordReport_MLP = false;
displayRecordReport_EQL = false;
displayRecordReport_KAN = false;
displayRecordReport_SINDy = false;
displayRecordReport_NeuralSINDy = false;

% -------------------------------------------------------------------------
% MLP case-local sweep controls
% Feynman-style depth/activation sweep, with 64 neurons in every hidden layer.
% Depth is the number of affine layers, so depth D has D-1 hidden layers.
% -------------------------------------------------------------------------
MLPSeed = 1;                       % shared round seed for MLP/KAN/EQL
MLPProtocol = 'kan_feynman_sweep';
MLPWidth = 64;
MLPDepthList = 2:6;
MLPActivationList = {'tanh','relu','silu'};
MLPOptimizer = 'LBFGS';
MLPSteps = 500;
MLPLearningRate = 1.0;
MLPDtype = 'float64';
MLPDevice = 'cpu';
MLPTorchNumThreads = 0;
MLPNormalizeInputs = true;
MLPNormalizeOutputs = true;
MLPPythonExe = Stage0PythonExe;
MLPPyKANRoot = fullfile(projectRoot,'third_party','pykan');
MLPWorkRoot = fullfile(projectRoot,'tmp','mlp_soft_saturated_lorenz96_runs');
MLPDepthEarlyStop = true;
MLPDepthEarlyStopPatience = 1;
MLPDepthEarlyStopRelativeTolerance = 0.0;
MLPDisplaySweepTable = true;
MLPVerbose = true;

% -------------------------------------------------------------------------
% KAN case-local sweep controls
% -------------------------------------------------------------------------
KANPythonExe = Stage0PythonExe;
PyKANRoot = fullfile(projectRoot,'third_party','pykan');
KANWidth = LorenzDimension;
KANDepthList = 2:6;
KANGridList = [3,5,10,20,50];
KANSplineOrder = 3;
% Accuracy-first training uses lambda=0 before any sparsification. These mild
% lambdas are attempted only after an accurate unpruned grid checkpoint exists.
KANSparsificationLambdaList = [1e-5,1e-4,1e-3];
KANStepsPerGrid = 200;
KANAccuracyStepsPerGrid = 200;
KANSparsificationSteps = 200;
KANRecoveryStepsPerGrid = 200;
KANOptimizer = 'LBFGS';
KANLearningRate = 1.0;
KANPruneNodeThreshold = 1e-2;
KANPruneEdgeThreshold = 3e-2;
% Both pruned and unpruned fallback branches complete zero-lambda recovery and
% the remaining grid-refinement schedule before validation comparison.
KANPruneValidationGuardEnable = true;
KANPruneMaxRelativeValidationIncrease = 0.0;
KANGridEarlyStop = true;
KANGridEarlyStopPatience = 2;
KANGridEarlyStopRelativeTolerance = 0.015;
KANDepthEarlyStop = true;
KANDepthEarlyStopPatience = 1; % stop after the next meaningfully worse depth
KANDepthEarlyStopRelativeTolerance = 0.01;
% For nested sample sizes, KAN complexity is defined by spline grid G only.
% A larger sample size starts at least from the previously selected G and also
% receives the previous selected native model as a non-regression/warm candidate.
KANEnforceNondecreasingGridAcrossSamples = true;
KANWarmStartAcrossSamples = true;
KANDtype = 'float64';
KANDevice = 'cpu';
KANTorchNumThreads = 0;
KANNormalizeInputs = true;
KANNormalizeOutputs = true;
KANWorkRoot = fullfile(projectRoot,'tmp','kan_soft_saturated_lorenz96_runs');
KANDisplaySweepTable = true;
KANVerbose = true;

% -------------------------------------------------------------------------
% EQL-Div case-local sweep controls
% -------------------------------------------------------------------------
EQLPythonExe = 'C:\Users\hhy\miniconda3\envs\eql_official\python.exe';
EQLOfficialRoot = fullfile(projectRoot,'baselines','eql','official_eql');
EQLDepthList = [2,3,4,5];
EQLLambdaList = [1e-5,1e-3];
EQLUnitsPerUnaryType = 10;
EQLMultiplicationUnits = 10;
EQLStepsPerHiddenLayer = 3000;
EQLBatchSize = 20;
EQLLearningRate = 1e-3;
EQLGradient = 'adam';
EQLLambdaL2 = 0;
EQLPenaltyEvery = 50;
EQLValidateEvery = 10;
EQLCandidateWorkers = 0;
EQLOfficialVerbose = false;
EQLNormalizeInputs = true;
EQLNormalizeOutputs = true;
EQLTheanoFlags = 'device=cpu,floatX=float64,optimizer=fast_run,exception_verbosity=high';
EQLWorkRoot = fullfile(projectRoot,'tmp','eql_soft_saturated_lorenz96_runs');
EQLDepthEarlyStop = true;
EQLDepthEarlyStopPatience = 1;
EQLDepthEarlyStopRelativeTolerance = 0.015;
% Paper protocol: every reported point is a model trained on exactly the
% current N samples. The previous smaller-N model is used only as a fixed-Val
% target and optional initialization for a new current-N official EQL run; it
% is never copied unchanged into the current-N paper curve.
EQLFullDepthScheduleEachSample = true;
EQLCheckpointSelectionMode = 'physical_validation_mse';
EQLUsePreviousModelAsSearchTargetAcrossSamples = true;
EQLWarmStartPreviousModel = true;
EQLWarmStartRestarts = 1;
% If the first current-N sweep does not strictly beat the previous fixed-Val
% reference, rerun the best two L/lambda configurations with up to three new
% independent seeds. Early stopping is overridden while the target is pending.
EQLAdaptiveRescueRestarts = 3;
EQLAdaptiveRescueTopK = 2;
EQLStrictImprovementRelativeMargin = 1e-3; % require at least 0.1% fixed-Val improvement
EQLStrictImprovementAbsoluteMargin = 0.0;
EQLStrictTargetOverridesDepthEarlyStop = true;
EQLDisplaySweepTable = true;
EQLVerbose = true;

% Build the complete option structs using the case-local helper, then apply
% every user-facing control above explicitly. The helper is not the hidden
% source of these settings; it only supplies unchanged shared/default fields.
baselineSweep = soft_saturated_lorenz96_baseline_sweep_options( ...
    projectRoot,Stage0PythonExe);
baselineSweep.runMLP = RunMLPBaseline;
baselineSweep.runEQL = RunEQLBaseline;
baselineSweep.runKAN = RunKANBaseline;
baselineSweep.runSINDy = RunSINDyBaseline;
baselineSweep.runNeuralSINDy = RunNeuralSINDyBaseline;

baselineSweep.mlp.protocol = MLPProtocol;
baselineSweep.mlp.width = MLPWidth;
baselineSweep.mlp.depthList = MLPDepthList;
baselineSweep.mlp.activationList = MLPActivationList;
baselineSweep.mlp.optimizer = MLPOptimizer;
baselineSweep.mlp.steps = MLPSteps;
baselineSweep.mlp.learningRate = MLPLearningRate;
baselineSweep.mlp.dtype = MLPDtype;
baselineSweep.mlp.device = MLPDevice;
baselineSweep.mlp.torchNumThreads = MLPTorchNumThreads;
baselineSweep.mlp.normalizeInputs = MLPNormalizeInputs;
baselineSweep.mlp.normalizeOutputs = MLPNormalizeOutputs;
baselineSweep.mlp.pythonExe = MLPPythonExe;
baselineSweep.mlp.pykanRoot = MLPPyKANRoot;
baselineSweep.mlp.workRoot = MLPWorkRoot;
baselineSweep.mlp.depthEarlyStop = MLPDepthEarlyStop;
baselineSweep.mlp.depthEarlyStopPatience = MLPDepthEarlyStopPatience;
baselineSweep.mlp.depthEarlyStopRelativeTolerance = MLPDepthEarlyStopRelativeTolerance;
baselineSweep.mlp.displaySweepTable = MLPDisplaySweepTable;
baselineSweep.mlp.verbose = MLPVerbose;

baselineSweep.kan.pythonExe = KANPythonExe;
baselineSweep.kan.pykanRoot = PyKANRoot;
baselineSweep.kan.width = KANWidth;
baselineSweep.kan.depthList = KANDepthList;
baselineSweep.kan.minimumDepth = min(KANDepthList);
baselineSweep.kan.gridList = KANGridList;
baselineSweep.kan.splineOrder = KANSplineOrder;
baselineSweep.kan.sparsificationLambdaList = KANSparsificationLambdaList;
baselineSweep.kan.stepsPerGrid = KANStepsPerGrid;
baselineSweep.kan.accuracyStepsPerGrid = KANAccuracyStepsPerGrid;
baselineSweep.kan.sparsificationSteps = KANSparsificationSteps;
baselineSweep.kan.recoveryStepsPerGrid = KANRecoveryStepsPerGrid;
baselineSweep.kan.optimizer = KANOptimizer;
baselineSweep.kan.learningRate = KANLearningRate;
baselineSweep.kan.pruneNodeThreshold = KANPruneNodeThreshold;
baselineSweep.kan.pruneEdgeThreshold = KANPruneEdgeThreshold;
baselineSweep.kan.pruneValidationGuardEnable = KANPruneValidationGuardEnable;
baselineSweep.kan.pruneMaxRelativeValidationIncrease = KANPruneMaxRelativeValidationIncrease;
baselineSweep.kan.minimumGrid = min(KANGridList);
baselineSweep.kan.warmStartEnable = KANWarmStartAcrossSamples;
baselineSweep.kan.warmStartCheckpointPath = '';
baselineSweep.kan.warmStartNormalization = struct();
baselineSweep.kan.gridEarlyStop = KANGridEarlyStop;
baselineSweep.kan.gridEarlyStopPatience = KANGridEarlyStopPatience;
baselineSweep.kan.gridEarlyStopRelativeTolerance = KANGridEarlyStopRelativeTolerance;
baselineSweep.kan.depthEarlyStop = KANDepthEarlyStop;
baselineSweep.kan.depthEarlyStopPatience = KANDepthEarlyStopPatience;
baselineSweep.kan.depthEarlyStopRelativeTolerance = KANDepthEarlyStopRelativeTolerance;
baselineSweep.kan.dtype = KANDtype;
baselineSweep.kan.device = KANDevice;
baselineSweep.kan.torchNumThreads = KANTorchNumThreads;
baselineSweep.kan.normalizeInputs = KANNormalizeInputs;
baselineSweep.kan.normalizeOutputs = KANNormalizeOutputs;
baselineSweep.kan.workRoot = KANWorkRoot;
baselineSweep.kan.displaySweepTable = KANDisplaySweepTable;
baselineSweep.kan.verbose = KANVerbose;

baselineSweep.eql.pythonExe = EQLPythonExe;
baselineSweep.eql.officialRoot = EQLOfficialRoot;
baselineSweep.eql.depthList = EQLDepthList;
baselineSweep.eql.minimumDepth = min(EQLDepthList); % compatibility field; adapter enforces full list
baselineSweep.eql.fullDepthScheduleEachSample = EQLFullDepthScheduleEachSample;
baselineSweep.eql.checkpointSelectionMode = EQLCheckpointSelectionMode;
baselineSweep.eql.warmStartPreviousModel = EQLWarmStartPreviousModel;
baselineSweep.eql.warmStartRestarts = EQLWarmStartRestarts;
baselineSweep.eql.adaptiveRescueRestarts = EQLAdaptiveRescueRestarts;
baselineSweep.eql.adaptiveRescueTopK = EQLAdaptiveRescueTopK;
baselineSweep.eql.strictImprovementRelativeMargin = EQLStrictImprovementRelativeMargin;
baselineSweep.eql.strictImprovementAbsoluteMargin = EQLStrictImprovementAbsoluteMargin;
baselineSweep.eql.strictTargetOverridesDepthEarlyStop = EQLStrictTargetOverridesDepthEarlyStop;
baselineSweep.eql.lambdaList = EQLLambdaList;
baselineSweep.eql.unitsPerUnaryType = EQLUnitsPerUnaryType;
baselineSweep.eql.multiplicationUnits = EQLMultiplicationUnits;
baselineSweep.eql.stepsPerHiddenLayer = EQLStepsPerHiddenLayer;
baselineSweep.eql.batchSize = EQLBatchSize;
baselineSweep.eql.learningRate = EQLLearningRate;
baselineSweep.eql.gradient = EQLGradient;
baselineSweep.eql.lambdaL2 = EQLLambdaL2;
baselineSweep.eql.penaltyEvery = EQLPenaltyEvery;
baselineSweep.eql.validateEvery = EQLValidateEvery;
baselineSweep.eql.candidateWorkers = EQLCandidateWorkers;
baselineSweep.eql.officialVerbose = EQLOfficialVerbose;
baselineSweep.eql.normalizeInputs = EQLNormalizeInputs;
baselineSweep.eql.normalizeOutputs = EQLNormalizeOutputs;
baselineSweep.eql.theanoFlags = EQLTheanoFlags;
baselineSweep.eql.workRoot = EQLWorkRoot;
baselineSweep.eql.depthEarlyStop = EQLDepthEarlyStop;
baselineSweep.eql.depthEarlyStopPatience = EQLDepthEarlyStopPatience;
baselineSweep.eql.depthEarlyStopRelativeTolerance = EQLDepthEarlyStopRelativeTolerance;
baselineSweep.eql.displaySweepTable = EQLDisplaySweepTable;
baselineSweep.eql.verbose = EQLVerbose;

SINDyThresholdList = [0,1e-8,3e-8,1e-7,3e-7,1e-6,3e-6,1e-5,3e-5,1e-4,3e-4,1e-3];
SINDyMaxSTLSQIter = 10;
SINDyRidgeLambda = 0;
SINDyDictionaryMode = 'general';
SINDyPolyOrder = 2;
% SINDy receives the same matched flat primitive library for all K outputs.
% No completed saturation composite or target-output-specific feature is added.
SINDyUnaryOperators = {'sqrt'};
SINDyTypedPhysicalPriorEnable = false;
SINDyTrigAllowedVariableIndex = [];
SINDyForbidStateDependentDivision = false;
SINDyIncludeUnaryOnMonomials = false; % sqrt is applied only to raw x1,...,xK
SINDyIncludeOperatorCrossTerms = false;
SINDySyncStage0InitialGuesses = false; % keep the declared matched dictionaries exact
ExpectedSINDyLibrarySize = 1+2*LorenzDimension+ ...
    LorenzDimension*(LorenzDimension+1)/2;
% Neural-SINDy = 1 + x1,...,xK + K neural ridges + sqrt(x1),...,sqrt(xK).
ExpectedNeuralSINDyLibrarySize = 1+3*LorenzDimension;
SINDyUsePhdnDictionarySupport = false;
SINDyCenterScaleLibrary = false;
SINDyRemoveNearConstantRows = false;
SINDyVerbose = true;
SINDyMaxTermsToPrint = 40;

% Neural-SINDy uses constant + raw inputs + K distinct neural-ridge bases +
% sqrt(raw inputs). It replaces only the degree-two polynomial block.
NeuralSINDyDictionaryMode = 'neural_general';
NeuralSINDyNeuralCount = Stage1AugmentationNeuralCount;
NeuralSINDyNeuralActivation = Stage1AugmentationNeuralActivation;
NeuralSINDyNeuralQuantiles = Stage1AugmentationNeuralQuantiles;
NeuralSINDyNeuralScales = Stage1AugmentationNeuralScales;
NeuralSINDyNeuralPoolRatio = Stage1AugmentationNeuralPoolRatio;
NeuralSINDyNeuralSeed = Stage1AugmentationNeuralSeed+1009+9176;
NeuralSINDyNeuralStdFloor = Stage1AugmentationNeuralStdFloor;
NeuralSINDyNeuralVarianceThreshold = Stage1AugmentationNeuralVarianceThreshold;
NeuralSINDyNeuralCorrelationThreshold = Stage1AugmentationNeuralCorrelationThreshold;
NeuralSINDyNeuralEnsureFullDirectionalSpan = ...
    Stage1AugmentationNeuralEnsureFullDirectionalSpan;
fprintf('Matched SINDy dictionary dimensions: standard=%d, neural=%d.\n', ...
    ExpectedSINDyLibrarySize,ExpectedNeuralSINDyLibrarySize);
fprintf('Neural-SINDy mandatory raw-input basis count: %d (x1,...,x%d).\n', ...
    LorenzDimension,LorenzDimension);
fprintf('Neural-SINDy builder: %s\n',which('make_sindy_neural_arch'));
fprintf('Shared neural generator: %s\n',which('make_fixed_neural_ridge_terms'));
fprintf('PhDN Lorenz--96 Stage-0 prior-level ablation:');
Stage0PriorInfoList = {Stage0PriorInfo_G1,Stage0PriorInfo_G2,Stage0PriorInfo_G3};
Stage0PriorGuessList = {Stage0SRInitialGuesses_G1, ...
    Stage0SRInitialGuesses_G2,Stage0SRInitialGuesses_G3};
Stage0PriorFractionList = [Stage0SRInitialGuessFraction_G1, ...
    Stage0SRInitialGuessFraction_G2,Stage0SRInitialGuessFraction_G3];
for iPriorDisplay = 1:3
    thisPriorInfo = Stage0PriorInfoList{iPriorDisplay};
    fprintf(['  %s (theory level %d): %s | %d expression(s), ', ...
        'replacement fraction=%.3f.'], ...
        thisPriorInfo.label,thisPriorInfo.theoreticalLevel, ...
        thisPriorInfo.description,numel(Stage0PriorGuessList{iPriorDisplay}), ...
        Stage0PriorFractionList(iPriorDisplay));
end

%% Run nested sample-efficiency sweep
allResults = struct();
systemIdentificationRows = struct([]);
standardSummaryRows = struct([]);
iStandardSummary = 0;

for iRound = 1:NumRounds
    roundStage0RandomState = Stage0RandomState+(iRound-1)*RoundRandomStateStride;
    roundMlpSeed = MLPSeed+(iRound-1)*RoundRandomStateStride;
    previousEQLResult = [];
    previousKANSelectedGrid = min(KANGridList);
    previousKANCheckpointPath = '';
    previousKANNormalization = struct();

    fprintf('\n############################################################\n');
    fprintf('SoftSaturatedLorenz96 round %d/%d | PySR state=%d | MLP seed=%d\n', ...
        iRound,NumRounds,roundStage0RandomState,roundMlpSeed);
    fprintf('############################################################\n');
    fprintf('MLP sweep: width=%d, depth=[%s], activations={%s}, depthEarlyStop=%d\n', ...
        baselineSweep.mlp.width,num2str(baselineSweep.mlp.depthList), ...
        strjoin(baselineSweep.mlp.activationList,','),baselineSweep.mlp.depthEarlyStop);
    fprintf(['KAN sweep: width=%d, depth=[%s], sparse lambda=[%s], ', ...
        'grid patience=%d, grid tolerance=%.2f%%, depth lookahead=%d, pruneGuard=%d\n'], ...
        baselineSweep.kan.width,num2str(baselineSweep.kan.depthList), ...
        num2str(baselineSweep.kan.sparsificationLambdaList), ...
        baselineSweep.kan.gridEarlyStopPatience, ...
        100*baselineSweep.kan.gridEarlyStopRelativeTolerance, ...
        baselineSweep.kan.depthEarlyStopPatience,baselineSweep.kan.pruneValidationGuardEnable);
    fprintf('EQL sweep: depth=[%s], lambda=[%s], depthEarlyStop=%d\n', ...
        num2str(baselineSweep.eql.depthList),num2str(baselineSweep.eql.lambdaList), ...
        baselineSweep.eql.depthEarlyStop);

    for iN = 1:numel(TrainingSampleList)
        nTrainRequested = TrainingSampleList(iN);
        plan = struct();
        plan.nTrain = nTrainRequested;
        plan.nValidation = NValidationSamples;
        plan.nTest = NIDTestSamples;
        plan.maxTrain = MaxTrainingSamples;
        plan.trainSeed = BaseTrainPoolSeed+iRound-1;
        plan.validationSeed = BaseValidationPoolSeed+iRound-1;
        plan.testSeed = BaseTestPoolSeed+iRound-1;
        plan.oodSeed = BaseOODPoolSeed+iRound-1;
        plan.samplingMethod = SamplingMethod;
        plan.sobolScrambleMethod = SobolScrambleMethod;
        plan.sobolSkip = SobolSkip;
        nTotal = plan.nTrain+plan.nValidation+plan.nTest;

        rng(1); % controlled framework split; sample helper inverse-arranges it
        task = task_soft_saturated_lorenz96( ...
            caseToRun,casemode,LorenzForcing,SaturationKappa);
        assert(task.parameters.F == LorenzForcing && ...
            task.parameters.kappa == SaturationKappa, ...
            'The generated task does not use the demo F/kappa parameters.');
        task.prior.stage0PriorAblation = Stage0PriorAblation;

        % Apply the demo-visible rollout settings explicitly. The task helper
        % supplies the OOD initial-condition box and reference initial point;
        % this demo owns the numerical integration and fail-fast settings.
        task.rollout.solver = RolloutSolver;
        task.rollout.horizon = RolloutHorizon;
        task.rollout.fixedStep = RolloutFixedStep;
        task.rollout.nOutputTimes = RolloutNOutputTimes;
        task.rollout.nInitialConditions = RolloutNInitialConditions;
        task.rollout.initialConditionDomain.lb = RolloutInitialConditionLowerBound;
        task.rollout.initialConditionDomain.ub = RolloutInitialConditionUpperBound;
        task.rollout.referenceInitialCondition = RolloutReferenceInitialCondition;
        task.rollout.initialConditionSeed = RolloutInitialConditionSeed;
        task.rollout.maxStateAbs = RolloutMaxStateAbs;
        task.rollout.maxDerivativeAbs = RolloutMaxDerivativeAbs;
        task.rollout.maxRhsEvaluationsPerIntegration = RolloutMaxRhsEvaluationsPerIntegration;
        task.rollout.maxWallTimePerIntegration = RolloutMaxWallTimePerIntegration;
        task.rollout.maxWallTimePerMethod = RolloutMaxWallTimePerMethod;
        task.rollout.abortAfterConsecutiveFailures = RolloutAbortAfterConsecutiveFailures;
        task.rollout.progressEveryIC = RolloutProgressEveryIC;
        task.rollout.failureMetricValue = RolloutFailureMetricValue;

        expectedSRNames = arrayfun(@(k) sprintf('x%d',k),1:task.nx,'UniformOutput',false);
        assert(isequal(task.variableNames,expectedSRNames), ...
            ['SoftSaturatedLorenz96 must use canonical one-based SR names ', ...
             sprintf('x1,...,x%d in the task file.',task.nx)]);
        task.samplingPlan = plan;
        task.sampleFcn = @(n,domain) sample_soft_saturated_lorenz96_split(n,domain,plan);
        task.DisplaySymbolic = false;

        fprintf('\n============================================================\n');
        fprintf('Case=%s | Ntrain=%d | Nval=%d | Ntest=%d | round=%d', ...
            task.name,plan.nTrain,plan.nValidation,plan.nTest,iRound);
        fprintf(['Data design: scrambled Sobol | nested train prefix + ', ...
            'independent fixed validation/ID-test/OOD pools\n']);
        fprintf('Sobol scramble=%s | skip=%d | seeds train/val/test/OOD=%d/%d/%d/%d\n', ...
            plan.sobolScrambleMethod,plan.sobolSkip,plan.trainSeed, ...
            plan.validationSeed,plan.testSeed,plan.oodSeed);
        fprintf('Case-local SR map: %s\n',task.variableMappingDescription);
        fprintf('============================================================\n');

        opts = phdnn_default_options(task);
        opts.init.mode = lower(strtrim(InitializationMode));
        opts.data.nSamples = nTotal;
        opts.data.ratioTrain = plan.nTrain/nTotal;
        opts.data.ratioVal = plan.nValidation/nTotal;
        opts.ood.enable = EnableOOD;
        % Case-local coordinate policy: Stage-0 SINDy/PySR expressions and the
        % compiled SR seed are both defined in raw physical coordinates. Keep
        % PhDN input/output coordinate normalization disabled so the exact
        % Stage-0 reproduction check compares identical functions. Output-scale
        % imbalance is handled by the existing std-normalized residual objective.
        opts.norm.useInputOutputNorm = false;
        opts.init.objective.normalizeResidual = true;
        opts.init.objective.residualScale = 'std';

        % Stage 0.
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
        opts.stage0.pysr.adaptiveRescueEnable = Stage0AdaptiveRescueEnable;
        opts.stage0.pysr.adaptiveRescueSoftNormalizedMSE = Stage0AdaptiveRescueSoftNormalizedMSE;
        opts.stage0.pysr.adaptiveRescueHardNormalizedMSE = Stage0AdaptiveRescueHardNormalizedMSE;
        opts.stage0.pysr.adaptiveRescueInstabilityFactor = Stage0AdaptiveRescueInstabilityFactor;
        opts.stage0.pysr.adaptiveRescueContinueImprovementRatio = Stage0AdaptiveRescueContinueImprovementRatio;
        opts.stage0.pysr.adaptiveRescueUseKnownNoiseFloor = Stage0AdaptiveRescueUseKnownNoiseFloor;
        opts.stage0.pysr.adaptiveRescueNoiseFloorMultiplier = Stage0AdaptiveRescueNoiseFloorMultiplier;
        opts.stage0.pysr.adaptiveRescueMaxRestartsPerOutput = Stage0AdaptiveRescueMaxRestartsPerOutput;
        opts.stage0.pysr.adaptiveRescuePopulationMultiplier = Stage0AdaptiveRescuePopulationMultiplier;
        opts.stage0.pysr.adaptiveRescueMinPopulations = Stage0AdaptiveRescueMinPopulations;
        opts.stage0.pysr.adaptiveRescueMaxPopulations = Stage0AdaptiveRescueMaxPopulations;
        opts.stage0.pysr.adaptiveRescuePopulations = Stage0AdaptiveRescueExplicitPopulations;
        opts.stage0.pysr.adaptiveRescueNIterations = Stage0AdaptiveRescueNIterations;
        opts.stage0.pysr.adaptiveRescueMaxOutputs = Stage0AdaptiveRescueMaxOutputs;
        opts.stage0.pysr.adaptiveRescueUseOutputSpecificInitialGuess = Stage0AdaptiveRescueUseOutputSpecificInitialGuess;
        opts.stage0.pysr.adaptiveRescueSeedOffset = Stage0AdaptiveRescueSeedOffset;
        opts.stage0.pysr.adaptiveRescueOutputSeedStride = Stage0AdaptiveRescueOutputSeedStride;
        opts.stage0.pysr.adaptiveRescueRestartSeedStride = Stage0AdaptiveRescueRestartSeedStride;
        % Official growth-biased mutation and weak-selection controls.
        opts.stage0.pysr.tournamentSelectionN = Stage0TournamentSelectionN;
        opts.stage0.pysr.tournamentSelectionP = Stage0TournamentSelectionP;
        opts.stage0.pysr.crossoverProbability = Stage0CrossoverProbability;
        opts.stage0.pysr.weightAddNode = Stage0WeightAddNode;
        opts.stage0.pysr.weightInsertNode = Stage0WeightInsertNode;
        opts.stage0.pysr.weightDeleteNode = Stage0WeightDeleteNode;
        opts.stage0.pysr.weightDoNothing = Stage0WeightDoNothing;
        opts.stage0.pysr.weightMutateConstant = Stage0WeightMutateConstant;
        opts.stage0.pysr.weightMutateOperator = Stage0WeightMutateOperator;
        opts.stage0.pysr.weightMutateFeature = Stage0WeightMutateFeature;
        opts.stage0.pysr.weightSwapOperands = Stage0WeightSwapOperands;
        opts.stage0.pysr.weightRotateTree = Stage0WeightRotateTree;
        opts.stage0.pysr.weightRandomize = Stage0WeightRandomize;
        opts.stage0.pysr.weightSimplify = Stage0WeightSimplify;
        opts.stage0.pysr.weightOptimize = Stage0WeightOptimize;
        opts.stage0.pysr.optimizeProbability = Stage0OptimizeProbability;
        opts.stage0.pysr.shouldSimplify = Stage0ShouldSimplify;
        opts.stage0.pysr.maxDepth = Stage0MaxDepth;
        opts.stage0.pysr.maxSize = Stage0MaxSize;
        opts.stage0.pysr.parsimony = Stage0Parsimony;
        if Stage0UseCustomSRGrammar
            opts.stage0.pysr.binaryOperators = Stage0SRBinaryOperators;
            opts.stage0.pysr.unaryOperators = Stage0SRUnaryOperators;
            opts.stage0.pysr.operatorComplexities = Stage0SROperatorComplexities;
            opts.stage0.pysr.forbidNestedTrig = Stage0SRForbidNestedTrig;
            opts.stage0.pysr.forbidNestedSquare = Stage0SRForbidNestedSquare;
            opts.stage0.pysr.forbidNestedSqrt = Stage0SRForbidNestedSqrt;
            opts.stage0.pysr.minimumPySRVersion = Stage0SRMinimumPySRVersion;
            opts.stage0.pysr.requirePySR2 = Stage0SRRequirePySR2;
            opts.stage0.pysr.initialGuessesEnable = false;
            opts.stage0.pysr.initialGuesses = {};
            opts.stage0.pysr.fractionReplacedGuesses = Stage0SRInitialGuessFraction;
            opts.stage0.pysr.initialGuessScope = 'shared_all_unresolved_outputs';
            opts.stage0.pysr.typedPhysicalPriorEnable = Stage0TypedPhysicalPriorEnable;
            opts.stage0.pysr.typedPhysicalConstraints = Stage0TypedPhysicalPriorMode;
            opts.stage0.pysr.trigAllowedVariables = Stage0SRTrigAllowedVariables;
            opts.stage0.pysr.forbidStateDependentDivision = Stage0SRForbidStateDependentDivision;
            opts.stage0.pysr.strictTrigAtomsOnly = Stage0SRStrictTrigAtomsOnly;
        end
        opts.stage0.pysr.modelSelection = Stage0ModelSelection;
        opts.stage0.pysr.randomState = roundStage0RandomState;
        opts.stage0.pysr.deterministic = Stage0Deterministic;
        opts.stage0.pysr.parallelism = Stage0Parallelism;
        opts.stage0.pysr.batching = Stage0Batching;
        opts.stage0.pysr.batchSize = Stage0BatchSizeList(iN);
        opts.stage0.pysr.strictDeterministicTestMode = Stage0StrictDeterministicTestMode;
        opts.stage0.pysr.repeatabilityPredictionTolerance = Stage0RepeatabilityPredictionTolerance;
        opts.stage0.pysr.machinePrecisionEarlyStopEnable = Stage0MachinePrecisionEarlyStopEnable;
        opts.stage0.pysr.machinePrecisionEarlyStopAbsMSE = Stage0MachinePrecisionEarlyStopAbsMSE;
        opts.stage0.pysr.machinePrecisionEarlyStopRelMSE = Stage0MachinePrecisionEarlyStopRelMSE;
        opts.stage0.pysr.machinePrecisionEarlyStopCheckInterval = Stage0MachinePrecisionEarlyStopCheckInterval;
        opts.stage0.pysr.machinePrecisionEarlyStopMinIterations = Stage0MachinePrecisionEarlyStopMinIterations;
        opts.stage0.pysr.machinePrecisionEarlyStopAcrossRestarts = Stage0MachinePrecisionEarlyStopAcrossRestarts;
        opts.stage0.pysr.verbosity = Stage0Verbosity;
        opts.stage0.pysr.progress = Stage0Progress;
        opts.stage0.pysr.topKExpressionsToReport = Stage0TopKExpressionsToReport;
        opts.stage0.pysr.displayCandidateRankings = Stage0DisplayCandidateRankings;
        opts.stage0.pysr.candidateRankingTopK = Stage0CandidateRankingTopK;
        opts.stage0.pysr.semanticDedupTolerance = Stage0SemanticDedupTolerance;
        opts.stage0.pysr.structureScoreEnable = Stage0StructureScoreEnable;
        opts.stage0.pysr.structureValidationMultiplier = Stage0StructureValidationMultiplier;
        opts.stage0.pysr.structureValidationWeight = Stage0StructureValidationWeight;
        opts.stage0.pysr.structureMachineErrorAbsMSEFloor = Stage0StructureMachineErrorAbsMSEFloor;
        opts.stage0.pysr.structureMachineErrorRelMSEFloor = Stage0StructureMachineErrorRelMSEFloor;
        opts.stage0.pysr.structureFrontierMaxAbs = Stage0StructureFrontierMaxAbs;
        opts.stage0.pysr.structureNeighborhoodMaxDistance = Stage0StructureNeighborhoodMaxDistance;
        opts.stage0.pysr.structureNeighborhoodMinDistance = Stage0StructureNeighborhoodMinDistance;
        opts.stage0.pysr.structureNeighborhoodComplexityWindow = Stage0StructureNeighborhoodComplexityWindow;
        opts.stage0.verbose = true;

        % Stage 1: SR-compiled DAG plus fixed neural-ridge augmentation.
        opts.stage1.method = 'sr_to_phdn_augmented_dag_initialization';
        opts.stage1.dictionaryMode = ...
            'sr_structural_dag_plus_uniform_fixed_neural_ridge_augmentation';
        opts.stage1.enableAugmentation = Stage1EnableAugmentation;
        opts.stage1.includeBestExpressionPath = true;
        opts.stage1.augmentationMode = Stage1AugmentationMode;
        opts.stage1.augmentationNeuralCount = Stage1AugmentationNeuralCount;
        opts.stage1.augmentationNeuralActivation = Stage1AugmentationNeuralActivation;
        opts.stage1.augmentationNeuralQuantiles = Stage1AugmentationNeuralQuantiles;
        opts.stage1.augmentationNeuralScales = Stage1AugmentationNeuralScales;
        opts.stage1.augmentationNeuralPoolRatio = Stage1AugmentationNeuralPoolRatio;
        opts.stage1.augmentationNeuralSeed = Stage1AugmentationNeuralSeed;
        opts.stage1.augmentationNeuralStdFloor = Stage1AugmentationNeuralStdFloor;
        opts.stage1.augmentationNeuralVarianceThreshold = ...
            Stage1AugmentationNeuralVarianceThreshold;
        opts.stage1.augmentationNeuralCorrelationThreshold = ...
            Stage1AugmentationNeuralCorrelationThreshold;
        opts.stage1.augmentationNeuralEnsureFullDirectionalSpan = ...
            Stage1AugmentationNeuralEnsureFullDirectionalSpan;
        opts.stage1.augmentationNeuralIncludeLinearTerms = ...
            Stage1AugmentationNeuralIncludeLinearTerms;
        opts.stage1.forceStage0SeedOnly = Stage1ForceStage0SeedOnly;
        opts.stage1.expandBoundsToIncludeStage0Seed = Stage1ExpandBoundsToIncludeStage0Seed;
        opts.stage1.stage0SeedBoundMargin = Stage1Stage0SeedBoundMargin;
        opts.stage1.requireExactStage0Reproduction = Stage1RequireExactStage0Reproduction;
        opts.stage1.stage0ReproductionRelTolerance = Stage1Stage0ReproductionRelTolerance;
        opts.stage1.stage0ReproductionAbsTolerance = Stage1Stage0ReproductionAbsTolerance;
        opts.stage1.verbose = true;

        % Stage 2.
        opts.stage2.enable = Stage2Enable;
        opts.init.lsq.enable = Stage2Enable;
        opts.init.lsq.maxIter = FinalLSQMaxIter;
        opts.init.lsq.maxFunEvals = FinalLSQMaxFunEvals;
        opts.init.lsq.maxRelValIncrease = FinalLSQMaxRelValIncrease;
        opts.init.augmentationDiagnostics.enable = ...
            Stage2Enable && Stage2AugmentationDiagnosticsEnable;
        opts.init.augmentationDiagnostics.nonzeroThreshold = 1e-10;
        opts.init.augmentationDiagnostics.printPerOutput = true;
        opts.init.postBPPrune.enable = Stage2Enable && PostBPPruneEnable;
        opts.init.postBPPrune.numIterations = PostBPPruneNumIterations;
        opts.init.postBPPrune.absThreshold = PostBPPruneAbsThreshold;
        opts.init.postBPPrune.relThreshold = PostBPPruneRelThreshold;
        opts.init.postBPPrune.contributionAbsThreshold = PostBPPruneAbsThreshold;
        opts.init.postBPPrune.contributionRelThreshold = PostBPPruneRelThreshold;
        opts.init.postBPPrune.maxRelValIncrease = PostBPPruneMaxRelValIncrease;
        opts.init.postBPPrune.acceptByValidation = true;
        opts.init.postBPPrune.verbose = true;
        opts.output.skipSymbolicDisplay = true;
        opts.output.printFinalXiMatrices = PrintFinalXiMatrices;
        opts.output.finalXiPrintPrecision = FinalXiPrintPrecision;
        opts.output.finalXiPrintOnlyActive = PrintFinalXiMatrices && FinalXiPrintOnlyActive;

        if DisplayDictionary
            fprintf(['Structure selection MSE floor: max(abs %.3e, rel %.3e ', ...
                '* max(1,mean(y_val.^2))); simplicity-first within floor.\n'], ...
                Stage0StructureMachineErrorAbsMSEFloor, ...
                Stage0StructureMachineErrorRelMSEFloor);
            fprintf('Stage 0: bypass=%d, restarts=%d, iterations=%d, total populations=%d\n', ...
                opts.stage0.singleLayerBypassEnable,opts.stage0.pysr.innerNumRestarts, ...
                opts.stage0.pysr.innerNIterations,opts.stage0.pysr.innerPopulations);
            fprintf(['Stage 0 Lorenz--96 grammar: binary={%s}; unary={%s}; ', ...
                'target composite omitted.\n'], ...
                strjoin(Stage0SRBinaryOperators,','), ...
                strjoin(Stage0SRUnaryOperators,','));
            fprintf(['Stage 0 PySR batching: enabled=%d, batch size=%d; ', ...
                'full training set retained for Hall-of-Fame evaluation.\n'], ...
                opts.stage0.pysr.batching,opts.stage0.pysr.batchSize);
            fprintf(['Stage 0 machine-precision early stop: enabled=%d, ', ...
                'all outputs required, abs/rel MSE=%.3e/%.3e, ', ...
                'check/min iterations=%d/%d, across restarts=%d.\n'], ...
                opts.stage0.pysr.machinePrecisionEarlyStopEnable, ...
                opts.stage0.pysr.machinePrecisionEarlyStopAbsMSE, ...
                opts.stage0.pysr.machinePrecisionEarlyStopRelMSE, ...
                opts.stage0.pysr.machinePrecisionEarlyStopCheckInterval, ...
                opts.stage0.pysr.machinePrecisionEarlyStopMinIterations, ...
                opts.stage0.pysr.machinePrecisionEarlyStopAcrossRestarts);
            fprintf(['Stage 0 forbidden nesting: trig-in-trig=%d, ', ...
                'square-in-square=%d, sqrt-in-sqrt=%d.\n'], ...
                opts.stage0.pysr.forbidNestedTrig, ...
                opts.stage0.pysr.forbidNestedSquare, ...
                opts.stage0.pysr.forbidNestedSqrt);
            fprintf(['Stage 1: SR DAG + %d fixed %s neural-ridge bases per ', ...
                'active branch; Stage 2=%d\n'], ...
                opts.stage1.augmentationNeuralCount, ...
                opts.stage1.augmentationNeuralActivation,opts.stage2.enable);
            fprintf('Model variant: %s\n',task.modelVariant);
            fprintf(['State/output normalization=%d; rollout solver=%s, horizon=%.2f s, ', ...
                'dt=%.4g s, output times=%d, ICs=%d\n'], ...
                opts.norm.useInputOutputNorm,task.rollout.solver,task.rollout.horizon, ...
                task.rollout.fixedStep,task.rollout.nOutputTimes, ...
                task.rollout.nInitialConditions);
            fprintf('Rollout IC box: lb=[%s], ub=[%s], representative=[%s], seed=%d\n', ...
                num2str(task.rollout.initialConditionDomain.lb), ...
                num2str(task.rollout.initialConditionDomain.ub), ...
                num2str(task.rollout.referenceInitialCondition), ...
                task.rollout.initialConditionSeed);
            fprintf(['Rollout guards: max|x|=[%s], max|dx/dt|=%.3g, ', ...
                'RHS cap=%d, integration/method wall caps=%.1f/%.1f s, ', ...
                'abort after %d consecutive failures\n'], ...
                num2str(task.rollout.maxStateAbs),task.rollout.maxDerivativeAbs, ...
                task.rollout.maxRhsEvaluationsPerIntegration, ...
                task.rollout.maxWallTimePerIntegration, ...
                task.rollout.maxWallTimePerMethod, ...
                task.rollout.abortAfterConsecutiveFailures);
            fprintf('Rollout IC shift: %s | source=%s\n', ...
                task.rollout.initialConditionShift,task.rollout.initialConditionDomain.source);
        end

        roundKey = sprintf('round_%02d',iRound);
        sampleKey = sprintf('N_%05d',plan.nTrain);
        sampleOutputDir = fullfile(OutputCaseRoot,roundKey,sampleKey);

        % Build three matched PhDN configurations from one common option
        % template. Only the Stage-0 prior, method field, and isolated PySR
        % work root differ. This mirrors the previous PhDN-G1--G3 demo.
        phdnVariants = make_soft_saturated_lorenz96_phdn_variant( ...
            'PhDN-G1','phdn_g1','Stage0-SR-G1','stage0sr_g1', ...
            RunPhDNMainModel_G1,displayRecordReport_PhDN_G1, ...
            Stage0SRInitialGuesses_G1,opts,Stage0WorkRoot,Stage0PriorInfo_G1);
        phdnVariants(2) = make_soft_saturated_lorenz96_phdn_variant( ...
            'PhDN-G2','phdn_g2','Stage0-SR-G2','stage0sr_g2', ...
            RunPhDNMainModel_G2,displayRecordReport_PhDN_G2, ...
            Stage0SRInitialGuesses_G2,opts,Stage0WorkRoot,Stage0PriorInfo_G2);
        phdnVariants(3) = make_soft_saturated_lorenz96_phdn_variant( ...
            'PhDN-G3','phdn_g3','Stage0-SR-G3','stage0sr_g3', ...
            RunPhDNMainModel_G3,displayRecordReport_PhDN_G3, ...
            Stage0SRInitialGuesses_G3,opts,Stage0WorkRoot,Stage0PriorInfo_G3);

        % Conventional baselines are unique and prior-independent in this case.
        % Use the strongest G3 option context, matching the previous G1--G3 demo.
        sindyMatchedOpts = phdnVariants(3).opts;
        methodPersistenceContexts = struct();
        for iVariant = 1:numel(phdnVariants)
            ctx = make_soft_saturated_lorenz96_method_context( ...
                task,plan,iRound,phdnVariants(iVariant).opts,baselineSweep, ...
                OutputCaseRoot,TrainingSampleList,NumRounds, ...
                roundStage0RandomState,roundMlpSeed);
            methodPersistenceContexts.(phdnVariants(iVariant).fieldName) = ctx;
            methodPersistenceContexts.(phdnVariants(iVariant).stage0FieldName) = ctx;
        end
        methodPersistenceContext = methodPersistenceContexts.phdn_g3;
        methodPersistenceContexts.sindy = methodPersistenceContext;
        methodPersistenceContexts.neural_sindy = methodPersistenceContext;

        if DisplayDictionary
            fprintf('PhDN Lorenz--96 prior-level methods:');
            for iVariant = 1:numel(phdnVariants)
                variant = phdnVariants(iVariant);
                fprintf(['  %s | prior=%s | guesses=%d | replacement ', ...
                    'fraction=%.3f | work root=%s'], ...
                    variant.label,variant.id,numel(variant.initialGuesses), ...
                    variant.opts.stage0.pysr.fractionReplacedGuesses, ...
                    variant.opts.stage0.pysr.workRoot);
            end
            fprintf('SINDy/Neural-SINDy use one prior-independent matched dictionary.');
            fprintf('Stage-1 augmentation: %d fixed %s neural-ridge bases per active branch.', ...
                Stage1AugmentationNeuralCount,Stage1AugmentationNeuralActivation);
        end

        resultPack = struct();
        % Train/replay every G level as an independent method.
        for iVariant = 1:numel(phdnVariants)
            variant = phdnVariants(iVariant);
            resultPhdn = [];
            resultStage0SR = [];
            if variant.runEnabled
                fprintf('############################################################');
                fprintf(['Running %s | prior=%s | K=%d | F=%.3g | ', ...
                    'kappa=%.3g'],variant.label,variant.id,LorenzDimension, ...
                    LorenzForcing,SaturationKappa);
                fprintf('############################################################');
                rng(1);
                resultPhdn = phdnn_identify(task,variant.opts);
                resultPhdn.priorVariant = variant.id;
                resultPhdn.methodFamily = 'phdn';
                resultPhdn.methodLabel = variant.label;
                print_demo_output(task,resultPhdn);
                [resultPhdn,resultStage0SR] = ...
                    attach_soft_saturated_lorenz96_stage0_sr_ablation(resultPhdn);
                resultPhdn = record_soft_saturated_lorenz96_method_report( ...
                    resultPhdn,'phdn',task);
            elseif variant.displayRecordedReport
                [resultPhdn,~] = reuse_soft_saturated_lorenz96_recorded_baseline( ...
                    RecordedBaselineSourceRoot,iRound,plan.nTrain, ...
                    variant.fieldName,variant.label,RecordedBaselineReplayStrict, ...
                    task,variant.opts,RecordedReportCompactMode);
                if ~isempty(resultPhdn)
                    [resultPhdn,resultStage0SR] = ...
                        attach_soft_saturated_lorenz96_stage0_sr_ablation(resultPhdn);
                end
            end
            if isempty(resultPhdn)
                continue;
            end
            resultPhdn.priorVariant = variant.id;
            resultPhdn.priorInfo = variant.priorInfo;
            resultPhdn.methodFamily = 'phdn';
            resultPhdn.methodLabel = variant.label;
            resultPack.(variant.fieldName) = resultPhdn;
            if ~isempty(resultStage0SR) && ...
                    (~isfield(resultStage0SR,'available') || logical(resultStage0SR.available))
                resultStage0SR.priorVariant = variant.id;
                resultStage0SR.priorInfo = variant.priorInfo;
                resultStage0SR.methodFamily = 'stage0-sr';
                resultStage0SR.methodLabel = variant.stage0Label;
                resultPack.(variant.stage0FieldName) = resultStage0SR;
                print_stage0_sr_ablation_result(resultStage0SR,RecordedReportCompactMode);
            end
            if SaveResults
                variantContext = methodPersistenceContexts.(variant.fieldName);
                [resultPack.(variant.fieldName),~] = ...
                    save_soft_saturated_lorenz96_method_result( ...
                    sampleOutputDir,variant.fieldName,variant.label, ...
                    resultPack.(variant.fieldName),variantContext, ...
                    'post_training',[],[],[]);
                if isfield(resultPack,variant.stage0FieldName)
                    [resultPack.(variant.stage0FieldName),~] = ...
                        save_soft_saturated_lorenz96_method_result( ...
                        sampleOutputDir,variant.stage0FieldName,variant.stage0Label, ...
                        resultPack.(variant.stage0FieldName),variantContext, ...
                        'post_training',[],[],[]);
                end
            end
        end

        % Prefer G3 as the shared deterministic data carrier, then G2/G1.
        baselineDataResult = [];
        phdnDataPreference = {'phdn_g3','phdn_g2','phdn_g1'};
        for iDataPreference = 1:numel(phdnDataPreference)
            fieldCandidate = phdnDataPreference{iDataPreference};
            if isfield(resultPack,fieldCandidate)
                baselineDataResult = resultPack.(fieldCandidate);
                break;
            end
        end
        if isempty(baselineDataResult)
            rng(1);
            baselineDataResult = make_baseline_data_result_from_task(task,sindyMatchedOpts);
        end

        if RunMLPBaseline
            mlpOpts = baselineSweep.mlp;
            mlpOpts.seed = roundMlpSeed;
            resultMlp = run_soft_saturated_lorenz96_mlp_baseline_from_phdn_result(baselineDataResult,mlpOpts);
            if isfield(resultMlp,'net')
                resultMlp.parameterCount = count_matlab_network_parameters(resultMlp.net);
                resultMlp.nActiveCoefficients = resultMlp.parameterCount;
            end
            resultMlp = record_soft_saturated_lorenz96_method_report(resultMlp,'mlp');
            resultPack.mlp = resultMlp;
        elseif displayRecordReport_MLP
            [reusedMlp,~] = reuse_soft_saturated_lorenz96_recorded_baseline( ...
                RecordedBaselineSourceRoot,iRound,plan.nTrain,'mlp','MLP', ...
                RecordedBaselineReplayStrict,RecordedReportCompactMode);
            if ~isempty(reusedMlp); resultPack.mlp = reusedMlp; end
        end

        if SaveResults && isfield(resultPack,'mlp')
            [resultPack.mlp,~] = save_soft_saturated_lorenz96_method_result( ...
                sampleOutputDir,'mlp','MLP',resultPack.mlp, ...
                methodPersistenceContext,'post_training',[],[],[]);
        end

        if RunEQLBaseline
            eqlOpts = baselineSweep.eql;
            eqlOpts.seed = roundMlpSeed;
            % The SI adapter ignores any inherited larger minimumDepth and always
            % re-enables the complete EQLDepthList for this sample size.
            eqlOpts.minimumDepth = min(EQLDepthList);
            eqlOpts.fullDepthScheduleEachSample = EQLFullDepthScheduleEachSample;
            eqlOpts.checkpointSelectionMode = EQLCheckpointSelectionMode;
            if EQLUsePreviousModelAsSearchTargetAcrossSamples && ~isempty(previousEQLResult)
                % The wrapper evaluates this model only as a fixed-validation
                % target and optional native-state warm start. It cannot be
                % returned unchanged as the current-N paper result.
                eqlOpts.previousResult = previousEQLResult;
            end
            resultPack.eql = run_soft_saturated_lorenz96_eql_baseline_from_phdn_result(baselineDataResult,eqlOpts);
            resultPack.eql = record_soft_saturated_lorenz96_method_report(resultPack.eql,'eql');
            previousEQLResult = resultPack.eql;
        elseif displayRecordReport_EQL
            [reusedEql,~] = reuse_soft_saturated_lorenz96_recorded_baseline( ...
                RecordedBaselineSourceRoot,iRound,plan.nTrain,'eql','EQL-Div', ...
                RecordedBaselineReplayStrict,RecordedReportCompactMode);
            if ~isempty(reusedEql)
                resultPack.eql = reusedEql;
                previousEQLResult = reusedEql;
            end
        end

        if SaveResults && isfield(resultPack,'eql')
            [resultPack.eql,~] = save_soft_saturated_lorenz96_method_result( ...
                sampleOutputDir,'eql','EQL-Div',resultPack.eql, ...
                methodPersistenceContext,'post_training',[],[],[]);
            previousEQLResult = resultPack.eql;
        end

        if RunKANBaseline
            kanOpts = baselineSweep.kan;
            kanOpts.seed = roundMlpSeed;
            if KANEnforceNondecreasingGridAcrossSamples
                kanOpts.minimumGrid = previousKANSelectedGrid;
            end
            if KANWarmStartAcrossSamples && ~isempty(previousKANCheckpointPath)
                kanOpts.warmStartEnable = true;
                kanOpts.warmStartCheckpointPath = previousKANCheckpointPath;
                kanOpts.warmStartNormalization = previousKANNormalization;
            else
                kanOpts.warmStartCheckpointPath = '';
                kanOpts.warmStartNormalization = struct();
            end
            resultPack.kan = run_soft_saturated_lorenz96_kan_baseline_from_phdn_result( ...
                baselineDataResult,kanOpts);
            resultPack.kan = record_soft_saturated_lorenz96_method_report(resultPack.kan,'kan');
            previousKANSelectedGrid = max(previousKANSelectedGrid,resultPack.kan.grid);
            if isfield(resultPack.kan,'nativeCheckpointPath') && ...
                    exist(resultPack.kan.nativeCheckpointPath,'file') == 2
                previousKANCheckpointPath = resultPack.kan.nativeCheckpointPath;
                previousKANNormalization = resultPack.kan.normalization;
            else
                warning(['Selected KAN native checkpoint was not found; the next ', ...
                    'sample size will keep the minimum-grid floor but cannot warm start.']);
                previousKANCheckpointPath = '';
                previousKANNormalization = struct();
            end
        elseif displayRecordReport_KAN
            [reusedKan,kanReplayInfo] = reuse_soft_saturated_lorenz96_recorded_baseline( ...
                RecordedBaselineSourceRoot,iRound,plan.nTrain,'kan','KAN', ...
                RecordedBaselineReplayStrict,RecordedReportCompactMode);
            if ~isempty(reusedKan)
                resultPack.kan = reusedKan;
                if isfield(reusedKan,'grid') && isscalar(reusedKan.grid) && ...
                        isfinite(reusedKan.grid)
                    previousKANSelectedGrid = max(previousKANSelectedGrid,reusedKan.grid);
                end
                if isfield(reusedKan,'nativeCheckpointPath') && ...
                        exist(reusedKan.nativeCheckpointPath,'file') == 2
                    previousKANCheckpointPath = reusedKan.nativeCheckpointPath;
                    if isfield(reusedKan,'normalization')
                        previousKANNormalization = reusedKan.normalization;
                    end
                elseif isfield(kanReplayInfo,'loadInfo') && ...
                        isfield(kanReplayInfo.loadInfo,'methodRecords') && ...
                        isfield(kanReplayInfo.loadInfo.methodRecords,'kan') && ...
                        isfield(kanReplayInfo.loadInfo.methodRecords.kan,'artifacts')
                    checkpointCandidate = ...
                        kanReplayInfo.loadInfo.methodRecords.kan.artifacts.kanNativeCheckpointPath;
                    if exist(checkpointCandidate,'file') == 2
                        previousKANCheckpointPath = checkpointCandidate;
                    end
                end
            end
        end

        if SaveResults && isfield(resultPack,'kan')
            [resultPack.kan,~] = save_soft_saturated_lorenz96_method_result( ...
                sampleOutputDir,'kan','KAN',resultPack.kan, ...
                methodPersistenceContext,'post_training',[],[],[]);
            if isfield(resultPack.kan,'nativeCheckpointPath') && ...
                    exist(resultPack.kan.nativeCheckpointPath,'file') == 2
                previousKANCheckpointPath = resultPack.kan.nativeCheckpointPath;
                if isfield(resultPack.kan,'normalization')
                    previousKANNormalization = resultPack.kan.normalization;
                end
            end
        end

        if RunSINDyBaseline
            sindyOpts = make_default_sindy_options_for_demo();
            sindyOpts.thresholdList = SINDyThresholdList;
            sindyOpts.maxSTLSQIter = SINDyMaxSTLSQIter;
            sindyOpts.ridgeLambda = SINDyRidgeLambda;
            sindyOpts.dictionaryMode = SINDyDictionaryMode;
            sindyOpts.polyOrder = SINDyPolyOrder;
            sindyOpts.unaryOperators = SINDyUnaryOperators;
            sindyOpts.includeUnaryOnMonomials = SINDyIncludeUnaryOnMonomials;
            sindyOpts.includeOperatorCrossTerms = SINDyIncludeOperatorCrossTerms;
            sindyOpts.typedPhysicalPriorEnable = SINDyTypedPhysicalPriorEnable;
            sindyOpts.trigAllowedVariableIndex = SINDyTrigAllowedVariableIndex;
            sindyOpts.forbidStateDependentDivision = SINDyForbidStateDependentDivision;
            sindyOpts.usePhdnDictionarySupport = SINDyUsePhdnDictionarySupport;
            sindyOpts.centerScaleLibrary = SINDyCenterScaleLibrary;
            sindyOpts.removeNearConstantRows = SINDyRemoveNearConstantRows;
            sindyOpts.syncStage0InitialGuesses = SINDySyncStage0InitialGuesses;
            sindyOpts.strictLibraryAssertions = true;
            sindyOpts.expectedLibrarySize = ExpectedSINDyLibrarySize;
            sindyOpts.expectedNeuralCount = 0;
            sindyOpts.verbose = SINDyVerbose;
            sindyOpts.maxTermsToPrint = SINDyMaxTermsToPrint;
            resultPack.sindy = run_sindy_baseline_from_phdn_result( ...
                baselineDataResult,task,sindyOpts,sindyMatchedOpts);
            resultPack.sindy = record_soft_saturated_lorenz96_method_report(resultPack.sindy,'sindy');
        elseif displayRecordReport_SINDy
            [reusedSindy,~] = reuse_soft_saturated_lorenz96_recorded_baseline( ...
                RecordedBaselineSourceRoot,iRound,plan.nTrain,'sindy','SINDy', ...
                RecordedBaselineReplayStrict,task,sindyMatchedOpts,RecordedReportCompactMode);
            if ~isempty(reusedSindy); resultPack.sindy = reusedSindy; end
        end

        if SaveResults && isfield(resultPack,'sindy')
            [resultPack.sindy,~] = save_soft_saturated_lorenz96_method_result( ...
                sampleOutputDir,'sindy','SINDy',resultPack.sindy, ...
                methodPersistenceContext,'post_training',[],[],[]);
        end

        if RunNeuralSINDyBaseline
            neuralSindyOpts = make_default_sindy_options_for_demo();
            neuralSindyOpts.thresholdList = SINDyThresholdList;
            neuralSindyOpts.maxSTLSQIter = SINDyMaxSTLSQIter;
            neuralSindyOpts.ridgeLambda = SINDyRidgeLambda;
            neuralSindyOpts.dictionaryMode = NeuralSINDyDictionaryMode;
            neuralSindyOpts.polyOrder = SINDyPolyOrder;
            neuralSindyOpts.unaryOperators = SINDyUnaryOperators;
            neuralSindyOpts.includeUnaryOnMonomials = SINDyIncludeUnaryOnMonomials;
            neuralSindyOpts.includeOperatorCrossTerms = SINDyIncludeOperatorCrossTerms;
            neuralSindyOpts.typedPhysicalPriorEnable = SINDyTypedPhysicalPriorEnable;
            neuralSindyOpts.trigAllowedVariableIndex = SINDyTrigAllowedVariableIndex;
            neuralSindyOpts.forbidStateDependentDivision = SINDyForbidStateDependentDivision;
            neuralSindyOpts.usePhdnDictionarySupport = false;
            neuralSindyOpts.centerScaleLibrary = SINDyCenterScaleLibrary;
            neuralSindyOpts.removeNearConstantRows = SINDyRemoveNearConstantRows;
            neuralSindyOpts.syncStage0InitialGuesses = SINDySyncStage0InitialGuesses;
            neuralSindyOpts.strictLibraryAssertions = true;
            neuralSindyOpts.expectedLibrarySize = ExpectedNeuralSINDyLibrarySize;
            neuralSindyOpts.expectedNeuralCount = NeuralSINDyNeuralCount;
            neuralSindyOpts.verbose = SINDyVerbose;
            neuralSindyOpts.maxTermsToPrint = SINDyMaxTermsToPrint;
            neuralSindyOpts.neuralCount = NeuralSINDyNeuralCount;
            neuralSindyOpts.neuralActivation = NeuralSINDyNeuralActivation;
            neuralSindyOpts.neuralQuantiles = NeuralSINDyNeuralQuantiles;
            neuralSindyOpts.neuralScales = NeuralSINDyNeuralScales;
            neuralSindyOpts.neuralPoolRatio = NeuralSINDyNeuralPoolRatio;
            neuralSindyOpts.neuralSeed = NeuralSINDyNeuralSeed;
            neuralSindyOpts.neuralStdFloor = NeuralSINDyNeuralStdFloor;
            neuralSindyOpts.neuralVarianceThreshold = ...
                NeuralSINDyNeuralVarianceThreshold;
            neuralSindyOpts.neuralCorrelationThreshold = ...
                NeuralSINDyNeuralCorrelationThreshold;
            neuralSindyOpts.neuralEnsureFullDirectionalSpan = ...
                NeuralSINDyNeuralEnsureFullDirectionalSpan;
            resultPack.neural_sindy = run_sindy_baseline_from_phdn_result( ...
                baselineDataResult,task,neuralSindyOpts,sindyMatchedOpts);
            resultPack.neural_sindy = record_soft_saturated_lorenz96_method_report( ...
                resultPack.neural_sindy,'neural_sindy');
        elseif displayRecordReport_NeuralSINDy
            [reusedNeuralSindy,~] = reuse_soft_saturated_lorenz96_recorded_baseline( ...
                RecordedBaselineSourceRoot,iRound,plan.nTrain, ...
                'neural_sindy','Neural-SINDy',RecordedBaselineReplayStrict, ...
                task,sindyMatchedOpts,RecordedReportCompactMode);
            if ~isempty(reusedNeuralSindy)
                resultPack.neural_sindy = reusedNeuralSindy;
            end
        end

        if SaveResults && isfield(resultPack,'neural_sindy')
            [resultPack.neural_sindy,~] = save_soft_saturated_lorenz96_method_result( ...
                sampleOutputDir,'neural_sindy','Neural-SINDy', ...
                resultPack.neural_sindy,methodPersistenceContext, ...
                'post_training',[],[],[]);
        end

        % Dynamic rollout and sample-efficiency rows. Every G level has an
        % independent method field, persistence record, timing, and curve.
        methodSpecs = { ...
            'PhDN-G1','phdn_g1'; ...
            'PhDN-G2','phdn_g2'; ...
            'PhDN-G3','phdn_g3'};
        % Stage0-SR rollout selection is independent for G1/G2/G3.
        % The legacy IncludeStage0SRAblationsInComparison=true still forces
        % all three on for backward compatibility.
        stage0SRRolloutEnabled = [ ...
            displayRecordReport_SR1, ...
            displayRecordReport_SR2, ...
            displayRecordReport_SR3];
        if IncludeStage0SRAblationsInComparison
            stage0SRRolloutEnabled(:) = true;
        end
        stage0SRSpecs = { ...
            'Stage0-SR-G1','stage0sr_g1'; ...
            'Stage0-SR-G2','stage0sr_g2'; ...
            'Stage0-SR-G3','stage0sr_g3'};
        methodSpecs = [methodSpecs; stage0SRSpecs(stage0SRRolloutEnabled,:)];

        methodSpecs = [methodSpecs; { ...
            'MLP','mlp'; ...
            'EQL-Div','eql'; ...
            'KAN','kan'; ...
            'SINDy','sindy'; ...
            'Neural-SINDy','neural_sindy'}];
        for iMethod = 1:size(methodSpecs,1)
            methodLabel = methodSpecs{iMethod,1};
            fieldName = methodSpecs{iMethod,2};
            if ~isfield(resultPack,fieldName)
                continue;
            end
            methodResult = resultPack.(fieldName);
            rollout = evaluate_soft_saturated_lorenz96_rollout(task,methodLabel,methodResult);
            systemIdentificationRows = append_system_identification_result_row( ...
                systemIdentificationRows,plan.nTrain,iRound,methodLabel,methodResult,rollout);
            [standardSummaryRows,iStandardSummary] = append_method_summary_row( ...
                standardSummaryRows,iStandardSummary, ...
                sprintf('%s_N%d',task.name,plan.nTrain), ...
                methodLabel,methodResult,iRound);
            if SaveResults
                currentMethodRow = systemIdentificationRows(end);
                currentMethodSummaryRow = standardSummaryRows(end);
                if isfield(methodPersistenceContexts,fieldName)
                    currentPersistenceContext = ...
                        methodPersistenceContexts.(fieldName);
                else
                    currentPersistenceContext = methodPersistenceContext;
                end
                [resultPack.(fieldName),~] = ...
                    save_soft_saturated_lorenz96_method_result( ...
                    sampleOutputDir,fieldName,methodLabel,resultPack.(fieldName), ...
                    currentPersistenceContext,'post_rollout',rollout, ...
                    currentMethodRow,currentMethodSummaryRow);
            end
        end

        roundKey = sprintf('round_%02d',iRound);
        sampleKey = sprintf('N_%05d',plan.nTrain);
        if ~isfield(allResults,roundKey); allResults.(roundKey) = struct(); end
        allResults.(roundKey).(sampleKey) = resultPack;
        allResults.(roundKey).(sampleKey).task = task;
        allResults.(roundKey).(sampleKey).samplingPlan = plan;

        % Display and persist the state trajectories immediately after this
        % training-sample round. Plotting is deliberately failure-tolerant:
        % trained models and scalar/rollout data are still checkpointed if a
        % graphics helper is missing or throws an error.
        currentRows = systemIdentificationRows( ...
            [systemIdentificationRows.nTrain] == plan.nTrain);
        figCurrentTrajectory = [];
        trajectoryFigureData = struct();
        trajectoryPlotError = '';
        try
            [figCurrentTrajectory,trajectoryFigureData] = ...
                plot_soft_saturated_lorenz96_trajectory(task,currentRows,plan.nTrain);
            if isgraphics(figCurrentTrajectory)
                set(figCurrentTrajectory,'Visible','on');
                figure(figCurrentTrajectory);
                drawnow;
            else
                warning('Trajectory plotting returned no valid figure for Ntrain=%d.', ...
                    plan.nTrain);
            end
        catch MEtrajectoryPlot
            trajectoryPlotError = getReport(MEtrajectoryPlot,'extended','hyperlinks','off');
            warning('Trajectory plotting failed for Ntrain=%d, round=%d: %s', ...
                plan.nTrain,iRound,MEtrajectoryPlot.message);
            trajectoryFigureData = struct( ...
                'selectedNTrain',plan.nTrain, ...
                'selectedRound',iRound, ...
                'plotError',trajectoryPlotError, ...
                'generatedAt',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')));
        end

        if SaveResults
            sampleOutputDir = fullfile(OutputCaseRoot,roundKey,sampleKey);
            sampleOutputInfo = save_soft_saturated_lorenz96_sample_outputs( ...
                sampleOutputDir,resultPack,task,plan,iRound,currentRows, ...
                figCurrentTrajectory,trajectoryFigureData);
            allResults.(roundKey).(sampleKey).outputInfo = sampleOutputInfo;
            allResults.(roundKey).(sampleKey).trajectoryPlotError = trajectoryPlotError;
        end
    end
end

%% Summaries and manuscript-oriented figures
completeCaseWallTime = toc(CaseSimulationWallTimer);
fprintf('Complete SoftSaturatedLorenz96 case wall time: %.3f s (%.3f h)', ...
    completeCaseWallTime,completeCaseWallTime/3600);

print_method_comparison_summary_local(standardSummaryRows);
if NumRounds > 1
    print_round_statistics_summary(standardSummaryRows);
end
SampleEfficiencyCaseLabel = 'SoftSaturatedLorenz96';
sampleEfficiencyTable = print_system_identification_sample_efficiency( ...
    systemIdentificationRows,SampleEfficiencyCaseLabel);
figSampleEfficiency = [];
sampleEfficiencyFigureData = struct();
sampleEfficiencyPlotError = '';
try
    [figSampleEfficiency,sampleEfficiencyFigureData] = ...
        plot_system_identification_sample_efficiency( ...
            systemIdentificationRows,SampleEfficiencyCaseLabel);
    if isgraphics(figSampleEfficiency)
        set(figSampleEfficiency,'Visible','on');
        figure(figSampleEfficiency);
        drawnow;
    else
        warning('Sample-efficiency plotting returned no valid figure.');
    end
catch MEsampleEfficiencyPlot
    sampleEfficiencyPlotError = getReport( ...
        MEsampleEfficiencyPlot,'extended','hyperlinks','off');
    warning('Sample-efficiency plotting failed');
    sampleEfficiencyFigureData = struct( ...
        'plotError',sampleEfficiencyPlotError, ...
        'generatedAt',char(datetime('now','Format','yyyy-MM-dd HH:mm:ss')));
end

runMetadata = struct();
runMetadata.generatedAt = char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
runMetadata.completeCaseWallTime = completeCaseWallTime;
runMetadata.trainingSampleList = TrainingSampleList;
runMetadata.nValidationSamples = NValidationSamples;
runMetadata.nIDTestSamples = NIDTestSamples;
runMetadata.numRounds = NumRounds;
runMetadata.samplingMethod = SamplingMethod;
runMetadata.sobolScrambleMethod = SobolScrambleMethod;
runMetadata.outputRoot = OutputCaseRoot;
runMetadata.stage0PriorAblation = Stage0PriorAblation;
runMetadata.recordedBaselineSourceRoot = RecordedBaselineSourceRoot;
runMetadata.recordedBaselineReplayStrict = RecordedBaselineReplayStrict;
runMetadata.methodPersistenceSchema = 'independent_method_record_v1';
runMetadata.methodPersistencePolicy = ...
    'per-method immediate checkpoint + non-destructive legacy/aggregate merge';
runMetadata.eqlPaperSampleEfficiencyProtocol = ...
    'exact_N_current_sample_training_with_adaptive_validation_search';
runMetadata.eqlPreviousModelRole = ...
    'fixed-Val target and optional current-N warm start; never unchanged substitution';
runMetadata.eqlAdaptiveSearch = struct( ...
    'warmStartPreviousModel',EQLWarmStartPreviousModel, ...
    'warmStartRestarts',EQLWarmStartRestarts, ...
    'rescueRestarts',EQLAdaptiveRescueRestarts, ...
    'rescueTopK',EQLAdaptiveRescueTopK, ...
    'strictRelativeMargin',EQLStrictImprovementRelativeMargin, ...
    'strictAbsoluteMargin',EQLStrictImprovementAbsoluteMargin, ...
    'targetOverridesDepthEarlyStop',EQLStrictTargetOverridesDepthEarlyStop);
runMetadata.phdnConfiguration = struct( ...
    'dimension',LorenzDimension, ...
    'forcing',LorenzForcing, ...
    'saturationKappa',SaturationKappa, ...
    'augmentationMode',Stage1AugmentationMode, ...
    'augmentationNeuralCount',Stage1AugmentationNeuralCount, ...
    'augmentationNeuralIncludeLinearTerms',Stage1AugmentationNeuralIncludeLinearTerms, ...
    'augmentationActivation',Stage1AugmentationNeuralActivation, ...
    'augmentationQuantiles',Stage1AugmentationNeuralQuantiles, ...
    'augmentationScales',Stage1AugmentationNeuralScales, ...
    'augmentationDiagnosticsEnabled',Stage2AugmentationDiagnosticsEnable, ...
    'includeStage0SR',IncludeStage0SRAblationsInComparison, ...
    'stage0SRRolloutEnabled',struct( ...
        'G1',displayRecordReport_SR1, ...
        'G2',displayRecordReport_SR2, ...
        'G3',displayRecordReport_SR3));
runMetadata.phdnPriorAblation = struct( ...
    'G1',Stage0PriorAblation.G1, ...
    'G2',Stage0PriorAblation.G2, ...
    'G3',Stage0PriorAblation.G3, ...
    'runEnabled',struct('G1',RunPhDNMainModel_G1, ...
        'G2',RunPhDNMainModel_G2,'G3',RunPhDNMainModel_G3), ...
    'baselineContextLevel','G3', ...
    'includeStage0SRAblations',IncludeStage0SRAblationsInComparison, ...
    'stage0SRRolloutEnabled',struct( ...
        'G1',displayRecordReport_SR1, ...
        'G2',displayRecordReport_SR2, ...
        'G3',displayRecordReport_SR3));
runMetadata.neuralSindyConfiguration = struct( ...
    'enabled',RunNeuralSINDyBaseline, ...
    'dictionaryMode',NeuralSINDyDictionaryMode, ...
    'polynomialTermsReplaced',true, ...
    'includeRawInputs',true, ...
    'rawInputCount',LorenzDimension, ...
    'neuralCount',NeuralSINDyNeuralCount, ...
    'activation',NeuralSINDyNeuralActivation, ...
    'quantiles',NeuralSINDyNeuralQuantiles, ...
    'scales',NeuralSINDyNeuralScales, ...
    'seed',NeuralSINDyNeuralSeed, ...
    'unaryOperators',{SINDyUnaryOperators}, ...
    'includeUnaryOnMonomials',SINDyIncludeUnaryOnMonomials, ...
    'includeOperatorCrossTerms',SINDyIncludeOperatorCrossTerms, ...
    'syncStage0InitialGuesses',SINDySyncStage0InitialGuesses, ...
    'expectedStandardLibrarySize',ExpectedSINDyLibrarySize, ...
    'expectedNeuralLibrarySize',ExpectedNeuralSINDyLibrarySize, ...
    'strictLibraryAssertions',true);
runMetadata.displayRecordReport = struct( ...
    'PhDN_G1',displayRecordReport_PhDN_G1, ...
    'PhDN_G2',displayRecordReport_PhDN_G2, ...
    'PhDN_G3',displayRecordReport_PhDN_G3, ...
    'MLP',displayRecordReport_MLP,'EQL',displayRecordReport_EQL, ...
    'KAN',displayRecordReport_KAN,'SINDy',displayRecordReport_SINDy, ...
    'NeuralSINDy',displayRecordReport_NeuralSINDy);
runMetadata.sampleEfficiencyPlotError = sampleEfficiencyPlotError;

if SaveResults
    summaryOutputInfo = save_soft_saturated_lorenz96_summary_outputs( ...
        OutputSummaryDir,allResults,systemIdentificationRows,standardSummaryRows, ...
        sampleEfficiencyTable,figSampleEfficiency,sampleEfficiencyFigureData, ...
        runMetadata);
    ResultsFile = summaryOutputInfo.resultsMatPath;
    fprintf('Saved SoftSaturatedLorenz96 aggregate results to:\n%s\n',ResultsFile);
end











