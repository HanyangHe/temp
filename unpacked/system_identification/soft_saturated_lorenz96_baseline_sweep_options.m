function sweep = soft_saturated_lorenz96_baseline_sweep_options(projectRoot, stage0PythonExe)
%SOFT_SATURATED_LORENZ96_BASELINE_SWEEP_OPTIONS Case-local baseline sweeps.
%
% Keep the K-state Lorenz--96 baseline policy in one editable file.
% The shared MLP/KAN/EQL defaults remain unchanged for all other cases.
%
% Depth early stopping is validation-only.  At each attempted depth, all
% candidates associated with that depth are completed first.  The sweep stops
% before deeper networks when the best validation MSE at the current depth is
% worse than the best validation MSE achieved by a shallower depth by more than
% the configured relative tolerance for the configured patience.

    if nargin < 1 || isempty(projectRoot)
        projectRoot = fileparts(fileparts(mfilename('fullpath')));
    end
    if nargin < 2 || isempty(stage0PythonExe)
        stage0PythonExe = 'python';
    end

    sweep = struct();

    %% Which baselines are active for this case
    sweep.runMLP = true;
    sweep.runKAN = true;
    sweep.runEQL = true;
    sweep.runSINDy = true;
    sweep.runNeuralSINDy = true;

    %% MLP: Feynman-style depth/activation sweep, width enlarged to 64
    mlp = mlp_default_options();
    mlp.protocol = 'kan_feynman_sweep';
    mlp.width = 64;
    mlp.depthList = 2:6;  % affine-layer count; hidden-layer count is depth-1
    mlp.activationList = {'tanh','relu','silu'};
    mlp.optimizer = 'LBFGS';
    mlp.steps = 500;
    mlp.learningRate = 1.0;
    mlp.pythonExe = stage0PythonExe;
    mlp.pykanRoot = fullfile(projectRoot,'third_party','pykan');
    mlp.workRoot = fullfile(projectRoot,'tmp','mlp_soft_saturated_lorenz96_runs');
    mlp.depthEarlyStop = true;
    mlp.depthEarlyStopPatience = 1;
    mlp.depthEarlyStopRelativeTolerance = 0.0;
    mlp.displaySweepTable = true;
    mlp.verbose = true;
    sweep.mlp = mlp;

    %% KAN: accuracy-first SI sweep with grid inheritance and prune guard
    kan = kan_default_options();
    kan.pythonExe = stage0PythonExe;
    kan.pykanRoot = fullfile(projectRoot,'third_party','pykan');
    kan.workRoot = fullfile(projectRoot,'tmp','kan_soft_saturated_lorenz96_runs');
    kan.width = 8;
    kan.depthList = 2:6;
    kan.minimumDepth = min(kan.depthList);
    kan.gridList = [3,5,10,20,50,100,200];
    kan.minimumGrid = min(kan.gridList);
    kan.splineOrder = 3;
    % Accuracy is learned first with lambda=0. Mild sparsification is applied
    % only after the validation-best unpruned grid checkpoint is available.
    kan.sparsificationLambdaList = [1e-5,1e-4,1e-3];
    kan.stepsPerGrid = 200;
    kan.accuracyStepsPerGrid = 200;
    kan.sparsificationSteps = 200;
    kan.recoveryStepsPerGrid = 200;
    kan.pruneValidationGuardEnable = true;
    kan.pruneMaxRelativeValidationIncrease = 0.0;
    kan.gridEarlyStop = true;
    kan.gridEarlyStopPatience = 2;
    kan.gridEarlyStopRelativeTolerance = 0.015;
    kan.depthEarlyStop = true;
    kan.depthEarlyStopPatience = 2; % test two deeper layers before stopping
    kan.depthEarlyStopRelativeTolerance = 0.0;
    kan.warmStartEnable = true;
    kan.warmStartCheckpointPath = '';
    kan.warmStartNormalization = struct();
    kan.dtype = 'float64';
    kan.displaySweepTable = true;
    kan.verbose = true;
    sweep.kan = kan;

    %% EQL-Div: official depth/lambda sweep + validation depth early stopping
    eql = eql_default_options();
    eql.pythonExe = 'C:\Users\hhy\miniconda3\envs\eql_official\python.exe';
    eql.officialRoot = fullfile(projectRoot,'baselines','eql','official_eql');
    eql.workRoot = fullfile(projectRoot,'tmp','eql_soft_saturated_lorenz96_runs');
    eql.depthList = [2,3,4,5];
    eql.minimumDepth = min(eql.depthList); % compatibility only; SI adapter enforces full list
    eql.fullDepthScheduleEachSample = true;
    eql.checkpointSelectionMode = 'physical_validation_mse';
    eql.previousCandidateRelativeTolerance = 0.0; % legacy compatibility
    eql.previousCandidateAbsoluteTolerance = 0.0; % legacy compatibility
    eql.warmStartPreviousModel = true;
    eql.warmStartRestarts = 1;
    eql.adaptiveRescueRestarts = 3;
    eql.adaptiveRescueTopK = 2;
    eql.strictImprovementRelativeMargin = 1e-3; % require a visible 0.1% Val-MSE decrease
    eql.strictImprovementAbsoluteMargin = 0.0;
    eql.strictTargetOverridesDepthEarlyStop = true;
    % A compact initial lambda set for this expensive case.  This field is
    % intentionally isolated here so the scan can be densified later.
    eql.lambdaList = [1e-5,1e-3];
    eql.unitsPerUnaryType = 10;
    eql.multiplicationUnits = 10;
    eql.stepsPerHiddenLayer = 3000;
    eql.candidateWorkers = 0;
    eql.depthEarlyStop = true;
    eql.depthEarlyStopPatience = 1;
    eql.depthEarlyStopRelativeTolerance = 0.015;
    eql.displaySweepTable = true;
    eql.verbose = true;
    sweep.eql = eql;
end
