function eqlOpts = eql_default_options()
%EQL_DEFAULT_OPTIONS Defaults for the bundled official EQL-Div baseline.
%
% The network, loss, division curriculum, regularization phases, optimizer,
% active-unit count, and Vint-S selector are executed by the unchanged
% martius-lab/EQL Theano source under baselines/eql/official_eql.

    eqlDir = fileparts(mfilename('fullpath'));
    projectRoot = fileparts(fileparts(eqlDir));

    eqlOpts = struct();
    eqlOpts.protocol = 'official_eql_div_theano_external_validation_mse';
    eqlOpts.seed = 1;

    % ICML-2018 unified EQL-Div sweep. L is the paper depth; the upstream
    % hidden-layer count passed to MLFG is L-1.
    eqlOpts.depthList = [2,3,4];
    % Retained only for backward compatibility. The SI adapter always resets
    % this to min(depthList), so nested sample sizes cannot inherit a depth floor.
    eqlOpts.minimumDepth = min(eqlOpts.depthList);
    eqlOpts.fullDepthScheduleEachSample = true;
    eqlOpts.checkpointSelectionMode = 'physical_validation_mse';
    % Cross-sample policy for nested sample-efficiency studies.
    % The previous smaller-N model is a validation target and optional warm
    % start only; it is never copied unchanged into the current-N paper point.
    eqlOpts.previousCandidateRelativeTolerance = 0.0; % legacy compatibility
    eqlOpts.previousCandidateAbsoluteTolerance = 0.0; % legacy compatibility
    eqlOpts.warmStartPreviousModel = true;
    eqlOpts.warmStartRestarts = 1;
    eqlOpts.adaptiveRescueRestarts = 3;
    eqlOpts.adaptiveRescueTopK = 2;
    eqlOpts.strictImprovementRelativeMargin = 1e-3; % 0.1% guard against numerical pseudo-improvement
    eqlOpts.strictImprovementAbsoluteMargin = 0.0;
    eqlOpts.strictTargetOverridesDepthEarlyStop = true;
    eqlOpts.lambdaList = 10.^(-6:0.1:-3.5);
    eqlOpts.unitsPerUnaryType = 10;
    eqlOpts.multiplicationUnits = 10; % upstream uses the same n_per_base
    eqlOpts.operatorFamily = {'identity','sin','cos','multiply','division'};

    % Paper training settings, while all update equations remain upstream.
    eqlOpts.stepsPerHiddenLayer = 10000;
    eqlOpts.batchSize = 20;
    eqlOpts.learningRate = 1e-3;
    eqlOpts.gradient = 'adam';
    eqlOpts.lambdaL2 = 0; % Eq. (8) uses squared-error loss + lambda*L1
    eqlOpts.penaltyEvery = 50;
    eqlOpts.validateEvery = 10;

    % Candidate scans are independent in the official implementation.
    % 0 means automatic candidate-level parallelism: the Python adapter uses
    % min(number of candidates, available logical CPU count). A positive
    % integer remains available as an explicit manual cap.
    eqlOpts.candidateWorkers = 0;
    eqlOpts.officialVerbose = false;

    % The attached upstream experiments use inputs in [-1,1] and outputs of
    % order one. For arbitrary Feynman domains, the adapter applies only an
    % external affine scale transform; the official model/training code is
    % unchanged. The bundled Eq. (11) acceptance demo disables both transforms.
    eqlOpts.normalizeInputs = true;
    eqlOpts.normalizeOutputs = true;

    % External/runtime controls.
    eqlOpts.pythonExe = 'python';
    eqlOpts.officialRoot = fullfile(eqlDir,'official_eql');
    eqlOpts.theanoFlags = 'device=cpu,floatX=float64,optimizer=fast_run,exception_verbosity=high';
    eqlOpts.workRoot = fullfile(projectRoot,'tmp','eql_official_runs');
    eqlOpts.displaySweepTable = true;
    eqlOpts.verbose = true;
    eqlOpts.useBundledOfficialEq11Data = false;
end
