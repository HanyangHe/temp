function result = train_kan_paper_mlp_sweep_baseline(XTrain, YTrain, XVal, YVal, XTest, YTest, XOod, YOod, mlpOpts)
%TRAIN_KAN_PAPER_MLP_SWEEP_BASELINE Sweep KAN-paper MLP structures in Python.
    if nargin < 9 || isempty(mlpOpts); mlpOpts = mlp_default_options(); end
    if nargin < 7 || isempty(XOod); XOod = zeros(0,size(XTrain,2)); YOod = zeros(0,size(YTrain,2)); end
    mlpDir = fileparts(mfilename('fullpath'));
    adapterPath = fullfile(mlpDir, 'kan_paper_mlp_adapter.py');
    cfg = struct();
    cfg.pykan_root = mlpOpts.pykanRoot;
    cfg.seed = mlpOpts.seed;
    cfg.width = mlpOpts.width;
    cfg.depth_list = mlpOpts.depthList;
    cfg.activation_list = mlpOpts.activationList;
    cfg.steps = mlpOpts.steps;
    cfg.optimizer = mlpOpts.optimizer;
    cfg.learning_rate = mlpOpts.learningRate;
    cfg.dtype = mlpOpts.dtype;
    cfg.device = mlpOpts.device;
    cfg.torch_num_threads = mlpOpts.torchNumThreads;
    cfg.normalize_inputs = logical(mlpOpts.normalizeInputs);
    cfg.normalize_outputs = logical(mlpOpts.normalizeOutputs);
    data = struct('Xtr',XTrain,'Ytr',YTrain,'Xval',XVal,'Yval',YVal, ...
        'Xte',XTest,'Yte',YTest,'Xood',XOod,'Yood',YOod);
    [py, pred, ts, info] = run_python_baseline_adapter(adapterPath, mlpOpts.pythonExe, mlpOpts.workRoot, cfg, data);

    result = struct();
    result.method = 'MLP-KAN-paper-sweep';
    result.protocol = 'kan_feynman_sweep';
    result.seed = mlpOpts.seed;
    result.width = py.selected_width;
    result.depth = py.selected_depth;
    result.hiddenLayerCount = py.selected_hidden_layer_count;
    result.hiddenLayerSizes = repmat(py.selected_width, 1, py.selected_hidden_layer_count);
    result.activation = char(py.selected_activation);
    result.trainFcn = sprintf('pykan-%s', mlpOpts.optimizer);
    result.parameterCount = py.parameter_count;
    result.nActiveCoefficients = py.parameter_count;
    result.structureLabel = sprintf('depth=%d,width=%d,act=%s', result.depth, result.width, result.activation);
    result.selectionMetric = 'validation_mse';
    result.candidates = py.candidates;
    result.candidateCount = py.candidate_count;
    result.trainMetrics = compute_regression_metrics(pred.train, YTrain);
    result.valMetrics = compute_regression_metrics(pred.val, YVal);
    result.testMetrics = compute_regression_metrics(pred.test, YTest);
    if ~isempty(XOod); result.oodMetrics = compute_regression_metrics(pred.ood, YOod); else; result.oodMetrics = empty_metrics_local(); end
    result.YTrainPred = pred.train; result.YValPred = pred.val; result.YTestPred = pred.test; result.YOodPred = pred.ood;
    result.trainTime = py.total_time_seconds; % complete 15-candidate selection cost
    result.selectedModelTrainTime = py.selected_candidate_time_seconds;
    result.timeStats = ts;
    result.timeStats.sweepTime = py.total_time_seconds;
    result.timeStats.selectedModelTrainTime = py.selected_candidate_time_seconds;
    result.workDir = info.workDir; result.configPath = info.configPath; result.resultJsonPath = info.resultJsonPath;
    result.pyResult = py; result.opts = mlpOpts;
end

function m = empty_metrics_local()
    m = struct('mse',NaN,'rmse',NaN,'mae',NaN,'nrmse',NaN,'nrmseRange',NaN,'nmae',NaN);
end
