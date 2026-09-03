function result = train_eql_sweep_baseline(XTrain,YTrain,XVal,YVal,XTest,YTest,XOod,YOod,eqlOpts,varargin)
%TRAIN_EQL_SWEEP_BASELINE Run the bundled official EQL-Div Theano sweep.
%
% varargin is accepted only for backward compatibility with v73i callers;
% custom penalty pools are not used because the official upstream core owns
% its penalty-epoch sampling and loss.

    if nargin < 9 || isempty(eqlOpts); eqlOpts=eql_default_options(); end
    if nargin < 7 || isempty(XOod)
        XOod=zeros(0,size(XTrain,2)); YOod=zeros(0,size(YTrain,2));
    end

    eqlDir=fileparts(mfilename('fullpath'));
    adapterPath=fullfile(eqlDir,'eql_official_adapter.py');
    check_eql_official_environment(eqlOpts.pythonExe,eqlOpts.officialRoot);

    cfg=struct();
    cfg.seed=eqlOpts.seed;
    cfg.depth_list=eqlOpts.depthList;
    cfg.lambda_list=eqlOpts.lambdaList;
    cfg.units_per_unary_type=eqlOpts.unitsPerUnaryType;
    cfg.steps_per_hidden_layer=eqlOpts.stepsPerHiddenLayer;
    cfg.batch_size=eqlOpts.batchSize;
    cfg.learning_rate=eqlOpts.learningRate;
    cfg.gradient=eqlOpts.gradient;
    cfg.lambda_l2=eqlOpts.lambdaL2;
    cfg.penalty_every=eqlOpts.penaltyEvery;
    cfg.validate_every=eqlOpts.validateEvery;
    if ~isfield(eqlOpts,'candidateWorkers') || isempty(eqlOpts.candidateWorkers)
        cfg.candidate_workers=0; % auto: resolved by Python from candidate/core counts
    else
        cfg.candidate_workers=eqlOpts.candidateWorkers;
    end
    cfg.official_verbose=logical(eqlOpts.officialVerbose);
    cfg.normalize_inputs=logical(eqlOpts.normalizeInputs);
    cfg.normalize_outputs=logical(eqlOpts.normalizeOutputs);
    cfg.official_root=eqlOpts.officialRoot;
    cfg.theano_flags=eqlOpts.theanoFlags;
    cfg.use_bundled_official_eq11_data=logical(eqlOpts.useBundledOfficialEq11Data);

    data=struct('Xtr',XTrain,'Ytr',YTrain,'Xval',XVal,'Yval',YVal, ...
        'Xte',XTest,'Yte',YTest,'Xood',XOod,'Yood',YOod);
    [py,pred,ts,info]=run_python_baseline_adapter( ...
        adapterPath,eqlOpts.pythonExe,eqlOpts.workRoot,cfg,data);

    result=struct();
    result.method='EQL-official-Theano-sweep';
    result.protocol=char(py.protocol);
    result.seed=eqlOpts.seed;
    result.depth=py.selected_depth;
    result.functionalLayerCount=py.selected_functional_layer_count;
    result.lambda=py.selected_lambda;
    result.selectionScore=py.selected_score;
    result.unitsPerUnaryType=py.units_per_unary_type;
    result.multiplicationUnits=py.multiplication_units;
    result.operatorFamily=eqlOpts.operatorFamily;
    result.parameterCount=py.parameter_count;
    result.activeWeightCount=py.selected_active_weight_count;
    result.activeBiasCount=py.selected_active_bias_count;
    result.nActiveCoefficients=py.selected_active_parameter_count;
    result.nActiveUnits=py.selected_connected_unit_count;
    result.connectedUnitsByLayer=getfield_default_local(py,'selected_connected_units_by_layer',[]);
    result.connectedUnitsByType=getfield_default_local(py,'selected_connected_units_by_type',struct());
    result.connectedActiveWeightCount=py.selected_connected_active_weight_count;
    result.structureLabel=sprintf('official EQL: L=%d,lambda=%.1e,activeUnits=%d', ...
        result.depth,result.lambda,result.nActiveUnits);
    result.selectionMetric=char(py.selection_metric);
    result.trainFcn='Unchanged martius-lab/EQL Theano core';
    result.candidates=py.candidates;
    result.candidateCount=py.candidate_count;
    result.successfulCandidateCount=py.successful_candidate_count;
    result.officialSourceCommit=char(py.source_commit);
    result.officialSourceUnmodified=logical(py.official_source_unmodified);
    result.officialSourceSha256=py.official_source_sha256;
    result.officialSettings=py.official_settings;
    result.normalization=py.normalization;
    result.dataMode=char(py.data_mode);
    result.usesOodLabelsForSelection=logical(py.uses_ood_labels_for_selection);
    result.requestedBatchSize=getfield_default_local(py,'requested_batch_size',eqlOpts.batchSize);
    result.effectiveBatchSize=getfield_default_local(py,'effective_batch_size',result.requestedBatchSize);
    result.batchSizeAdjusted=logical(getfield_default_local(py,'batch_size_adjusted',false));

    % Trust the upstream adapter metrics. This also supports the temporary
    % Eq. (11) demo, which can use the exact bundled official data files.
    result.trainMetrics=py.train_metrics;
    result.valMetrics=py.val_metrics;
    result.testMetrics=py.test_metrics;
    result.oodMetrics=py.ood_metrics;
    result.YTrainPred=pred.train; result.YValPred=pred.val;
    result.YTestPred=pred.test; result.YOodPred=pred.ood;

    result.trainTime=py.total_time_seconds;
    result.selectedModelTrainTime=py.selected_candidate_time_seconds;
    result.timeStats=ts;
    result.timeStats.sweepTime=py.total_time_seconds;
    result.timeStats.selectedModelTrainTime=py.selected_candidate_time_seconds;
    result.workDir=info.workDir;
    result.configPath=info.configPath;
    result.resultJsonPath=info.resultJsonPath;
    result.pyResult=py;
    result.opts=eqlOpts;
end

function v=getfield_default_local(s,n,d)
    if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=s.(n);else;v=d;end
end
