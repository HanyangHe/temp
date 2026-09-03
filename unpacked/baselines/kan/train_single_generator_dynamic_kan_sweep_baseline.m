function result=train_single_generator_dynamic_kan_sweep_baseline(XTrain,YTrain,XVal,YVal,XTest,YTest,XOod,YOod,kanOpts)
%TRAIN_SINGLE_GENERATOR_DYNAMIC_KAN_SWEEP_BASELINE Accuracy-first SI pyKAN sweep.
% The system-identification route uses validation-only model selection,
% minimum-grid inheritance across nested sample sizes, optional native-model
% warm start, train-cache-safe prune/refine calls, and pruning fallback that
% still completes zero-lambda recovery plus the remaining grid schedule.
    if nargin<9||isempty(kanOpts);kanOpts=kan_default_options();end
    if nargin<7||isempty(XOod);XOod=zeros(0,size(XTrain,2));YOod=zeros(0,size(YTrain,2));end
    kanDir=fileparts(mfilename('fullpath'));
    adapterPath=fullfile(kanDir,'kan_paper_pruned_adapter_single_generator.py');

    cfg=struct();
    cfg.pykan_root=kanOpts.pykanRoot;
    cfg.seed=kanOpts.seed;
    cfg.width=kanOpts.width;
    cfg.depth_list=kanOpts.depthList;
    cfg.minimum_depth=getfield_default_local(kanOpts,'minimumDepth',min(kanOpts.depthList));
    cfg.grid_list=kanOpts.gridList;
    cfg.minimum_grid=getfield_default_local(kanOpts,'minimumGrid',min(kanOpts.gridList));
    cfg.spline_order=kanOpts.splineOrder;
    cfg.sparsification_lambda_list=kanOpts.sparsificationLambdaList;
    cfg.steps_per_grid=kanOpts.stepsPerGrid;
    cfg.accuracy_steps_per_grid=getfield_default_local(kanOpts,'accuracyStepsPerGrid',kanOpts.stepsPerGrid);
    cfg.sparsification_steps=getfield_default_local(kanOpts,'sparsificationSteps',kanOpts.stepsPerGrid);
    cfg.recovery_steps_per_grid=getfield_default_local(kanOpts,'recoveryStepsPerGrid',kanOpts.stepsPerGrid);
    cfg.optimizer=kanOpts.optimizer;
    cfg.learning_rate=kanOpts.learningRate;
    cfg.prune_node_threshold=kanOpts.pruneNodeThreshold;
    cfg.prune_edge_threshold=kanOpts.pruneEdgeThreshold;
    cfg.prune_validation_guard_enable=logical(getfield_default_local(kanOpts,'pruneValidationGuardEnable',true));
    cfg.prune_max_relative_validation_increase=getfield_default_local(kanOpts,'pruneMaxRelativeValidationIncrease',0.0);
    cfg.grid_early_stop=logical(getfield_default_local(kanOpts,'gridEarlyStop',true));
    cfg.grid_early_stop_patience=getfield_default_local(kanOpts,'gridEarlyStopPatience',2);
    cfg.grid_early_stop_relative_tolerance=getfield_default_local(kanOpts,'gridEarlyStopRelativeTolerance',0.015);
    cfg.depth_early_stop=logical(getfield_default_local(kanOpts,'depthEarlyStop',true));
    cfg.depth_early_stop_patience=getfield_default_local(kanOpts,'depthEarlyStopPatience',2);
    cfg.depth_early_stop_relative_tolerance=getfield_default_local(kanOpts,'depthEarlyStopRelativeTolerance',0.0);
    cfg.warm_start_enable=logical(getfield_default_local(kanOpts,'warmStartEnable',true));
    cfg.warm_start_checkpoint_path=getfield_default_local(kanOpts,'warmStartCheckpointPath','');
    fixedNorm=getfield_default_local(kanOpts,'warmStartNormalization',struct());
    if isstruct(fixedNorm)&&~isempty(fieldnames(fixedNorm))
        cfg.fixed_normalization=normalize_normalization_struct_local(fixedNorm);
    end
    cfg.dtype=kanOpts.dtype;
    cfg.device=kanOpts.device;
    cfg.torch_num_threads=kanOpts.torchNumThreads;
    cfg.normalize_inputs=logical(kanOpts.normalizeInputs);
    cfg.normalize_outputs=logical(kanOpts.normalizeOutputs);

    data=struct('Xtr',XTrain,'Ytr',YTrain,'Xval',XVal,'Yval',YVal, ...
        'Xte',XTest,'Yte',YTest,'Xood',XOod,'Yood',YOod);
    [py,pred,ts,info]=run_python_baseline_adapter( ...
        adapterPath,kanOpts.pythonExe,kanOpts.workRoot,cfg,data);

    result=struct();
    result.method=getfield_default_local(py,'method','Official-pyKAN-SI-accuracy-first-grid-inheritance');
    result.protocol=getfield_default_local(py,'protocol','official_pykan_si_accuracy_first_prune_guard_grid_inheritance');
    result.seed=kanOpts.seed;
    result.width=py.selected_width;
    result.depth=py.selected_depth;
    result.hiddenLayerCount=py.selected_hidden_layer_count;
    result.sparsificationLambda=py.selected_sparsification_lambda;
    result.grid=py.selected_grid;
    result.minimumGrid=getfield_default_local(py,'minimum_grid',cfg.minimum_grid);
    result.eligibleGridList=getfield_default_local(py,'eligible_grid_list',kanOpts.gridList);
    result.gridEarlyStopPatience=getfield_default_local(py,'grid_early_stop_patience',cfg.grid_early_stop_patience);
    result.gridEarlyStopRelativeTolerance=getfield_default_local(py,'grid_early_stop_relative_tolerance',cfg.grid_early_stop_relative_tolerance);
    result.gridStopReason=getfield_default_local(py,'selected_grid_stop_reason','');
    result.lastAttemptedGrid=getfield_default_local(py,'selected_last_attempted_grid',result.grid);
    result.gridStages=getfield_default_local(py,'selected_grid_stages',struct([]));
    result.initialShape=py.selected_initial_shape;
    result.prunedShape=py.selected_pruned_shape;
    result.finalShape=getfield_default_local(py,'selected_final_shape',py.selected_pruned_shape);
    result.activeEdgeCount=py.active_edge_count;
    result.activeCoefficientCount=py.active_coefficient_count;
    result.trainableParameterCount=py.trainable_parameter_count;
    result.nActiveCoefficients=py.active_coefficient_count;
    result.pruningAccepted=logical(getfield_default_local(py,'selected_pruning_accepted',false));
    result.pruningGuardReason=getfield_default_local(py,'selected_pruning_guard_reason','');
    result.selectedStructureSource=getfield_default_local(py,'selected_structure_source','accuracy_unpruned');
    result.selectedWarmStart=logical(getfield_default_local(py,'selected_warm_start',false));
    result.pruneValidationGuardEnable=logical(getfield_default_local(py,'prune_validation_guard_enable',cfg.prune_validation_guard_enable));
    result.pruneMaxRelativeValidationIncrease=getfield_default_local(py,'prune_max_relative_validation_increase',cfg.prune_max_relative_validation_increase);
    result.prePruneTrainMetrics=getfield_default_local(py,'selected_pre_prune_train_metrics',struct());
    result.prePruneValMetrics=getfield_default_local(py,'selected_pre_prune_val_metrics',struct());
    result.immediatePostPruneTrainMetrics=getfield_default_local(py,'selected_immediate_post_prune_train_metrics',struct());
    result.immediatePostPruneValMetrics=getfield_default_local(py,'selected_immediate_post_prune_val_metrics',struct());
    result.postRefinementTrainMetrics=getfield_default_local(py,'selected_post_refit_train_metrics',struct());
    result.postRefinementValMetrics=getfield_default_local(py,'selected_post_refit_val_metrics',struct());
    result.structureLabel=sprintf( ...
        'depth=%d,width=%d,lambda=%.0e,grid=%d,shape=%s,source=%s', ...
        result.depth,result.width,result.sparsificationLambda,result.grid, ...
        format_pykan_shape(result.finalShape),result.selectedStructureSource);
    result.selectionMetric='validation_mse';
    result.candidates=py.candidates;
    result.candidateCount=py.candidate_count;
    result.configuredCandidateCount=getfield_default_local(py,'configured_candidate_count',py.candidate_count);
    result.depthEarlyStop=logical(getfield_default_local(py,'depth_early_stop',false));
    result.depthStopRecords=getfield_default_local(py,'depth_stop_records',struct([]));
    result.bestByLambda=getfield_default_local(py,'best_by_lambda',struct([]));
    result.normalization=getfield_default_local(py,'normalization',struct());
    result.minimumDepth=getfield_default_local(py,'minimum_depth',cfg.minimum_depth);
    result.eligibleDepthList=getfield_default_local(py,'eligible_depth_list',kanOpts.depthList);
    result.portableModel=getfield_default_local(py,'portable_model',struct());
    result.nativeCheckpointPath=getfield_default_local(py,'native_checkpoint_path','');
    result.warmStartCheckpointRequested=getfield_default_local(py,'warm_start_checkpoint_requested','');
    result.warmStartNormalizationInherited=logical(getfield_default_local(py,'warm_start_normalization_inherited',false));
    result.settings=getfield_default_local(py,'settings',struct());

    result.trainMetrics=compute_regression_metrics(pred.train,YTrain);
    result.valMetrics=compute_regression_metrics(pred.val,YVal);
    result.testMetrics=compute_regression_metrics(pred.test,YTest);
    if ~isempty(XOod)
        result.oodMetrics=compute_regression_metrics(pred.ood,YOod);
    else
        result.oodMetrics=empty_metrics_local();
    end
    result.YTrainPred=pred.train;
    result.YValPred=pred.val;
    result.YTestPred=pred.test;
    result.YOodPred=pred.ood;
    result.trainTime=py.total_time_seconds;
    result.selectedModelTrainTime=py.selected_candidate_time_seconds;
    result.timeStats=ts;
    result.timeStats.sweepTime=py.total_time_seconds;
    result.timeStats.selectedModelTrainTime=py.selected_candidate_time_seconds;
    result.workDir=info.workDir;
    result.configPath=info.configPath;
    result.resultJsonPath=info.resultJsonPath;
    result.pyResult=py;
    result.opts=kanOpts;

    if ~isempty(fieldnames(result.portableModel))
        portableTest=predict_single_generator_dynamic_kan_baseline(result,XTest);
        result.portablePredictorMaxAbsTestDifference=max(abs(portableTest(:)-pred.test(:)));
        scale=max(1,abs(pred.test));
        result.portablePredictorMaxRelTestDifference=max(abs(portableTest(:)-pred.test(:))./scale(:));
        if ~isfinite(result.portablePredictorMaxAbsTestDifference) || ...
                (result.portablePredictorMaxAbsTestDifference>5e-5 && ...
                 result.portablePredictorMaxRelTestDifference>5e-5)
            error('Portable KAN predictor mismatch: max abs/rel %.6e / %.6e.', ...
                result.portablePredictorMaxAbsTestDifference, ...
                result.portablePredictorMaxRelTestDifference);
        end
    end
end

function n=normalize_normalization_struct_local(n)
% Convert jsondecoded portable tensor forms, when present, to row vectors.
    names={'x_mean','x_std','y_mean','y_std'};
    for i=1:numel(names)
        name=names{i};
        if ~isfield(n,name);continue;end
        value=n.(name);
        if isstruct(value)&&isfield(value,'data')
            value=value.data;
        end
        n.(name)=reshape(double(value),1,[]);
    end
end
function m=empty_metrics_local();m=struct('mse',NaN,'rmse',NaN,'mae',NaN,'nrmse',NaN,'nrmseRange',NaN,'nmae',NaN);end
function v=getfield_default_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=s.(n);else;v=d;end;end
