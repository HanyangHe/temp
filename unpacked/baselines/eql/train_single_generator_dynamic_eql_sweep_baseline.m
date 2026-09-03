function result = train_single_generator_dynamic_eql_sweep_baseline(XTrain,YTrain,XVal,YVal,XTest,YTest,XOod,YOod,eqlOpts,varargin)
%TRAIN_SINGLE_GENERATOR_DYNAMIC_EQL_SWEEP_BASELINE Run the bundled official EQL-Div Theano sweep.
%
% varargin is accepted only for backward compatibility with v73i callers;
% custom penalty pools are not used because the official upstream core owns
% its penalty-epoch sampling and loss.

    if nargin < 9 || isempty(eqlOpts); eqlOpts=eql_default_options(); end
    if nargin < 7 || isempty(XOod)
        XOod=zeros(0,size(XTrain,2)); YOod=zeros(0,size(YTrain,2));
    end

    eqlDir=fileparts(mfilename('fullpath'));
    adapterPath=fullfile(eqlDir,'eql_official_adapter_single_generator.py');
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
    cfg.depth_early_stop=logical(getfield_default_local(eqlOpts,'depthEarlyStop',false));
    cfg.depth_early_stop_patience=getfield_default_local(eqlOpts,'depthEarlyStopPatience',1);
    cfg.depth_early_stop_relative_tolerance=getfield_default_local(eqlOpts,'depthEarlyStopRelativeTolerance',0.0);
    % SI policy: every N starts from the complete configured depth schedule.
    % Do not inherit a larger minimumDepth from a previous sample size.
    cfg.minimum_depth=min(eqlOpts.depthList);
    cfg.checkpoint_selection_mode='physical_validation_mse';
    % Optional cross-sample metadata is absent for the first N. MATLAB
    % jsonencode maps NaN to JSON null, which older adapters attempted to
    % convert with float(None). Omit unavailable optional numeric fields; the
    % Python adapter also accepts null defensively for archived callers.
    previousValidationMSE=getfield_default_local(eqlOpts,'previousValidationMSE',NaN);
    if is_finite_numeric_scalar_local(previousValidationMSE)
        cfg.previous_validation_mse=double(previousValidationMSE);
    end
    cfg.previous_selected_state_path=getfield_default_local(eqlOpts,'previousSelectedStatePath','');
    cfg.previous_selected_depth=getfield_default_local(eqlOpts,'previousSelectedDepth',0);
    previousSelectedLambda=getfield_default_local(eqlOpts,'previousSelectedLambda',NaN);
    if is_finite_numeric_scalar_local(previousSelectedLambda)
        cfg.previous_selected_lambda=double(previousSelectedLambda);
    end
    cfg.warm_start_previous_model=logical(getfield_default_local(eqlOpts, ...
        'warmStartPreviousModel',true));
    cfg.warm_start_restarts=getfield_default_local(eqlOpts,'warmStartRestarts',1);
    cfg.adaptive_rescue_restarts=getfield_default_local(eqlOpts,'adaptiveRescueRestarts',3);
    cfg.adaptive_rescue_top_k=getfield_default_local(eqlOpts,'adaptiveRescueTopK',2);
    cfg.strict_improvement_relative_margin=getfield_default_local(eqlOpts, ...
        'strictImprovementRelativeMargin',0.0);
    cfg.strict_improvement_absolute_margin=getfield_default_local(eqlOpts, ...
        'strictImprovementAbsoluteMargin',0.0);
    cfg.strict_target_overrides_depth_early_stop=logical(getfield_default_local(eqlOpts, ...
        'strictTargetOverridesDepthEarlyStop',true));
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
    result.reportedMSEScale=char(getfield_default_local(py,'reported_mse_scale', ...
        'original_physical_output_units'));
    result.selectedCheckpoint=char(getfield_default_local(py,'selected_checkpoint','final_state'));
    result.selectedCheckpointEpoch=getfield_default_local(py,'selected_checkpoint_epoch',NaN);
    result.selectedCheckpointPhase=char(getfield_default_local(py,'selected_checkpoint_phase','unknown'));
    result.bestStateEpoch=getfield_default_local(py,'selected_best_state_epoch',NaN);
    result.bestStatePhase=char(getfield_default_local(py,'selected_best_state_phase','unknown'));
    result.upstreamBestValidationEpoch=getfield_default_local(py, ...
        'selected_upstream_best_validation_epoch',result.bestStateEpoch);
    result.finalStateEpoch=getfield_default_local(py,'selected_final_state_epoch',NaN);
    result.finalStatePhase=char(getfield_default_local(py,'selected_final_state_phase','unknown'));
    result.selectedCandidateSource=char(getfield_default_local(py,'selected_candidate_source','scratch_full_sweep'));
    result.bestScratchCurrentNValMSE=getfield_default_local(py, ...
        'best_scratch_current_N_validation_mse',NaN);
    result.bestWarmStartCurrentNValMSE=getfield_default_local(py, ...
        'best_warm_start_current_N_validation_mse',NaN);
    result.selectedRestartIndex=getfield_default_local(py,'selected_restart_index',0);
    result.selectedWarmStartUsed=logical(getfield_default_local(py,'selected_warm_start_used',false));
    result.selectedStatePath=char(getfield_default_local(py,'selected_state_path',''));
    result.checkpointSelectionMetric=char(getfield_default_local(py, ...
        'checkpoint_selection_metric','external_validation_mse_physical_scale'));
    result.bestStateValMSE=getfield_default_local(py,'selected_checkpoint_best_state_val_mse',NaN);
    result.finalStateValMSE=getfield_default_local(py,'selected_checkpoint_final_state_val_mse',NaN);
    result.trainFcn='Unchanged martius-lab/EQL Theano core';
    result.candidates=py.candidates;
    result.candidateCount=py.candidate_count;
    result.configuredCandidateCount=getfield_default_local(py,'configured_candidate_count',py.candidate_count);
    result.baseConfiguredCandidateCount=getfield_default_local(py,'base_configured_candidate_count',result.configuredCandidateCount);
    result.attemptedCandidateCount=getfield_default_local(py,'attempted_candidate_count',py.candidate_count);
    result.paperSampleEfficiencyProtocol=char(getfield_default_local(py, ...
        'paper_sample_efficiency_protocol','exact_N_current_sample_training_with_adaptive_validation_search'));
    result.previousModelRole=char(getfield_default_local(py,'previous_model_role', ...
        'validation_target_and_optional_warm_start_only_never_unchanged_current_N_substitution'));
    result.previousValidationReferenceMSE=getfield_default_local(py,'previous_validation_reference_mse',NaN);
    result.strictCurrentNValidationTargetMSE=getfield_default_local(py,'strict_current_N_validation_target_mse',NaN);
    result.strictCurrentNImprovementAchieved=getfield_default_local(py,'strict_current_N_improvement_achieved',[]);
    result.strictImprovementRelativeMargin=getfield_default_local(py, ...
        'strict_improvement_relative_margin',0.0);
    result.strictImprovementAbsoluteMargin=getfield_default_local(py, ...
        'strict_improvement_absolute_margin',0.0);
    result.adaptiveRescueAttempted=logical(getfield_default_local(py,'adaptive_rescue_attempted',false));
    result.adaptiveRescueRestartsConfigured=getfield_default_local(py, ...
        'adaptive_rescue_restarts_configured',0);
    result.adaptiveRescueRoundsCompleted=getfield_default_local(py,'adaptive_rescue_rounds_completed',0);
    result.adaptiveRescueCandidateCount=getfield_default_local(py,'adaptive_rescue_candidate_count',0);
    result.depthEarlyStop=logical(getfield_default_local(py,'depth_early_stop',false));
    result.depthStopReason=char(getfield_default_local(py,'depth_stop_reason',''));
    result.lastAttemptedDepth=getfield_default_local(py,'last_attempted_depth',result.depth);
    result.depthStages=getfield_default_local(py,'depth_stages',struct([]));
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
    result.minimumDepth=getfield_default_local(py,'minimum_depth',cfg.minimum_depth);
    result.fullDepthScheduleEachSample=logical(getfield_default_local(py, ...
        'full_depth_schedule_each_sample',true));
    result.eligibleDepthList=getfield_default_local(py,'eligible_depth_list',eqlOpts.depthList);
    result.portableModel=getfield_default_local(py,'portable_model',struct());

    % Preserve adapter diagnostics, but compute every public metric through the
    % same MATLAB routine used by the other baselines. In particular, NMAE uses
    % a fixed per-output scale derived from the corresponding true evaluation
    % pool. Therefore an unchanged model evaluated on the same fixed Val/Test/OOD
    % pool must produce exactly the same NMAE at every training-sample size.
    result.adapterMetrics=struct('train',py.train_metrics,'val',py.val_metrics, ...
        'test',py.test_metrics,'ood',py.ood_metrics);
    result.YTrainPred=pred.train; result.YValPred=pred.val;
    result.YTestPred=pred.test; result.YOodPred=pred.ood;
    result.trainMetrics=compute_regression_metrics(result.YTrainPred,YTrain);
    result.valMetrics=compute_regression_metrics(result.YValPred,YVal);
    result.testMetrics=compute_regression_metrics(result.YTestPred,YTest);
    if isempty(XOod)
        result.oodMetrics=empty_metrics_local(size(YTrain,2));
    else
        result.oodMetrics=compute_regression_metrics(result.YOodPred,YOod);
    end
    result.normalizedMetricPolicy='common_framework_per_output_true_pool_scale';
    result.selectionScore=result.valMetrics.mse;

    result.trainTime=py.total_time_seconds;
    result.selectedModelTrainTime=py.selected_candidate_time_seconds;
    result.timeStats=ts;
    result.timeStats.sweepTime=py.total_time_seconds;
    result.timeStats.selectedModelTrainTime=py.selected_candidate_time_seconds;
    result.workDir=info.workDir;
    result.configPath=info.configPath;
    result.resultJsonPath=info.resultJsonPath;
    result.pyResult=py;
    storedOpts=eqlOpts;
    if isfield(storedOpts,'previousResult');storedOpts=rmfield(storedOpts,'previousResult');end
    result.opts=storedOpts;
    if ~isempty(fieldnames(result.portableModel))
        portableTest=predict_single_generator_dynamic_eql_baseline(result,XTest);
        result.portablePredictorMaxAbsTestDifference=max(abs(portableTest(:)-pred.test(:)));
        scale=max(1,abs(pred.test));
        result.portablePredictorMaxRelTestDifference=max(abs(portableTest(:)-pred.test(:))./scale(:));
        if ~isfinite(result.portablePredictorMaxAbsTestDifference) || ...
                (result.portablePredictorMaxAbsTestDifference>1e-8 && ...
                 result.portablePredictorMaxRelTestDifference>1e-8)
            error('Portable EQL predictor mismatch: max abs/rel %.6e / %.6e.', ...
                result.portablePredictorMaxAbsTestDifference, ...
                result.portablePredictorMaxRelTestDifference);
        end
    end
end

function m=empty_metrics_local(ny)
    m=struct('mse',NaN,'rmse',NaN,'mae',NaN,'maxAbsError',NaN, ...
        'mseByOutput',NaN(1,ny),'rmseByOutput',NaN(1,ny), ...
        'maeByOutput',NaN(1,ny),'maxAbsErrorByOutput',NaN(1,ny), ...
        'scaleByOutput',NaN(1,ny),'nrmseByOutput',NaN(1,ny), ...
        'nmaeByOutput',NaN(1,ny),'nmaxAbsErrorByOutput',NaN(1,ny), ...
        'nrmse',NaN,'nmae',NaN,'nmaxAbsError',NaN, ...
        'globalScale',NaN,'globalNRMSE',NaN,'globalNMAE',NaN);
end

function tf=is_finite_numeric_scalar_local(v)
    tf=isnumeric(v)&&isscalar(v)&&isreal(v)&&isfinite(v);
end

function v=getfield_default_local(s,n,d)
    if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=s.(n);else;v=d;end
end
