function result=train_pruned_kan_sweep_baseline(XTrain,YTrain,XVal,YVal,XTest,YTest,XOod,YOod,kanOpts)
%TRAIN_PRUNED_KAN_SWEEP_BASELINE Run official-pyKAN Feynman pruned refinement sweep.
    if nargin<9||isempty(kanOpts);kanOpts=kan_default_options();end
    if nargin<7||isempty(XOod);XOod=zeros(0,size(XTrain,2));YOod=zeros(0,size(YTrain,2));end
    kanDir=fileparts(mfilename('fullpath')); adapterPath=fullfile(kanDir,'kan_paper_pruned_adapter.py');
    cfg=struct(); cfg.pykan_root=kanOpts.pykanRoot; cfg.seed=kanOpts.seed; cfg.width=kanOpts.width;
    cfg.depth_list=kanOpts.depthList; cfg.grid_list=kanOpts.gridList; cfg.spline_order=kanOpts.splineOrder;
    cfg.sparsification_lambda_list=kanOpts.sparsificationLambdaList; cfg.steps_per_grid=kanOpts.stepsPerGrid;
    cfg.optimizer=kanOpts.optimizer; cfg.learning_rate=kanOpts.learningRate;
    cfg.prune_node_threshold=kanOpts.pruneNodeThreshold; cfg.prune_edge_threshold=kanOpts.pruneEdgeThreshold;
    cfg.grid_early_stop=logical(getfield_default_local(kanOpts,'gridEarlyStop',true));
    cfg.grid_early_stop_patience=getfield_default_local(kanOpts,'gridEarlyStopPatience',1);
    cfg.grid_early_stop_relative_tolerance=getfield_default_local(kanOpts,'gridEarlyStopRelativeTolerance',0.0);
    cfg.dtype=kanOpts.dtype; cfg.device=kanOpts.device; cfg.torch_num_threads=kanOpts.torchNumThreads;
    cfg.normalize_inputs=logical(kanOpts.normalizeInputs); cfg.normalize_outputs=logical(kanOpts.normalizeOutputs);
    data=struct('Xtr',XTrain,'Ytr',YTrain,'Xval',XVal,'Yval',YVal,'Xte',XTest,'Yte',YTest,'Xood',XOod,'Yood',YOod);
    [py,pred,ts,info]=run_python_baseline_adapter(adapterPath,kanOpts.pythonExe,kanOpts.workRoot,cfg,data);
    result=struct(); result.method='Official-pyKAN-Feynman-pruned-refinement-validation-early-stop-sweep'; result.protocol='official_pykan_feynman_pruned_refinement_validation_early_stop'; result.seed=kanOpts.seed;
    result.width=py.selected_width; result.depth=py.selected_depth; result.hiddenLayerCount=py.selected_hidden_layer_count;
    result.sparsificationLambda=py.selected_sparsification_lambda; result.grid=py.selected_grid;
    result.gridStopReason=getfield_default_local(py,'selected_grid_stop_reason','');
    result.lastAttemptedGrid=getfield_default_local(py,'selected_last_attempted_grid',result.grid);
    result.gridStages=getfield_default_local(py,'selected_grid_stages',struct([]));
    result.initialShape=py.selected_initial_shape; result.prunedShape=py.selected_pruned_shape;
    result.finalShape=getfield_default_local(py,'selected_final_shape',py.selected_pruned_shape);
    result.activeEdgeCount=py.active_edge_count; result.activeCoefficientCount=py.active_coefficient_count;
    result.trainableParameterCount=py.trainable_parameter_count; result.nActiveCoefficients=py.active_coefficient_count;
    result.prePruneTrainMetrics=getfield_default_local(py,'selected_pre_prune_train_metrics',struct());
    result.prePruneValMetrics=getfield_default_local(py,'selected_pre_prune_val_metrics',struct());
    result.immediatePostPruneTrainMetrics=getfield_default_local(py,'selected_immediate_post_prune_train_metrics',struct());
    result.immediatePostPruneValMetrics=getfield_default_local(py,'selected_immediate_post_prune_val_metrics',struct());
    result.postRefinementTrainMetrics=getfield_default_local(py,'selected_post_refit_train_metrics',struct());
    result.postRefinementValMetrics=getfield_default_local(py,'selected_post_refit_val_metrics',struct());
    result.structureLabel=sprintf('depth=%d,width=%d,lambda=%.0e,grid=%d,shape=%s',result.depth,result.width,result.sparsificationLambda,result.grid,format_pykan_shape(result.prunedShape));
    result.selectionMetric='validation_mse'; result.candidates=py.candidates; result.candidateCount=py.candidate_count;
    result.bestByLambda=getfield_default_local(py,'best_by_lambda',struct([]));
    result.normalization=getfield_default_local(py,'normalization',struct());
    result.trainMetrics=compute_regression_metrics(pred.train,YTrain); result.valMetrics=compute_regression_metrics(pred.val,YVal);
    result.testMetrics=compute_regression_metrics(pred.test,YTest);
    if ~isempty(XOod);result.oodMetrics=compute_regression_metrics(pred.ood,YOod);else;result.oodMetrics=empty_metrics_local();end
    result.YTrainPred=pred.train;result.YValPred=pred.val;result.YTestPred=pred.test;result.YOodPred=pred.ood;
    result.trainTime=py.total_time_seconds; result.selectedModelTrainTime=py.selected_candidate_time_seconds;
    result.timeStats=ts;result.timeStats.sweepTime=py.total_time_seconds;result.timeStats.selectedModelTrainTime=py.selected_candidate_time_seconds;
    result.workDir=info.workDir;result.configPath=info.configPath;result.resultJsonPath=info.resultJsonPath;result.pyResult=py;result.opts=kanOpts;
end
function m=empty_metrics_local();m=struct('mse',NaN,'rmse',NaN,'mae',NaN,'nrmse',NaN,'nrmseRange',NaN,'nmae',NaN);end
function v=getfield_default_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=s.(n);else;v=d;end;end
