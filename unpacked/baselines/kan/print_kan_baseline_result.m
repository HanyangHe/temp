function print_kan_baseline_result(result)
%PRINT_KAN_BASELINE_RESULT Print official-pyKAN Feynman pruned-refinement result.
    fprintf('\n========================================\nPruned KAN baseline result\n========================================\n');
    if isfield(result,'dataSource');fprintf('Data source                  : %s\n',result.dataSource);end
    fprintf('Method                       : %s\n',result.method);
    print_normalization_local(result);
    fprintf('Selected depth/width         : %d / %d\n',result.depth,result.width);
    fprintf('Selected sparsification lambda: %.3e\n',result.sparsificationLambda);
    fprintf('Selected validation-best grid: %d\n',result.grid);
    if isfield(result,'lastAttemptedGrid');fprintf('Last attempted grid           : %d\n',result.lastAttemptedGrid);end
    if isfield(result,'gridStopReason')&&~isempty(result.gridStopReason);fprintf('Grid stop reason              : %s\n',result.gridStopReason);end
    fprintf('Initial post-prune shape     : %s\n',format_pykan_shape(result.prunedShape));
    if isfield(result,'finalShape');fprintf('Final refined shape          : %s\n',format_pykan_shape(result.finalShape));end
    fprintf('Active edges / coefficients  : %d / %d\n',round(result.activeEdgeCount),round(result.activeCoefficientCount));
    fprintf('Complete sweep wall time     : %.3f s\n',result.trainTime);
    fprintf('Selected-candidate train time: %.3f s\n',result.selectedModelTrainTime);
    print_prune_diagnostics_local(result);
    if isfield(result,'gridStages')&&~isempty(result.gridStages);print_grid_stages_local(result.gridStages);end
    if isfield(result,'bestByLambda')&&~isempty(result.bestByLambda)
        fprintf('\nValidation-best candidate under each sparsification lambda:\n');
        fprintf('  %10s %6s %6s %13s %12s %s\n','lambda','depth','grid','ValMSE','Active','Pruned shape');
        for i=1:numel(result.bestByLambda)
            q=result.bestByLambda(i);
            fprintf('  %10.3e %6d %6d %13.4e %12d %s\n',q.sparsification_lambda,q.depth,q.selected_grid,q.val_mse,round(q.active_coefficient_count),format_pykan_shape(q.pruned_shape));
        end
    end
    if isfield(result,'candidates')&&isfield(result,'opts')&&result.opts.displaySweepTable;print_sweep_local(result.candidates);end
    fprintf('\nIn-distribution metrics:\n');print_metric_local('Train',result.trainMetrics);print_metric_local('Val  ',result.valMetrics);print_metric_local('Test ',result.testMetrics);
    if isfield(result,'oodMetrics')&&isfinite(result.oodMetrics.mse);fprintf('\nOut-of-distribution metrics:\n');print_metric_local('OOD  ',result.oodMetrics);end
    fprintf('========================================\n');
end

function print_normalization_local(result)
    inFlag=false; outFlag=false;
    if isfield(result,'normalization')&&isstruct(result.normalization)
        if isfield(result.normalization,'enabled_inputs');inFlag=logical(result.normalization.enabled_inputs);end
        if isfield(result.normalization,'enabled_outputs');outFlag=logical(result.normalization.enabled_outputs);end
    elseif isfield(result,'opts')
        if isfield(result.opts,'normalizeInputs');inFlag=logical(result.opts.normalizeInputs);end
        if isfield(result.opts,'normalizeOutputs');outFlag=logical(result.opts.normalizeOutputs);end
    end
    fprintf('Input normalization          : %d (training-split z-score)\n',inFlag);
    fprintf('Output normalization         : %d (training-split z-score)\n',outFlag);
    fprintf('Reported-output scale        : original units (prediction denormalized=%d)\n',outFlag);
end
function print_prune_diagnostics_local(result)
    fprintf('\nSelected-candidate pruning diagnostics:\n');
    if isfield(result,'prePruneValMetrics')&&isstruct(result.prePruneValMetrics)&&isfield(result.prePruneValMetrics,'mse')
        fprintf('  Initial-grid pre-prune Val MSE: %.6e\n',result.prePruneValMetrics.mse);
    end
    if isfield(result,'immediatePostPruneValMetrics')&&isstruct(result.immediatePostPruneValMetrics)&&isfield(result.immediatePostPruneValMetrics,'mse')
        fprintf('  Initial official-prune Val MSE: %.6e\n',result.immediatePostPruneValMetrics.mse);
    end
    if isfield(result,'postRefinementValMetrics')&&isstruct(result.postRefinementValMetrics)&&isfield(result.postRefinementValMetrics,'mse')
        fprintf('  Final post-refinement Val MSE : %.6e\n',result.postRefinementValMetrics.mse);
    end
end
function print_grid_stages_local(stages)
    fprintf('\nSelected-candidate official pyKAN progression:\n');
    fprintf('  %5s %-22s %18s %13s %13s %10s %s\n','Grid','Phase','Status','ValMSE','RelImprove','Selected','Note');
    fprintf('  %s\n',repmat('-',1,112));
    for i=1:numel(stages)
        q=stages(i); vm=get_numeric_local(q,'val_mse',NaN); ri=get_numeric_local(q,'relative_improvement',NaN); accepted=get_logical_local(q,'accepted',false); note=get_char_local(q,'stop_reason','');
        fprintf('  %5d %-22s %18s %13.4e %13.4e %10d %s\n',round(q.grid),get_char_local(q,'phase',''),get_char_local(q,'status',''),vm,ri,accepted,note);
        err=get_char_local(q,'error',''); if ~isempty(err);fprintf('        error: %s\n',err);end
    end
end
function print_sweep_local(c)
    fprintf('\nDepth/lambda sweep (selection uses validation MSE only):\n');
    fprintf('  %5s %10s %6s %13s %12s %10s %-35s\n','Depth','lambda','Grid','ValMSE','Active','Time_s','Status/failure');fprintf('  %s\n',repmat('-',1,101));
    for i=1:numel(c)
        if isfield(c(i),'status')&&strcmpi(char(c(i).status),'ok')
            fprintf('  %5d %10.3e %6d %13.4e %12d %10.3f %-35s\n',c(i).depth,c(i).sparsification_lambda,round(c(i).final_grid),c(i).val_metrics.mse,round(c(i).active_coefficient_count),c(i).time_seconds,get_char_local(c(i),'grid_stop_reason','completed'));
        else
            detail=sprintf('%s: %s',get_char_local(c(i),'failed_stage','unknown'),get_char_local(c(i),'error',''));
            fprintf('  %5d %10.3e %6s %13s %12s %10.3f %-35s\n',c(i).depth,c(i).sparsification_lambda,'-','FAILED','-',c(i).time_seconds,detail);
        end
    end
end
function v=get_numeric_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n))&&isnumeric(s.(n));v=double(s.(n));else;v=d;end;end
function v=get_logical_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=logical(s.(n));else;v=d;end;end
function v=get_char_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=char(string(s.(n)));else;v=d;end;end
function print_metric_local(label,m);fprintf('  %s MSE = %.6e, RMSE = %.6e, NMAE = %.6e\n',label,m.mse,m.rmse,m.nmae);end
