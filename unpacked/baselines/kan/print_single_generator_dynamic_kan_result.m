function print_single_generator_dynamic_kan_result(result)
%PRINT_SINGLE_GENERATOR_DYNAMIC_KAN_RESULT Print accuracy-first SI KAN result.
    fprintf('\n========================================\n');
    fprintf('KAN baseline result (accuracy-first + grid inheritance)\n');
    fprintf('========================================\n');
    if isfield(result,'dataSource');fprintf('Data source                  : %s\n',result.dataSource);end
    fprintf('Method                       : %s\n',result.method);
    print_normalization_local(result);
    fprintf('Selected depth/width         : %d / %d\n',result.depth,result.width);
    if isfield(result,'minimumDepth');fprintf('Applied minimum depth        : %d\n',round(result.minimumDepth));end
    if isfield(result,'minimumGrid');fprintf('Applied minimum grid G       : %d\n',round(result.minimumGrid));end
    fprintf('Selected sparsification lambda: %.3e\n',result.sparsificationLambda);
    fprintf('Selected validation-best grid: %d\n',result.grid);
    lastAttemptedGrid = resolve_last_attempted_grid_local(result);
    if isfinite(lastAttemptedGrid)
        fprintf('Last attempted grid           : %d\n',round(lastAttemptedGrid));
    end
    if isfield(result,'gridEarlyStopPatience')
        fprintf('Grid early-stop patience      : %d\n',round(result.gridEarlyStopPatience));
    end
    if isfield(result,'gridEarlyStopRelativeTolerance')
        fprintf('Grid rise tolerance           : %.2f %%\n',100*result.gridEarlyStopRelativeTolerance);
    end
    gridStopReason = resolve_grid_stop_reason_local(result);
    if ~isempty(gridStopReason)
        fprintf('Grid stop reason              : %s\n',gridStopReason);
    end
    if isfield(result,'selectedStructureSource')
        fprintf('Selected structure source    : %s\n',result.selectedStructureSource);
    end
    if isfield(result,'selectedWarmStart')
        fprintf('Selected from warm start     : %d\n',logical(result.selectedWarmStart));
    end
    if isfield(result,'warmStartNormalizationInherited')
        fprintf('Inherited train normalization: %d\n',logical(result.warmStartNormalizationInherited));
    end
    if isfield(result,'pruningAccepted');fprintf('Pruning accepted             : %d\n',logical(result.pruningAccepted));end
    if isfield(result,'pruningGuardReason')&&~isempty(result.pruningGuardReason)
        fprintf('Pruning guard decision       : %s\n',result.pruningGuardReason);
    end
    if isfield(result,'prunedShape')&&~isempty(result.prunedShape)
        fprintf('Post-prune shape             : %s\n',format_pykan_shape(result.prunedShape));
    end
    if isfield(result,'finalShape');fprintf('Final selected shape         : %s\n',format_pykan_shape(result.finalShape));end
    fprintf('Active edges / coefficients  : %d / %d\n',round(result.activeEdgeCount),round(result.activeCoefficientCount));
    fprintf('Complete sweep wall time     : %.3f s\n',result.trainTime);
    fprintf('Selected-candidate train time: %.3f s\n',result.selectedModelTrainTime);
    if isfield(result,'portablePredictorMaxAbsTestDifference')
        fprintf('Portable predictor max abs diff: %.3e\n',result.portablePredictorMaxAbsTestDifference);
    end
    if isfield(result,'configuredCandidateCount')
        fprintf('Configured/recorded candidates: %d / %d\n',round(result.configuredCandidateCount),round(result.candidateCount));
    end
    if isfield(result,'depthEarlyStop');fprintf('Validation depth early stop  : %d\n',logical(result.depthEarlyStop));end
    if isfield(result,'opts')&&isfield(result.opts,'depthEarlyStopPatience')
        fprintf('Depth lookahead layers       : %d\n',round(result.opts.depthEarlyStopPatience));
    end
    if isfield(result,'nativeCheckpointPath')&&~isempty(result.nativeCheckpointPath)
        fprintf('Native warm-start checkpoint : %s\n',result.nativeCheckpointPath);
    end

    if isfield(result,'depthStopRecords')&&~isempty(result.depthStopRecords)
        print_depth_stop_records_local(result.depthStopRecords);
    end
    print_prune_diagnostics_local(result);
    if isfield(result,'gridStages')&&~isempty(result.gridStages)
        print_grid_stages_local(result.gridStages);
    end
    if isfield(result,'bestByLambda')&&~isempty(result.bestByLambda)
        fprintf('\nValidation-best recorded candidate under each selected lambda:\n');
        fprintf('  %10s %6s %6s %13s %12s %-34s %s\n', ...
            'lambda','depth','grid','ValMSE','Active','Source','Final shape');
        for i=1:numel(result.bestByLambda)
            q=result.bestByLambda(i);
            fprintf('  %10.3e %6d %6d %13.4e %12d %-34s %s\n', ...
                q.sparsification_lambda,q.depth,q.selected_grid,q.val_mse, ...
                round(q.active_coefficient_count), ...
                get_char_local(q,'selected_structure_source',''), ...
                format_pykan_shape(q.pruned_shape));
        end
    end
    if isfield(result,'candidates')&&isfield(result,'opts')&&result.opts.displaySweepTable
        print_sweep_local(result.candidates);
    end

    fprintf('\nIn-distribution metrics:\n');
    print_metric_local('Train',result.trainMetrics);
    print_metric_local('Val  ',result.valMetrics);
    print_metric_local('Test ',result.testMetrics);
    if isfield(result,'oodMetrics')&&isfinite(result.oodMetrics.mse)
        fprintf('\nOut-of-distribution metrics:\n');
        print_metric_local('OOD  ',result.oodMetrics);
    end
    fprintf('========================================\n');
end

function lastGrid = resolve_last_attempted_grid_local(result)
%RESOLVE_LAST_ATTEMPTED_GRID_LOCAL
% The selected-candidate grid progression is authoritative. Older saved KAN
% records may contain lastAttemptedGrid=selectedGrid even though gridStages
% shows that larger grids were subsequently attempted and rejected.

    lastGrid = NaN;
    if isfield(result,'gridStages') && isstruct(result.gridStages) && ...
            ~isempty(result.gridStages)
        stages = result.gridStages;
        for iStage = numel(stages):-1:1
            candidateGrid = get_numeric_local(stages(iStage),'grid',NaN);
            if isfinite(candidateGrid)
                lastGrid = candidateGrid;
                return;
            end
        end
    end

    if isfield(result,'lastAttemptedGrid') && ...
            isnumeric(result.lastAttemptedGrid) && ...
            isscalar(result.lastAttemptedGrid) && ...
            isfinite(result.lastAttemptedGrid)
        lastGrid = double(result.lastAttemptedGrid);
    end
end

function stopReason = resolve_grid_stop_reason_local(result)
%RESOLVE_GRID_STOP_REASON_LOCAL
% Prefer the terminal selected-candidate progression record. This also fixes
% replay reports from older MAT files whose headline reason was stored as the
% generic "accuracy_validation_best_checkpoint".

    stopReason = '';
    if isfield(result,'gridStages') && isstruct(result.gridStages) && ...
            ~isempty(result.gridStages)
        stages = result.gridStages;
        for iStage = numel(stages):-1:1
            candidateReason = get_char_local(stages(iStage),'stop_reason','');
            if ~isempty(candidateReason)
                stopReason = candidateReason;
                return;
            end
        end
    end

    stopReason = get_char_local(result,'gridStopReason','');
end

function print_depth_stop_records_local(stages)
    fprintf('\nKAN scratch-depth validation progression:\n');
    fprintf('  %5s %12s %12s %9s %s\n','Depth','ValMSE','PrevBest','Accepted','Status');
    fprintf('  %s\n',repmat('-',1,72));
    for i=1:numel(stages)
        q=stages(i);
        fprintf('  %5d %12.4e %12.4e %9d %s\n', ...
            round(get_numeric_local(q,'depth',NaN)), ...
            get_numeric_local(q,'val_mse',NaN), ...
            get_numeric_local(q,'previous_best_val_mse',NaN), ...
            get_logical_local(q,'accepted',false), ...
            get_char_local(q,'status',''));
        reason=get_char_local(q,'stop_reason','');
        if ~isempty(reason);fprintf('        %s\n',reason);end
    end
end

function print_normalization_local(result)
    inFlag=false;outFlag=false;inherited=false;
    if isfield(result,'normalization')&&isstruct(result.normalization)
        if isfield(result.normalization,'enabled_inputs');inFlag=logical(result.normalization.enabled_inputs);end
        if isfield(result.normalization,'enabled_outputs');outFlag=logical(result.normalization.enabled_outputs);end
        if isfield(result.normalization,'inherited_from_previous_sample');inherited=logical(result.normalization.inherited_from_previous_sample);end
    elseif isfield(result,'opts')
        if isfield(result.opts,'normalizeInputs');inFlag=logical(result.opts.normalizeInputs);end
        if isfield(result.opts,'normalizeOutputs');outFlag=logical(result.opts.normalizeOutputs);end
    end
    fprintf('Input normalization          : %d (training-only z-score)\n',inFlag);
    fprintf('Output normalization         : %d (training-only z-score)\n',outFlag);
    fprintf('Normalization inherited     : %d\n',inherited);
    fprintf('Reported-output scale        : original units (prediction denormalized=%d)\n',outFlag);
end

function print_prune_diagnostics_local(result)
    fprintf('\nSelected-candidate branch diagnostics:\n');
    if isfield(result,'prePruneValMetrics')&&isstruct(result.prePruneValMetrics)&&isfield(result.prePruneValMetrics,'mse')
        fprintf('  Sparse pre-prune Val MSE    : %.6e\n',result.prePruneValMetrics.mse);
    end
    if isfield(result,'immediatePostPruneValMetrics')&&isstruct(result.immediatePostPruneValMetrics)&&isfield(result.immediatePostPruneValMetrics,'mse')
        fprintf('  Immediate post-prune Val MSE: %.6e\n',result.immediatePostPruneValMetrics.mse);
    end
    if isfield(result,'postRefinementValMetrics')&&isstruct(result.postRefinementValMetrics)&&isfield(result.postRefinementValMetrics,'mse')
        fprintf('  Final recovered Val MSE     : %.6e\n',result.postRefinementValMetrics.mse);
    end
end

function print_grid_stages_local(stages)
    fprintf('\nSelected-candidate pyKAN progression:\n');
    fprintf('  %5s %-38s %18s %13s %13s %10s %s\n', ...
        'Grid','Phase','Status','ValMSE','RelImprove','Selected','Note');
    fprintf('  %s\n',repmat('-',1,130));
    for i=1:numel(stages)
        q=stages(i);
        fprintf('  %5d %-38s %18s %13.4e %13.4e %10d %s\n', ...
            round(get_numeric_local(q,'grid',NaN)), ...
            get_char_local(q,'phase',''),get_char_local(q,'status',''), ...
            get_numeric_local(q,'val_mse',NaN), ...
            get_numeric_local(q,'relative_improvement',NaN), ...
            get_logical_local(q,'accepted',false), ...
            get_char_local(q,'stop_reason',''));
        err=get_char_local(q,'error','');if ~isempty(err);fprintf('        error: %s\n',err);end
    end
end

function print_sweep_local(c)
    fprintf('\nKAN candidate sweep (selection uses physical-scale validation MSE only):\n');
    fprintf('  %5s %10s %6s %13s %8s %12s %-38s %10s %-30s\n', ...
        'Depth','lambda','Grid','ValMSE','Edges','Active','Source','Time_s','Status/failure');
    fprintf('  %s\n',repmat('-',1,150));
    for i=1:numel(c)
        if isfield(c(i),'status')&&strcmpi(char(c(i).status),'ok')
            fprintf('  %5d %10.3e %6d %13.4e %8d %12d %-38s %10.3f %-30s\n', ...
                round(get_numeric_local(c(i),'depth',NaN)), ...
                get_numeric_local(c(i),'sparsification_lambda',NaN), ...
                round(get_numeric_local(c(i),'final_grid',NaN)), ...
                get_nested_mse_local(c(i),'val_metrics'), ...
                round(get_numeric_local(c(i),'active_edge_count',NaN)), ...
                round(get_numeric_local(c(i),'active_coefficient_count',NaN)), ...
                get_char_local(c(i),'selected_structure_source',''), ...
                get_numeric_local(c(i),'time_seconds',NaN), ...
                get_char_local(c(i),'grid_stop_reason','completed'));
        else
            detail=sprintf('%s: %s',get_char_local(c(i),'failed_stage','unknown'),get_char_local(c(i),'error',''));
            fprintf('  %5d %10s %6s %13s %8s %12s %-38s %10.3f %-30s\n', ...
                round(get_numeric_local(c(i),'depth',NaN)),'-','-','FAILED','-','-', ...
                get_char_local(c(i),'selected_structure_source','failed'), ...
                get_numeric_local(c(i),'time_seconds',NaN),detail);
        end
    end
end

function v=get_nested_mse_local(s,name)
    v=NaN;if isstruct(s)&&isfield(s,name)&&isstruct(s.(name))&&isfield(s.(name),'mse');v=double(s.(name).mse);end
end
function v=get_numeric_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n))&&isnumeric(s.(n));v=double(s.(n));else;v=d;end;end
function v=get_logical_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=logical(s.(n));else;v=d;end;end
function v=get_char_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=char(string(s.(n)));else;v=d;end;end
function print_metric_local(label,m);fprintf('  %s MSE = %.6e, RMSE = %.6e, NMAE = %.6e\n',label,m.mse,m.rmse,m.nmae);end
