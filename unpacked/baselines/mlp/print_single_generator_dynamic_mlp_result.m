function print_single_generator_dynamic_mlp_result(result)
%PRINT_SINGLE_GENERATOR_DYNAMIC_MLP_RESULT Print selected MLP and architecture sweep metrics.
    fprintf('\n========================================\nMLP baseline result\n========================================\n');
    if isfield(result,'dataSource'); fprintf('Data source                  : %s\n',result.dataSource); end
    fprintf('Method                       : %s\n',result.method);
    if isfield(result,'protocol'); fprintf('Protocol                     : %s\n',result.protocol); end
    if isfield(result,'depth'); fprintf('Selected depth               : %d affine layers (%d hidden layers)\n',result.depth,result.hiddenLayerCount); end
    if isfield(result,'hiddenLayerSizes'); fprintf('Selected hidden layers       : [%s]\n',num2str(result.hiddenLayerSizes)); end
    if isfield(result,'activation'); fprintf('Selected activation          : %s\n',result.activation); end
    if isfield(result,'parameterCount'); fprintf('Selected parameter count     : %d\n',round(result.parameterCount)); end
    fprintf('Training function            : %s\n',result.trainFcn);
    fprintf('Complete sweep wall time     : %.3f s\n',result.trainTime);
    if isfield(result,'selectedModelTrainTime'); fprintf('Selected-candidate train time: %.3f s\n',result.selectedModelTrainTime); end
    if isfield(result,'configuredCandidateCount'); fprintf('Configured/attempted candidates: %d / %d\n',round(result.configuredCandidateCount),round(result.candidateCount)); end
    if isfield(result,'depthEarlyStop'); fprintf('Validation depth early stop  : %d\n',logical(result.depthEarlyStop)); end
    if isfield(result,'lastAttemptedDepth'); fprintf('Last attempted depth         : %d\n',round(result.lastAttemptedDepth)); end
    if isfield(result,'depthStopReason') && ~isempty(result.depthStopReason); fprintf('Depth stop reason            : %s\n',result.depthStopReason); end
    if isfield(result,'depthStages') && ~isempty(result.depthStages); print_depth_stages_local(result.depthStages); end
    if isfield(result,'candidates') && get_opt_local(result,'displaySweepTable',true)
        print_sweep_table_local(result.candidates);
    end
    fprintf('\nIn-distribution metrics:\n');
    print_metric_local('Train',result.trainMetrics); print_metric_local('Val  ',result.valMetrics); print_metric_local('Test ',result.testMetrics);
    if isfield(result,'oodMetrics') && isfinite(result.oodMetrics.mse)
        fprintf('\nOut-of-distribution metrics:\n'); print_metric_local('OOD  ',result.oodMetrics);
    end
    fprintf('========================================\n');
end

function print_depth_stages_local(stages)
    fprintf('\nDepth-wise validation progression:\n');
    fprintf('  %5s %-8s %12s %12s %9s %s\n','Depth','BestAct','BestValMSE','PrevBest','Accepted','Status');
    fprintf('  %s\n',repmat('-',1,78));
    for i=1:numel(stages)
        q=stages(i);
        fprintf('  %5d %-8s %12.4e %12.4e %9d %s\n', ...
            round(getfield_default_local(q,'depth',NaN)), ...
            char(getfield_default_local(q,'best_activation','')), ...
            getfield_default_local(q,'best_val_mse',NaN), ...
            getfield_default_local(q,'previous_global_best_val_mse',NaN), ...
            logical(getfield_default_local(q,'accepted',false)), ...
            char(getfield_default_local(q,'status','')));
        reason=char(getfield_default_local(q,'stop_reason',''));
        if ~isempty(reason);fprintf('        %s\n',reason);end
    end
end

function print_sweep_table_local(c)
    if isempty(c); return; end
    fprintf('\nArchitecture sweep (selection uses validation MSE only):\n');
    fprintf('  %5s %7s %-7s %12s %12s %10s\n','Depth','Hidden','Act','ValMSE','Params','Time_s');
    fprintf('  %s\n',repmat('-',1,61));
    for i=1:numel(c)
        if isfield(c(i),'status') && strcmpi(char(c(i).status),'ok')
            vm=get_nested_local(c(i),{'val_metrics','mse'},NaN);
            pc=getfield_default_local(c(i),'parameter_count',NaN);
            fprintf('  %5d %7d %-7s %12.4e %12.0f %10.3f\n',c(i).depth,c(i).hidden_layer_count,char(c(i).activation),vm,pc,c(i).time_seconds);
        else
            fprintf('  %5d %7d %-7s %12s %12s %10.3f\n',c(i).depth,c(i).hidden_layer_count,char(c(i).activation),'FAILED','-',c(i).time_seconds);
        end
    end
end
function print_metric_local(label,m)
    fprintf('  %s MSE = %.6e, RMSE = %.6e, NMAE = %.6e\n',label,m.mse,m.rmse,m.nmae);
end
function tf=get_opt_local(r,n,d); tf=d; if isfield(r,'opts')&&isstruct(r.opts)&&isfield(r.opts,n);tf=logical(r.opts.(n));end;end
function v=get_nested_local(s,p,d);v=s;for k=1:numel(p);if isstruct(v)&&isfield(v,p{k});v=v.(p{k});else;v=d;return;end;end;end
function v=getfield_default_local(s,n,d);if isstruct(s)&&isfield(s,n)&&~isempty(s.(n));v=s.(n);else;v=d;end;end
