function print_eql_baseline_result(result)
%PRINT_EQL_BASELINE_RESULT Print a consistency-checked official EQL-Div report.
%
% This reporting function is deliberately read-only: it does not alter the
% trained model, selected predictions, metrics, or model-selection result.
% It fixes two reporting failure modes:
%   1) Python JSON candidate arrays may arrive in MATLAB as a cell array.
%      The old reporter treated every cell as a failed struct and printed
%      NaN/FAILED rows.
%   2) A cached/result-level depth field may disagree with the candidate
%      carrying selected=true.  The selected candidate is the authoritative
%      source for the displayed L/lambda, while both values are shown when a
%      mismatch is detected.

    candidates = normalize_candidates_local(getfield_default_local(result,'candidates',[]));
    [selectedCandidate, selectedIndex] = find_selected_candidate_local(candidates,result);

    resultDepth = scalar_number_local(getfield_default_local(result,'depth',NaN));
    resultLambda = scalar_number_local(getfield_default_local(result,'lambda',NaN));
    resultFunctionalLayers = scalar_number_local( ...
        getfield_default_local(result,'functionalLayerCount',NaN));

    selectedDepth = resultDepth;
    selectedLambda = resultLambda;
    selectedFunctionalLayers = resultFunctionalLayers;
    if ~isempty(selectedCandidate)
        selectedDepth = scalar_number_local(getfield_default_local( ...
            selectedCandidate,'paper_depth_L',selectedDepth));
        selectedLambda = scalar_number_local(getfield_default_local( ...
            selectedCandidate,'lambda_l1',selectedLambda));
        selectedFunctionalLayers = scalar_number_local(getfield_default_local( ...
            selectedCandidate,'official_hidden_layers',selectedFunctionalLayers));
    end

    fprintf('\n========================================\n');
    fprintf('Official EQL-Div baseline result\n');
    fprintf('========================================\n');
    if isfield(result,'dataSource')
        fprintf('Data source                  : %s\n',char(result.dataSource));
    end
    fprintf('Method                       : %s\n',char(result.method));
    fprintf('Protocol                     : %s\n',char(result.protocol));
    if isfield(result,'officialSourceCommit')
        fprintf('Upstream repository/commit   : martius-lab/EQL / %s\n', ...
            char(result.officialSourceCommit));
    end
    if isfield(result,'officialSourceUnmodified')
        fprintf('Official source unmodified   : %d\n', ...
            logical(result.officialSourceUnmodified));
    end

    print_integer_or_na_local('Selected paper depth L       ',selectedDepth);
    print_integer_or_na_local('Official hidden layers L-1   ',selectedFunctionalLayers);
    print_float_or_na_local('Selected sparsity lambda     ',selectedLambda,'%.3e');
    checkpoint=getfield_default_local(result,'selectedCheckpoint','');
    if strlength(string(checkpoint))==0 && ~isempty(selectedCandidate)
        checkpoint=getfield_default_local(selectedCandidate,'selected_checkpoint','');
    end
    if strlength(string(checkpoint))>0
        fprintf('Exported upstream checkpoint : %s\n',char(string(checkpoint)));
    end

    if ~isempty(selectedCandidate)
        fprintf('Selected candidate source    : candidates(%d), selected=true\n',selectedIndex);
    else
        fprintf('Selected candidate source    : result-level selected fields\n');
    end

    depthMismatch = isfinite(resultDepth) && isfinite(selectedDepth) && ...
        round(resultDepth) ~= round(selectedDepth);
    lambdaMismatch = isfinite(resultLambda) && isfinite(selectedLambda) && ...
        abs(resultLambda-selectedLambda) > ...
        1e-12*max([1,abs(resultLambda),abs(selectedLambda)]);
    if depthMismatch || lambdaMismatch
        fprintf(['REPORT CONSISTENCY WARNING    : result-level selection fields ', ...
            'disagree with selected candidate metadata.\n']);
        print_integer_or_na_local('  result.depth               ',resultDepth);
        print_float_or_na_local('  result.lambda              ',resultLambda,'%.3e');
        fprintf(['  Displayed L/lambda use the candidate explicitly marked ', ...
            'selected; training outputs are unchanged.\n']);
    end

    print_float_field_local(result,'selectionScore', ...
        'Official Vint-S score        ','%.6e');
    fprintf('Reported regression MSE scale: original physical output units (same as other baselines)\n');
    print_integer_field_local(result,'nActiveUnits', ...
        'Official active units        ');
    print_integer_field_local(result,'unitsPerUnaryType', ...
        'Units per function type      ');
    if isfield(result,'operatorFamily') && ~isempty(result.operatorFamily)
        fprintf('Fixed operator family        : {%s}\n', ...
            strjoin(cellstr(string(result.operatorFamily)),','));
    end
    if isfield(result,'parameterCount') || isfield(result,'activeWeightCount')
        fprintf('Parameters / active weights  : %s / %s\n', ...
            integer_text_local(getfield_default_local(result,'parameterCount',NaN)), ...
            integer_text_local(getfield_default_local(result,'activeWeightCount',NaN)));
    end
    print_integer_field_local(result,'nActiveCoefficients', ...
        'Active coefficients total    ');
    if isfield(result,'trainFcn')
        fprintf('Training function            : %s\n',char(result.trainFcn));
    end
    if isfield(result,'dataMode')
        fprintf('Adapter data mode             : %s\n',char(result.dataMode));
    end
    if isfield(result,'officialSettings') && isstruct(result.officialSettings)
        fprintf('External input/output scaling : %d / %d\n', ...
            logical(getfield_default_local(result.officialSettings, ...
                'uses_input_normalization',false)), ...
            logical(getfield_default_local(result.officialSettings, ...
                'uses_output_normalization',false)));
    end
    if isfield(result,'usesOodLabelsForSelection')
        fprintf('OOD labels used for selection: %d\n', ...
            logical(result.usesOodLabelsForSelection));
    end
    requestedBatch = scalar_number_local(getfield_default_local(result,'requestedBatchSize',NaN));
    effectiveBatch = scalar_number_local(getfield_default_local(result,'effectiveBatchSize',requestedBatch));
    if isfinite(requestedBatch) || isfinite(effectiveBatch)
        fprintf('Requested/effective batch size: %s / %s\n', ...
            integer_text_local(requestedBatch),integer_text_local(effectiveBatch));
        if logical_scalar_local(getfield_default_local(result,'batchSizeAdjusted',false))
            fprintf('Batch-size adaptation         : complete-minibatch divisor; no samples dropped\n');
        end
    end

    configuredCount = scalar_number_local(getfield_default_local( ...
        result,'configuredCandidateCount',NaN));
    attemptedCount = scalar_number_local(getfield_default_local( ...
        result,'attemptedCandidateCount',NaN));
    if ~isfinite(attemptedCount)
        attemptedCount = scalar_number_local(getfield_default_local( ...
            result,'candidateCount',numel(candidates)));
    end
    if ~isfinite(attemptedCount)
        attemptedCount = numel(candidates);
    end
    successfulCount = count_successful_local(candidates);
    storedSuccessfulCount = scalar_number_local(getfield_default_local( ...
        result,'successfulCandidateCount',NaN));
    if successfulCount == 0 && isfinite(storedSuccessfulCount)
        successfulCount = storedSuccessfulCount;
    end

    if isfinite(configuredCount)
        fprintf('Configured/attempted candidates: %d / %d\n', ...
            round(configuredCount),round(attemptedCount));
    else
        fprintf('Attempted candidates          : %d\n',round(attemptedCount));
    end
    fprintf('Successful candidates         : %d\n',round(successfulCount));

    if isfield(result,'depthEarlyStop')
        fprintf('Validation depth early stop  : %d\n',logical(result.depthEarlyStop));
    end
    lastAttemptedDepth = infer_last_attempted_depth_local(result,candidates);
    if isfinite(lastAttemptedDepth)
        fprintf('Last attempted paper depth L : %d\n',round(lastAttemptedDepth));
    end
    if isfield(result,'depthStopReason') && ~isempty(result.depthStopReason)
        fprintf('Depth stop reason            : %s\n',char(result.depthStopReason));
    end

    print_float_field_local(result,'trainTime', ...
        'Complete sweep wall time     ','%.3f s');
    print_float_field_local(result,'selectedModelTrainTime', ...
        'Selected-candidate wall time ','%.3f s');

    if ~isempty(candidates) && get_opt_local(result,'displaySweepTable',true)
        print_sweep_table_local(candidates,selectedIndex);
    end

    fprintf('\nIn-distribution metrics:\n');
    print_metric_local('Train',getfield_default_local(result,'trainMetrics',struct()));
    print_metric_local('Val  ',getfield_default_local(result,'valMetrics',struct()));
    print_metric_local('Test ',getfield_default_local(result,'testMetrics',struct()));
    if isfield(result,'oodMetrics') && isstruct(result.oodMetrics) && ...
            isfield(result.oodMetrics,'mse') && isfinite(result.oodMetrics.mse)
        fprintf('\nOut-of-distribution metrics:\n');
        print_metric_local('OOD  ',result.oodMetrics);
    end
    fprintf('========================================\n');
end

function candidates = normalize_candidates_local(raw)
% Convert Python/JSON cell arrays, scalar structs, and struct arrays to a
% row cell array of scalar MATLAB structs.
    candidates = cell(1,0);
    if isempty(raw)
        return;
    end
    if iscell(raw)
        for i = 1:numel(raw)
            item = raw{i};
            if isstruct(item)
                for j = 1:numel(item)
                    candidates{end+1} = item(j); %#ok<AGROW>
                end
            end
        end
    elseif isstruct(raw)
        for i = 1:numel(raw)
            candidates{end+1} = raw(i); %#ok<AGROW>
        end
    end
end

function [selected,index] = find_selected_candidate_local(candidates,result)
    selected = [];
    index = NaN;
    for i = 1:numel(candidates)
        c = candidates{i};
        if logical_scalar_local(getfield_default_local(c,'selected',false))
            selected = c;
            index = i;
            return;
        end
    end

    selectedIndex = scalar_number_local(getfield_default_local( ...
        result,'selectedCandidateIndex',NaN));
    if ~isfinite(selectedIndex) && isfield(result,'pyResult') && ...
            isstruct(result.pyResult)
        selectedIndex = scalar_number_local(getfield_default_local( ...
            result.pyResult,'selected_index',NaN));
    end
    if isfinite(selectedIndex) && selectedIndex >= 1 && ...
            selectedIndex <= numel(candidates)
        index = round(selectedIndex);
        selected = candidates{index};
        return;
    end

    resultDepth = scalar_number_local(getfield_default_local(result,'depth',NaN));
    resultLambda = scalar_number_local(getfield_default_local(result,'lambda',NaN));
    for i = 1:numel(candidates)
        c = candidates{i};
        d = scalar_number_local(getfield_default_local(c,'paper_depth_L',NaN));
        lam = scalar_number_local(getfield_default_local(c,'lambda_l1',NaN));
        if isfinite(d) && isfinite(resultDepth) && round(d)==round(resultDepth) && ...
                isfinite(lam) && isfinite(resultLambda) && ...
                abs(lam-resultLambda) <= 1e-12*max([1,abs(lam),abs(resultLambda)])
            selected = c;
            index = i;
            return;
        end
    end
end

function print_sweep_table_local(candidates,selectedIndex)
    fprintf('\nOfficial EQL L/lambda sweep (candidate ValMSE in original output units; no OOD labels):\n');
    fprintf('  %3s %11s %10s %12s %9s %10s %10s %10s %8s\n', ...
        'L','lambda','Checkpoint','ValMSE','ActiveU','ActiveW','Params','Time_s','Selected');
    fprintf('  %s\n',repmat('-',1,114));

    for i = 1:numel(candidates)
        c = candidates{i};
        status = lower(strtrim(char(string(getfield_default_local(c,'status','')))));
        physicalVal=scalar_number_local(getfield_default_local(c,'framework_val_mse',NaN));
        if ~isfinite(physicalVal)
            physicalVal=scalar_number_local(getfield_default_local(c,'official_final_val_mse',NaN));
        end
        isOk = strcmp(status,'ok') || strcmp(status,'success') || ...
            (isempty(status) && isfinite(physicalVal));
        isSelected = logical_scalar_local(getfield_default_local(c,'selected',false)) || ...
            (isfinite(selectedIndex) && i==selectedIndex);
        checkpoint=char(string(getfield_default_local(c,'selected_checkpoint','legacy')));
        activeUnits=getfield_default_local(c,'selected_num_active', ...
            getfield_default_local(c,'official_num_active',NaN));

        if isOk
            fprintf('  %3s %11s %10s %12s %9s %10s %10s %10s %8s\n', ...
                integer_text_local(getfield_default_local(c,'paper_depth_L',NaN)), ...
                float_text_local(getfield_default_local(c,'lambda_l1',NaN),'%.3e'), ...
                checkpoint,float_text_local(physicalVal,'%.4e'), ...
                integer_text_local(activeUnits), ...
                integer_text_local(getfield_default_local(c,'active_weight_count',NaN)), ...
                integer_text_local(getfield_default_local(c,'parameter_count',NaN)), ...
                float_text_local(getfield_default_local(c,'time_seconds',NaN),'%.3f'), ...
                ternary_local(isSelected,'yes',''));
        else
            fprintf('  %3s %11s %10s %12s %9s %10s %10s %10s %8s\n', ...
                integer_text_local(getfield_default_local(c,'paper_depth_L',NaN)), ...
                float_text_local(getfield_default_local(c,'lambda_l1',NaN),'%.3e'), ...
                checkpoint,'FAILED','-','-','-', ...
                float_text_local(getfield_default_local(c,'time_seconds',NaN),'%.3f'),'-');
            err = getfield_default_local(c,'error','');
            if strlength(string(err)) > 0
                fprintf('      failure: %s\n',char(string(err)));
            end
            logPath = getfield_default_local(c,'stdout_path','');
            if strlength(string(logPath)) > 0
                fprintf('      log: %s\n',char(string(logPath)));
            end
        end
    end
end

function count = count_successful_local(candidates)
    count = 0;
    for i = 1:numel(candidates)
        c = candidates{i};
        status = lower(strtrim(char(string(getfield_default_local(c,'status','')))));
        if strcmp(status,'ok') || strcmp(status,'success') || ...
                (isempty(status) && isfinite(scalar_number_local( ...
                    getfield_default_local(c,'framework_val_mse', ...
                        getfield_default_local(c,'official_final_val_mse',NaN)))))
            count = count + 1;
        end
    end
end

function d = infer_last_attempted_depth_local(result,candidates)
    d = scalar_number_local(getfield_default_local(result,'lastAttemptedDepth',NaN));
    if ~isfinite(d)
        d = scalar_number_local(getfield_default_local(result,'lastAttemptedPaperDepth',NaN));
    end
    if ~isfinite(d)
        values = [];
        for i = 1:numel(candidates)
            candidateDepth = scalar_number_local(getfield_default_local( ...
                candidates{i},'paper_depth_L',NaN));
            if isfinite(candidateDepth)
                values(end+1) = candidateDepth; %#ok<AGROW>
            end
        end
        if ~isempty(values)
            d = max(values);
        end
    end
end

function print_metric_local(label,m)
    mse = scalar_number_local(getfield_default_local(m,'mse',NaN));
    rmse = scalar_number_local(getfield_default_local(m,'rmse',NaN));
    nmae = scalar_number_local(getfield_default_local(m,'nmae',NaN));
    fprintf('  %s MSE = %s, RMSE = %s, NMAE = %s\n',label, ...
        float_text_local(mse,'%.6e'),float_text_local(rmse,'%.6e'), ...
        float_text_local(nmae,'%.6e'));
end

function print_integer_field_local(s,field,label)
    if isfield(s,field)
        print_integer_or_na_local(label,scalar_number_local(s.(field)));
    end
end
function print_float_field_local(s,field,label,fmt)
    if isfield(s,field)
        print_float_or_na_local(label,scalar_number_local(s.(field)),fmt);
    end
end
function print_integer_or_na_local(label,v)
    fprintf('%s: %s\n',label,integer_text_local(v));
end
function print_float_or_na_local(label,v,fmt)
    fprintf('%s: %s\n',label,float_text_local(v,fmt));
end
function text = integer_text_local(v)
    v = scalar_number_local(v);
    if isfinite(v); text=sprintf('%d',round(v)); else; text='N/A'; end
end
function text = float_text_local(v,fmt)
    v = scalar_number_local(v);
    if isfinite(v); text=sprintf(fmt,v); else; text='N/A'; end
end
function v = scalar_number_local(raw)
    v = NaN;
    if isempty(raw); return; end
    if isnumeric(raw) || islogical(raw)
        if isscalar(raw); v=double(raw); end
    elseif ischar(raw) || isstring(raw)
        parsed = str2double(string(raw));
        if isscalar(parsed); v=double(parsed); end
    end
end
function tf = logical_scalar_local(raw)
    tf = false;
    if islogical(raw) || isnumeric(raw)
        if isscalar(raw); tf=logical(raw); end
    elseif ischar(raw) || isstring(raw)
        value = lower(strtrim(char(string(raw))));
        tf = any(strcmp(value,{'true','yes','1'}));
    end
end
function tf = get_opt_local(r,n,d)
    tf=d;
    if isfield(r,'opts') && isstruct(r.opts) && isfield(r.opts,n)
        tf=logical(r.opts.(n));
    end
end
function v = getfield_default_local(s,n,d)
    if isstruct(s) && isfield(s,n) && ~isempty(s.(n)); v=s.(n); else; v=d; end
end
function y = ternary_local(tf,a,b)
    if tf; y=a; else; y=b; end
end
