function rows = append_system_identification_result_row(rows, nTrain, roundIndex, methodName, methodResult, rollout)
%APPEND_SYSTEM_IDENTIFICATION_RESULT_ROW Add one sample-efficiency record.

    if nargin < 6 || isempty(rollout)
        rollout = empty_rollout_local();
    end

    row = struct();
    row.nTrain = nTrain;
    row.roundIndex = roundIndex;
    row.method = methodName;
    % Optional robustness-axis metadata. Ordinary sample-efficiency demos leave
    % these as NaN; the derivative-noise demo fills them after each append.
    row.noiseLevel = NaN;
    row.noisePercent = NaN;
    row.noiseSeed = NaN;
    methodFamily = method_family_local(methodName,methodResult);
    row.methodFamily = methodFamily;
    row.derivativeRMSE = extract_id_rmse_local(methodFamily,methodResult);
    row.derivativeNRMSE = extract_id_nrmse_local(methodFamily,methodResult);
    row.oodDerivativeRMSE = extract_ood_rmse_local(methodFamily,methodResult);
    row.validationMSE = extract_validation_mse_local(methodFamily,methodResult);
    row.activeCoefficients = extract_active_local(methodFamily,methodResult);
    row.trainTime = extract_train_time_local(methodFamily,methodResult);
    row.modelTrainingSampleCount = first_finite_local(methodResult, ...
        {{'modelTrainingSampleCount'},{'exactNTrainingSampleCount'}});
    if ~isfinite(row.modelTrainingSampleCount);row.modelTrainingSampleCount=nTrain;end
    row.sampleEfficiencyProtocol = char(get_text_local(methodResult, ...
        'paperSampleEfficiencyProtocol','exact_N_independent_training'));
    row.previousModelRole = char(get_text_local(methodResult, ...
        'previousModelRole','not_applicable'));
    row.selectedCheckpoint = char(get_text_local(methodResult, ...
        'selectedCheckpoint',''));
    row.selectedCheckpointEpoch = nested_local(methodResult, ...
        {'selectedCheckpointEpoch'},NaN);
    row.selectedCheckpointPhase = char(get_text_local(methodResult, ...
        'selectedCheckpointPhase',''));
    row.selectedCandidateSource = char(get_text_local(methodResult, ...
        'selectedCandidateSource',''));
    row.selectedRestartIndex = nested_local(methodResult, ...
        {'selectedRestartIndex'},NaN);
    row.strictCurrentNImprovementAchieved = get_logical_local(methodResult, ...
        'strictCurrentNImprovementAchieved',NaN);
    row.strictCurrentNValidationTargetMSE = nested_local(methodResult, ...
        {'strictCurrentNValidationTargetMSE'},NaN);
    row.monotoneEnvelopeValidationMSE = nested_local(methodResult, ...
        {'monotoneEnvelopeValMSE'},NaN);
    row.monotoneEnvelopeModelTrainingSampleCount = nested_local(methodResult, ...
        {'monotoneEnvelopeModelTrainingSampleCount'},NaN);
    row.trajectoryRMSE = rollout.rawRMSE;
    row.trajectoryNRMSE = rollout.normalizedRMSE;
    row.rolloutSuccessRate = rollout.successRate;
    row.rolloutAvailable = rollout.available;
    row.rolloutReason = rollout.reason;
    row.perStateTrajectoryRMSE = rollout.perStateRMSE;
    row.perStateTrajectoryNRMSE = rollout.perStateNormalizedRMSE;
    row.rollout = rollout;

    if isempty(rows)
        rows = row;
    else
        rows(end+1) = row;
    end
end

function value = extract_id_rmse_local(methodName,r)
    if strcmpi(methodName,'phdn')
        value = first_finite_local(r,{{'testMetrics','rmse'},{'physicalTestRMSE'},{'testRMSE'}});
    else
        value = nested_local(r,{'testMetrics','rmse'},NaN);
    end
end

function value = extract_id_nrmse_local(methodName,r)
    if strcmpi(methodName,'phdn')
        value = first_finite_local(r,{{'physicalTestMetrics','nrmse'}, ...
            {'testMetrics','nrmse'}});
    else
        value = nested_local(r,{'testMetrics','nrmse'},NaN);
    end
end

function value = extract_ood_rmse_local(methodName,r)
    if strcmpi(methodName,'phdn')
        value = first_finite_local(r,{{'oodTestMetrics','rmse'}, ...
            {'oodPhysicalTestMetrics','rmse'},{'oodPhysicalTestRMSE'}, ...
            {'oodTestRMSE'},{'timeStats','stage0OODTestRMSE'}});
    else
        value = nested_local(r,{'oodMetrics','rmse'},NaN);
    end
end

function value = extract_validation_mse_local(methodName,r)
    if strcmpi(methodName,'phdn')
        value = first_finite_local(r,{{'bestValidationMSE'},{'valMetrics','mse'}});
    else
        value = nested_local(r,{'valMetrics','mse'},NaN);
    end
end

function value = extract_active_local(methodName,r)
    if strcmpi(methodName,'phdn')
        value = nested_local(r,{'nActiveFinal'},NaN);
    elseif strcmpi(methodName,'eql-div')
        value = first_finite_local(r,{{'nActiveUnits'},{'nActiveCoefficients'}});
    else
        value = nested_local(r,{'nActiveCoefficients'},NaN);
        if ~isfinite(value)
            value = nested_local(r,{'parameterCount'},NaN);
        end
    end
end

function value = extract_train_time_local(methodName,r)
    if strcmpi(methodName,'phdn')
        parts = [nested_local(r,{'timeStats','stage0Time'},NaN), ...
            nested_local(r,{'timeStats','stage1Time'},NaN), ...
            nested_local(r,{'timeStats','lsqTime'},NaN)];
        if any(isfinite(parts))
            value = sum(parts(isfinite(parts)));
        else
            value = nested_local(r,{'timeStats','trainingWallTime'},NaN);
        end
    else
        value = nested_local(r,{'trainTime'},NaN);
        if ~isfinite(value)
            value = nested_local(r,{'timeStats','total'},NaN);
        end
    end
end


function family = method_family_local(methodName,r)
    if isstruct(r) && isfield(r,'methodFamily') && ~isempty(r.methodFamily)
        family = lower(strtrim(char(r.methodFamily)));
        return;
    end
    compact = regexprep(lower(strtrim(char(methodName))),'[^a-z0-9]','');
    if startsWith(compact,'phdn')
        family = 'phdn';
    elseif startsWith(compact,'stage0sr')
        family = 'stage0-sr';
    elseif startsWith(compact,'eql')
        family = 'eql-div';
    elseif startsWith(compact,'kan')
        family = 'kan';
    elseif startsWith(compact,'sindy')
        family = 'sindy';
    elseif startsWith(compact,'mlp')
        family = 'mlp';
    else
        family = lower(strtrim(char(methodName)));
    end
end

function value = first_finite_local(s,paths)
    value = NaN;
    for k = 1:numel(paths)
        candidate = nested_local(s,paths{k},NaN);
        if isfinite(candidate)
            value = candidate;
            return;
        end
    end
end

function value = get_text_local(s,fieldName,defaultValue)
    value=defaultValue;
    if isstruct(s)&&isfield(s,fieldName)&&~isempty(s.(fieldName))
        value=char(string(s.(fieldName)));
    end
end

function value = get_logical_local(s,fieldName,defaultValue)
    value=defaultValue;
    if isstruct(s)&&isfield(s,fieldName)&&~isempty(s.(fieldName))
        raw=s.(fieldName);
        if islogical(raw)
            value=double(raw(1));
        elseif isnumeric(raw)&&isfinite(raw(1))
            value=double(logical(raw(1)));
        end
    end
end


function rollout = empty_rollout_local()
    rollout = struct();
    rollout.rawRMSE = NaN;
    rollout.normalizedRMSE = NaN;
    rollout.successRate = NaN;
    rollout.available = false;
    rollout.reason = 'not_evaluated';
    rollout.perStateRMSE = [];
    rollout.perStateNormalizedRMSE = [];
end

function value = nested_local(s,path,defaultValue)
    value = defaultValue;
    current = s;
    for k = 1:numel(path)
        if ~isstruct(current) || ~isfield(current,path{k}) || isempty(current.(path{k}))
            return;
        end
        current = current.(path{k});
    end
    if (isnumeric(current) || islogical(current)) && isscalar(current)
        value = double(current);
    end
end
