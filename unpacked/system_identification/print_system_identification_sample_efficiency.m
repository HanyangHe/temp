function T = print_system_identification_sample_efficiency(rows,caseLabel)
%PRINT_SYSTEM_IDENTIFICATION_SAMPLE_EFFICIENCY Display derivative/rollout metrics.
%
% A one-method/one-sample diagnostic run produces a scalar structure.  In
% that case some text fields may be empty character arrays while the metric
% fields are scalar numerics.  MATLAB's default scalar-structure conversion
% interprets field contents as table columns and therefore rejects the mixed
% row counts.  Convert scalar records explicitly as a one-row structure
% array; ordinary non-scalar structure arrays retain the standard path.

    fprintf('\n===============================================================\n');
    if nargin < 2 || isempty(caseLabel)
        caseLabel = 'SingleGeneratorDynamic';
    end
    caseLabel = char(string(caseLabel));

    fprintf('%s sample-efficiency summary\n',caseLabel);
    fprintf('N = actual training samples; trajectory metric uses unseen ICs\n');
    fprintf('===============================================================\n');
    fprintf('%8s %8s %5s %-12s %14s %14s %14s %14s %10s %12s\n', ...
        'Ntrain','ModelN','Round','Method','dX_ID_RMSE','dX_OOD_RMSE', ...
        'Traj_RMSE','Traj_NRMSE','Active','Train_s');
    fprintf('%s\n',repmat('-',1,128));

    for k = 1:numel(rows)
        fprintf('%8d %8d %5d %-12s %14s %14s %14s %14s %10s %12s\n', ...
            rows(k).nTrain,round(rows(k).modelTrainingSampleCount), ...
            rows(k).roundIndex,rows(k).method, ...
            fmt_local(rows(k).derivativeRMSE),fmt_local(rows(k).oodDerivativeRMSE), ...
            fmt_local(rows(k).trajectoryRMSE), ...
            fmt_local(rows(k).trajectoryNRMSE),fmt_int_local(rows(k).activeCoefficients), ...
            fmt_local(rows(k).trainTime));
        if strcmpi(rows(k).method,'EQL-Div')
            fprintf('         EQL protocol: %s\n',rows(k).sampleEfficiencyProtocol);
            if isfinite(rows(k).strictCurrentNValidationTargetMSE)
                fprintf(['         strict current-N Val target: %.6e | achieved=%s | ', ...
                    'diagnostic envelope Val=%.6e (model N=%s)\n'], ...
                    rows(k).strictCurrentNValidationTargetMSE, ...
                    fmt_bool_local(rows(k).strictCurrentNImprovementAchieved), ...
                    rows(k).monotoneEnvelopeValidationMSE, ...
                    fmt_int_local(rows(k).monotoneEnvelopeModelTrainingSampleCount));
            end
        end
        if ~rows(k).rolloutAvailable || ~isempty(rows(k).rolloutReason)
            fprintf('         rollout note: %s\n',rows(k).rolloutReason);
        end
    end
    fprintf('===============================================================\n');

    if nargout > 0
        cleanRows = rmfield(rows, ...
            {'rollout','perStateTrajectoryRMSE','perStateTrajectoryNRMSE'});
        T = struct_rows_to_table_local(cleanRows);
    end
end

function T = struct_rows_to_table_local(rows)
%STRUCT_ROWS_TO_TABLE_LOCAL Convert both scalar and array records safely.
    if isempty(rows)
        T = table();
    elseif isscalar(rows)
        T = struct2table(rows,'AsArray',true);
    else
        T = struct2table(rows);
    end
end

function text = fmt_local(x)
    if isempty(x) || ~isscalar(x) || isnan(x)
        text = 'N/A';
    elseif isinf(x)
        text = 'Inf';
    elseif x == 0
        text = '0';
    else
        text = sprintf('%.6e',x);
    end
end

function text = fmt_bool_local(x)
    if isempty(x) || ~isscalar(x) || ~isfinite(x)
        text = 'N/A';
    elseif logical(x)
        text = 'yes';
    else
        text = 'no';
    end
end

function text = fmt_int_local(x)
    if isempty(x) || ~isscalar(x) || ~isfinite(x)
        text = 'N/A';
    else
        text = sprintf('%d',round(x));
    end
end
