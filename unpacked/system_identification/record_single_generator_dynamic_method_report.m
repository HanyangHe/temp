function result = record_single_generator_dynamic_method_report(result, methodField, task)
%RECORD_SINGLE_GENERATOR_DYNAMIC_METHOD_REPORT Store a replayable text report.
%
% The report is generated from the already completed result structure, so this
% function does not retrain or reevaluate the model. It preserves the detailed
% method report printed by the case-local printer and allows a later run to
% skip training while restoring the same report block.
%
% For PhDN, TASK is required because print_demo_output needs the task metadata.
% Existing baseline calls remain backward compatible and do not need TASK.

    if nargin < 2 || isempty(methodField)
        error('methodField is required.');
    end
    if nargin < 3
        task = [];
    end
    if ~isstruct(result) || isempty(result)
        error('A nonempty method result structure is required.');
    end

    reportText = render_report_local(result,methodField,task);
    result.recordedConsoleReport = reportText;
    result.recordedConsoleReportMethod = lower(char(methodField));
    result.recordedConsoleReportMode = 'rendered_from_completed_result';
    result.recordedConsoleReportVersion = 2;
    result.recordedConsoleReportGeneratedAt = ...
        char(datetime('now','Format','yyyy-MM-dd HH:mm:ss'));
end

function reportText = render_report_local(result,methodField,task)
    methodField = lower(strtrim(char(methodField)));
    if startsWith(methodField,'phdn')
        if isempty(task) || ~isstruct(task)
            error(['A valid task structure is required to render a PhDN ', ...
                'record report.']);
        end
        reportText = evalc('print_demo_output(task,result);');
        return;
    end

    switch methodField
        case 'mlp'
            reportText = evalc('print_single_generator_dynamic_mlp_result(result);');
        case 'eql'
            reportText = evalc('print_single_generator_dynamic_eql_result(result);');
        case 'kan'
            reportText = evalc('print_single_generator_dynamic_kan_result(result);');
        case 'sindy'
            reportText = evalc('print_sindy_baseline_result(result);');
        otherwise
            error('Unsupported recorded-report method field: %s',methodField);
    end
end
