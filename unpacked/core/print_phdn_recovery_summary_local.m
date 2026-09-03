function print_phdn_recovery_summary_local(summaryRows)
%PRINT_PHDN_RECOVERY_SUMMARY_LOCAL Robust demo-level recovery summary printer.
%
% v64h hotfix: the demo summary rows use the field name
% `validationMSE`, while an earlier standalone helper looked only for
% `valMSE`.  This function accepts both names, and also includes defensive
% fallbacks for result-like rows.  It is intentionally standalone because some
% live-script (.mlx) versions call this helper after patching without exposing
% the original local function.

    if nargin < 1 || isempty(summaryRows)
        fprintf('\n========================================\n');
        fprintf('PhDN recovery summary\n');
        fprintf('  No summary rows to display.\n');
        fprintf('========================================\n');
        return;
    end

    if istable(summaryRows)
        summaryRows = table2struct(summaryRows);
    end

    if ~isstruct(summaryRows)
        fprintf('\n========================================\n');
        fprintf('PhDN recovery summary\n');
        fprintf('  Summary object type is not supported for tabular printing: %s\n', class(summaryRows));
        fprintf('========================================\n');
        return;
    end

    nRows = numel(summaryRows);
    fprintf('\n========================================\n');
    fprintf('PhDN recovery summary across cases\n');
    fprintf('========================================\n');
    fprintf('%-28s %14s %10s %14s %14s %12s %12s %12s\n', ...
        'Case', 'ValMSE', 'Active', 'ID_RMSE', 'OOD_RMSE', 'Stage0_s', 'Stage1_s', 'Stage2_s');
    fprintf('%-28s %14s %10s %14s %14s %12s %12s %12s\n', ...
        repmat('-',1,28), repmat('-',1,14), repmat('-',1,10), ...
        repmat('-',1,14), repmat('-',1,14), repmat('-',1,12), repmat('-',1,12), repmat('-',1,12));

    for iRow = 1:nRows
        row = summaryRows(iRow);
        caseName = first_field_default_local(row, {'caseName','taskName','case'}, sprintf('case_%d', iRow));

        % The run_demo_feynman_dimless.mlx summary row stores this as
        % `validationMSE`.  Older helper versions printed NaN because they
        % looked only for `valMSE`.
        valMSE  = first_field_default_local(row, {'validationMSE','valMSE','bestValidationMSE','bestValMSE'}, NaN);
        activeN = first_field_default_local(row, {'activeCoefficients','nActiveFinal','finalActiveNumber','activeN'}, NaN);
        idRMSE  = first_field_default_local(row, {'idRMSE','testRMSE','physicalTestRMSE'}, NaN);
        oodRMSE = first_field_default_local(row, {'oodRMSE','oodTestRMSE','oodPhysicalTestRMSE'}, NaN);
        stage0T = first_field_default_local(row, {'stage0Time','stage0_s'}, NaN);
        stage1T = first_field_default_local(row, {'stage1Time','stage1_s'}, NaN);
        stage2T = first_field_default_local(row, {'stage2Time','lsqTime','stage2_s'}, NaN);

        fprintf('%-28s %14s %10s %14s %14s %12s %12s %12s\n', ...
            char_or_string_to_char_local(caseName), ...
            format_num_local(valMSE), ...
            format_active_local(activeN), ...
            format_num_local(idRMSE), ...
            format_num_local(oodRMSE), ...
            format_num_local(stage0T), ...
            format_num_local(stage1T), ...
            format_num_local(stage2T));
    end
    fprintf('========================================\n');
end

function v = first_field_default_local(s, fieldNames, defaultValue)
    v = defaultValue;
    if ~isstruct(s)
        return;
    end
    if ischar(fieldNames) || (isstring(fieldNames) && isscalar(fieldNames))
        fieldNames = cellstr(fieldNames);
    end
    for kk = 1:numel(fieldNames)
        name = fieldNames{kk};
        if isstring(name)
            name = char(name);
        end
        if isfield(s, name)
            try
                tmp = s.(name);
                if ~isempty(tmp)
                    v = tmp;
                    return;
                end
            catch
                % Continue to the next fallback field.
            end
        end
    end
end

function out = format_num_local(x)
    if isempty(x)
        out = 'NaN';
        return;
    end
    if iscell(x)
        x = x{1};
    end
    if isstring(x) || ischar(x)
        out = char_or_string_to_char_local(x);
        return;
    end
    if ~isnumeric(x) || ~isscalar(x) || ~isfinite(x)
        out = 'NaN';
        return;
    end
    ax = abs(x);
    if ax > 0 && (ax < 1e-3 || ax >= 1e4)
        out = sprintf('%.6e', x);
    else
        out = sprintf('%.6g', x);
    end
end

function out = format_active_local(x)
    if isempty(x)
        out = 'NaN';
        return;
    end
    if iscell(x)
        x = x{1};
    end
    if isnumeric(x) && isscalar(x) && isfinite(x)
        out = sprintf('%d', round(x));
    else
        out = char_or_string_to_char_local(x);
    end
end

function out = char_or_string_to_char_local(x)
    if isstring(x)
        if isscalar(x)
            out = char(x);
        else
            out = char(strjoin(x(:).', ','));
        end
    elseif ischar(x)
        out = x;
    elseif isnumeric(x) && isscalar(x)
        out = sprintf('%.6g', x);
    else
        try
            out = char(string(x));
        catch
            out = '<unprintable>';
        end
    end
end
