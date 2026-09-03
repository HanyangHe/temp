function print_method_comparison_summary_local(summaryRows)
%PRINT_METHOD_COMPARISON_SUMMARY_LOCAL Print the per-run method summary table.
%
% Metric values are never rounded to integers merely because they are close
% to zero.  Exact zero is printed as 0; small nonzero values are printed in
% scientific notation.  Active-coefficient counts use a separate integer
% formatter.

	fprintf('========================================\n');
	fprintf('Method recovery summary across cases\n');
	fprintf('========================================\n');

	fmt = '%-28s %-10s %14s %10s %14s %14s %14s %14s %14s %12s %12s %12s %12s %12s\n';
	fprintf(fmt, ...
		'Case', 'Method', 'ValMSE', 'Active', 'ID_RMSE', 'OOD_RMSE', ...
		'S0_ValMSE', 'S0_ID_RMSE', 'S0_OOD_RMSE', ...
		'Train_s', 'Selected_s', 'Stage0_s', 'Stage1_s', 'Stage2_s');
	fprintf(fmt, ...
		repmat('-', 1, 28), repmat('-', 1, 10), repmat('-', 1, 14), repmat('-', 1, 10), ...
		repmat('-', 1, 14), repmat('-', 1, 14), repmat('-', 1, 14), repmat('-', 1, 14), ...
		repmat('-', 1, 14), repmat('-', 1, 12), repmat('-', 1, 12), repmat('-', 1, 12), ...
		repmat('-', 1, 12), repmat('-', 1, 12));

	if nargin < 1 || isempty(summaryRows)
		fprintf('(no enabled method results)\n');
		fprintf('========================================\n');
		return;
	end
	if istable(summaryRows)
		summaryRows = table2struct(summaryRows);
	end
	if ~isstruct(summaryRows)
		fprintf('(unsupported summary object: %s)\n', class(summaryRows));
		fprintf('========================================\n');
		return;
	end

	for i = 1:numel(summaryRows)
		row = summaryRows(i);
		selectedTime = get_num_field_any_local(row, ...
			{'selectedTime','selectedModelTime','selectionTime','selectedCandidateTime','modelSelectionTime'}, NaN);
		fprintf(fmt, ...
			get_text_field_local(row, 'caseName', ''), ...
			get_text_field_local(row, 'method', ''), ...
			format_metric_local(get_num_field_local(row, 'validationMSE', NaN)), ...
			format_integer_local(get_num_field_local(row, 'activeCoefficients', NaN)), ...
			format_metric_local(get_num_field_local(row, 'idRMSE', NaN)), ...
			format_metric_local(get_num_field_local(row, 'oodRMSE', NaN)), ...
			format_metric_local(get_num_field_local(row, 'stage0ValidationMSE', NaN)), ...
			format_metric_local(get_num_field_local(row, 'stage0IDRMSE', NaN)), ...
			format_metric_local(get_num_field_local(row, 'stage0OODRMSE', NaN)), ...
			format_metric_local(get_num_field_local(row, 'trainTime', NaN)), ...
			format_metric_local(selectedTime), ...
			format_metric_local(get_num_field_local(row, 'stage0Time', NaN)), ...
			format_metric_local(get_num_field_local(row, 'stage1Time', NaN)), ...
			format_metric_local(get_num_field_local(row, 'stage2Time', NaN)));
	end
	fprintf('========================================\n');
end

function txt = get_text_field_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		txt = char(string(s.(name)));
	else
		txt = defaultVal;
	end
end

function val = get_num_field_local(s, name, defaultVal)
	val = defaultVal;
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		x = s.(name);
		if (isnumeric(x) || islogical(x)) && isscalar(x)
			val = double(x);
		end
	end
end

function val = get_num_field_any_local(s, names, defaultVal)
	val = defaultVal;
	for k = 1:numel(names)
		x = get_num_field_local(s, names{k}, NaN);
		if isfinite(x)
			val = x;
			return;
		end
	end
end

function out = format_integer_local(x)
	if ~isnumeric(x) || isempty(x) || ~isscalar(x) || ~isfinite(x)
		out = 'NaN';
	else
		out = sprintf('%d', round(x));
	end
end

function out = format_metric_local(x)
	if ~isnumeric(x) || isempty(x) || ~isscalar(x) || ~isfinite(x)
		out = 'NaN';
	elseif x == 0
		out = '0';
	elseif abs(x) < 1e-3 || abs(x) >= 1e4
		out = sprintf('%.6e', x);
	else
		out = sprintf('%.6g', x);
	end
end
