function print_round_statistics_summary(summaryRows)
%PRINT_ROUND_STATISTICS_SUMMARY Compact statistics over repeated demo rounds.
%
% The existing per-run method comparison table is left unchanged. This
% function appends one grouped section per case/method and reports
% mean +/- sample standard deviation together with [minimum, maximum].

	fprintf('\n========================================\n');
	fprintf('Repeated-round aggregate summary\n');
	fprintf('Statistics: mean +/- std [min, max]; finite completed rounds only\n');
	fprintf('========================================\n');

	if isempty(summaryRows)
		fprintf('(no enabled method results)\n');
		fprintf('========================================\n');
		return;
	end

	groupKeys = cell(0,1);
	groupCase = cell(0,1);
	groupMethod = cell(0,1);
	groupIndices = cell(0,1);

	for i = 1:numel(summaryRows)
		caseName = get_text_field_local(summaryRows(i), 'baseCaseName', ...
			get_text_field_local(summaryRows(i), 'caseName', ''));
		methodName = get_text_field_local(summaryRows(i), 'method', '');
		key = [lower(caseName), char(31), lower(methodName)];
		idx = find(strcmp(groupKeys, key), 1, 'first');
		if isempty(idx)
			groupKeys{end+1,1} = key; %#ok<AGROW>
			groupCase{end+1,1} = caseName; %#ok<AGROW>
			groupMethod{end+1,1} = methodName; %#ok<AGROW>
			groupIndices{end+1,1} = i; %#ok<AGROW>
		else
			groupIndices{idx}(end+1) = i; %#ok<AGROW>
		end
	end

	for iGroup = 1:numel(groupKeys)
		idx = groupIndices{iGroup};
		rounds = collect_numeric_field_local(summaryRows(idx), 'roundIndex');
		if isempty(rounds)
			nRounds = numel(idx);
			roundText = sprintf('n=%d', nRounds);
		else
			rounds = unique(round(rounds(:)'));
			nRounds = numel(rounds);
			roundText = sprintf('n=%d, rounds=%s', nRounds, compact_integer_list_local(rounds));
		end

		fprintf('%s / %s (%s)\n', groupCase{iGroup}, groupMethod{iGroup}, roundText);
		print_table_header_local();
		print_stat_row_local('Final',  'ValMSE',   collect_numeric_field_local(summaryRows(idx), 'validationMSE'));
		print_stat_row_local('Final',  'Active',   collect_numeric_field_local(summaryRows(idx), 'activeCoefficients'));
		print_stat_row_local('Final',  'ID_RMSE',  collect_numeric_field_local(summaryRows(idx), 'idRMSE'));
		print_stat_row_local('Final',  'OOD_RMSE', collect_numeric_field_local(summaryRows(idx), 'oodRMSE'));
		if ~strcmpi(groupMethod{iGroup}, 'Stage0-SR')
			% PhDN retains its internal stage diagnostics.  Stage0-SR is already a
			% standalone ablation row, so repeating Stage0 fields there would only
			% create an all-NaN duplicate section.
			print_stat_row_local('Stage0', 'ValMSE',   collect_numeric_field_local(summaryRows(idx), 'stage0ValidationMSE'));
			print_stat_row_local('Stage0', 'ID_RMSE',  collect_numeric_field_local(summaryRows(idx), 'stage0IDRMSE'));
			print_stat_row_local('Stage0', 'OOD_RMSE', collect_numeric_field_local(summaryRows(idx), 'stage0OODRMSE'));
		end
		print_stat_row_local('Time_s', 'Train', collect_numeric_field_local(summaryRows(idx), 'trainTime'));
		if ~strcmpi(groupMethod{iGroup}, 'Stage0-SR')
			print_stat_row_local('Time_s', 'Stage0',   collect_numeric_field_local(summaryRows(idx), 'stage0Time'));
			print_stat_row_local('Time_s', 'Stage1',   collect_numeric_field_local(summaryRows(idx), 'stage1Time'));
			print_stat_row_local('Time_s', 'Stage2',   collect_numeric_field_local(summaryRows(idx), 'stage2Time'));
			print_stat_row_local('Time_s', 'Selected', collect_numeric_field_local(summaryRows(idx), 'selectedModelTime'));
		end
		print_unavailable_reason_local(summaryRows(idx));
		print_structure_frequency_local(summaryRows(idx));
	end
	fprintf('========================================\n');
	print_figure_table_statistics_local(summaryRows);
end



function print_unavailable_reason_local(rows)
	available = true;
	reason = '';
	for i = 1:numel(rows)
		if isfield(rows(i), 'available') && ~logical(rows(i).available)
			available = false;
			if isfield(rows(i), 'statusReason') && ~isempty(rows(i).statusReason)
				reason = char(string(rows(i).statusReason));
			end
			break;
		end
	end
	if ~available
		fprintf('  Status: N/A (%s)\n', reason);
	end
end

function print_figure_table_statistics_local(summaryRows)
% One compact row per case/method using the paper-table statistics requested
% by the demo: ID mean+/-std, OOD mean+/-std, and mean complete method time.
	fprintf('\n========================================\n');
	fprintf('Compact ID/OOD/time statistical table\n');
	fprintf('RMSE format: mean +/- sample std; Time_s: mean over finite rounds\n');
	fprintf('Stage0-SR is collected from the existing PhDN Stage 0 and incurs no extra run.\n');
	fprintf('========================================\n');
	fprintf('%-24s %-12s %-29s %-29s %14s\n', ...
		'System', 'Method', 'ID_RMSE mean+/-std', 'OOD_RMSE mean+/-std', 'MeanTime_s');
	fprintf('%s\n', repmat('-', 1, 116));

	keys = cell(0,1);
	for i = 1:numel(summaryRows)
		caseName = get_text_field_local(summaryRows(i), 'baseCaseName', ...
			get_text_field_local(summaryRows(i), 'caseName', ''));
		methodName = get_text_field_local(summaryRows(i), 'method', '');
		key = [lower(caseName), char(31), lower(methodName)];
		if any(strcmp(keys, key)); continue; end
		keys{end+1,1} = key; %#ok<AGROW>
		idx = false(1, numel(summaryRows));
		for j = 1:numel(summaryRows)
			caseJ = get_text_field_local(summaryRows(j), 'baseCaseName', ...
				get_text_field_local(summaryRows(j), 'caseName', ''));
			methodJ = get_text_field_local(summaryRows(j), 'method', '');
			idx(j) = strcmpi(caseJ, caseName) && strcmpi(methodJ, methodName);
		end
		idVals = collect_numeric_field_local(summaryRows(idx), 'idRMSE');
		oodVals = collect_numeric_field_local(summaryRows(idx), 'oodRMSE');
		timeVals = collect_numeric_field_local(summaryRows(idx), 'trainTime');
		fprintf('%-24s %-12s %-29s %-29s %14s\n', caseName, methodName, ...
			format_mean_std_local(idVals), format_mean_std_local(oodVals), ...
			format_num_local(mean_or_nan_local(timeVals)));
	end
	fprintf('========================================\n');
end

function txt = format_mean_std_local(vals)
	vals = vals(isfinite(vals));
	if isempty(vals)
		txt = 'N/A';
		return;
	end
	mu = mean(vals);
	if numel(vals) > 1; sigma = std(vals, 0); else; sigma = 0; end
	txt = sprintf('%s +/- %s', format_num_local(mu), format_num_local(sigma));
end

function value = mean_or_nan_local(vals)
	vals = vals(isfinite(vals));
	if isempty(vals); value = NaN; else; value = mean(vals); end
end

function print_structure_frequency_local(rows)
	labels = cell(0,1);
	for i = 1:numel(rows)
		if isstruct(rows(i)) && isfield(rows(i), 'structureLabel') && ~isempty(rows(i).structureLabel)
			labels{end+1,1} = char(string(rows(i).structureLabel)); %#ok<AGROW>
		end
	end
	if isempty(labels); return; end
	u = cell(0,1);
	counts = zeros(0,1);
	for i = 1:numel(labels)
		j = find(strcmp(u, labels{i}), 1, 'first');
		if isempty(j)
			u{end+1,1} = labels{i}; %#ok<AGROW>
			counts(end+1,1) = 1; %#ok<AGROW>
		else
			counts(j) = counts(j) + 1;
		end
	end
	fprintf('  Selected structures:\n');
	for i = 1:numel(u)
		fprintf('    %d/%d  %s\n', counts(i), numel(labels), u{i});
	end
end

function vals = collect_numeric_field_local(rows, name)
	vals = [];
	for i = 1:numel(rows)
		if isstruct(rows(i)) && isfield(rows(i), name) && ...
				isnumeric(rows(i).(name)) && isscalar(rows(i).(name)) && isfinite(rows(i).(name))
			vals(end+1) = double(rows(i).(name)); %#ok<AGROW>
		end
	end
end

function print_table_header_local()
	fprintf('  %-8s %-12s %4s %14s %14s %14s %14s\n', ...
		'Section', 'Metric', 'N', 'Mean', 'Std', 'Min', 'Max');
	fprintf('  %s\n', repmat('-', 1, 88));
end

function print_stat_row_local(sectionName, metricName, vals)
	vals = vals(isfinite(vals));
	if isempty(vals)
		n = 0;
		mu = NaN;
		sigma = NaN;
		lo = NaN;
		hi = NaN;
	else
		n = numel(vals);
		mu = mean(vals);
		if n > 1
			sigma = std(vals, 0); % sample standard deviation, N-1 denominator
		else
			sigma = 0;
		end
		lo = min(vals);
		hi = max(vals);
	end
	fprintf('  %-8s %-12s %4d %14s %14s %14s %14s\n', ...
		sectionName, metricName, n, ...
		format_num_local(mu), format_num_local(sigma), ...
		format_num_local(lo), format_num_local(hi));
end

function out = format_num_local(x)
	if ~isnumeric(x) || isempty(x) || ~isfinite(x)
		out = 'NaN';
	elseif x == 0
		% Preserve a mathematically exact zero, but never round a small
		% nonzero error metric (for example 1e-32) down to zero.
		out = '0';
	elseif x == fix(x) && abs(x) < 1e9
		% Integer-valued quantities such as active coefficient counts.
		out = sprintf('%d', x);
	elseif abs(x) >= 1e-3 && abs(x) < 1e4
		out = sprintf('%.5g', x);
	else
		out = sprintf('%.4e', x);
	end
end

function txt = get_text_field_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		txt = char(string(s.(name)));
	else
		txt = defaultVal;
	end
end

function txt = compact_integer_list_local(vals)
	vals = sort(unique(vals(:)'));
	if isempty(vals)
		txt = '-';
		return;
	end
	parts = cell(1, numel(vals));
	for i = 1:numel(vals)
		parts{i} = sprintf('%d', vals(i));
	end
	txt = strjoin(parts, ',');
end
