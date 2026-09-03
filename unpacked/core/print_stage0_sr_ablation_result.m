function print_stage0_sr_ablation_result(resultSR,compactMode)
%PRINT_STAGE0_SR_ABLATION_RESULT Print the collected Stage-0 SR/bypass record.

	if nargin < 2 || isempty(compactMode); compactMode = false; end

	fprintf('========================================\n');
	fprintf('PhDN Stage-0 SR ablation result\n');
	fprintf('========================================\n');
	fprintf('Execution mode                 : collected from PhDN Stage 0; no extra SR run\n');
	if logical(compactMode); fprintf('Report mode                    : minimal\n'); end

	if nargin < 1 || ~isstruct(resultSR)
		fprintf('Status                         : N/A\n');
		fprintf('Reason                         : missing_result_structure\n');
		fprintf('========================================\n');
		return;
	end

	isAvailable = logical(getfield_default_local(resultSR, 'available', false));
	isBypass = logical(getfield_default_local(resultSR, 'bypassed', false));
	if ~isAvailable
		fprintf('Status                         : N/A\n');
		fprintf('Reason                         : %s\n', ...
			char(string(getfield_default_local(resultSR, 'reason', 'unavailable'))));
		fprintf('========================================\n');
		return;
	end

	if isBypass
		fprintf('Status                         : BYPASS\n');
		fprintf('Stage-0 route                  : all outputs accepted by fixed single-layer SINDy; native PySR skipped\n');
		fprintf('Native PySR search executed    : 0\n');
		mask = getfield_default_local(resultSR, 'bypassOutputMask', []);
		if ~isempty(mask)
			fprintf('Accepted output mask           : [%s]\n', num2str(logical(mask)));
		end
		print_metrics_local(resultSR,compactMode);
		fprintf('Stage-0 SINDy bypass time      : %.3f s\n', ...
			getfield_default_local(resultSR, 'trainTime', NaN));
		fprintf('No additional baseline cost    : yes\n');
		fprintf('========================================\n');
		return;
	end

	fprintf('Status                         : AVAILABLE\n');
	fprintf('Stage-0 route                  : SINDy bypass for accepted outputs + native PySR for unresolved outputs\n');
	fprintf('Native PySR search executed    : 1\n');
	fprintf('Per-output SINDy bypass used   : %d\n', ...
		logical(getfield_default_local(resultSR, 'usedPerOutputSindyBypass', false)));
	rescue = getfield_default_local(resultSR,'adaptiveRescue',struct());
	if isstruct(rescue) && logical(getfield_default_local(rescue,'enabled',false))
		fprintf('Adaptive per-output rescue     : %s | flagged=%d | extra restarts=%d\n', ...
			char(string(getfield_default_local(rescue,'status','unknown'))), ...
			getfield_default_local(rescue,'nFlaggedOutputs',0), ...
			getfield_default_local(rescue,'totalExtraRestarts',0));
	end
	print_metrics_local(resultSR,compactMode);
	fprintf('Stage-0 total time             : %.3f s\n', ...
		getfield_default_local(resultSR, 'trainTime', NaN));
	searchTime = getfield_default_local(resultSR, 'stage0SearchTime', NaN);
	if isfinite(searchTime)
		fprintf('Native PySR search time        : %.3f s\n', searchTime);
	end
	fprintf('No additional baseline cost    : yes\n');
	fprintf('========================================\n');
end

function print_metrics_local(resultSR,compactMode)
	fprintf('Validation MSE/RMSE            : %.6e / %.6e\n', ...
		get_metric_local(resultSR, 'valMetrics', 'mse'), ...
		get_metric_local(resultSR, 'valMetrics', 'rmse'));
	fprintf('In-distribution test MSE/RMSE  : %.6e / %.6e\n', ...
		get_metric_local(resultSR, 'testMetrics', 'mse'), ...
		get_metric_local(resultSR, 'testMetrics', 'rmse'));
	fprintf('OOD test MSE/RMSE              : %.6e / %.6e\n', ...
		get_metric_local(resultSR, 'oodMetrics', 'mse'), ...
		get_metric_local(resultSR, 'oodMetrics', 'rmse'));
	structureLabel = char(string(getfield_default_local(resultSR, 'structureLabel', '')));
	if ~isempty(strtrim(structureLabel))
		if logical(compactMode)
			fprintf('Selected Stage-0 expression    : [omitted in minimal replay; archived in result]\n');
		else
			fprintf('Selected Stage-0 expression    : %s\n', structureLabel);
		end
	end
end

function value = get_metric_local(s, groupName, metricName)
	value = NaN;
	if isstruct(s) && isfield(s, groupName) && isstruct(s.(groupName)) && ...
			isfield(s.(groupName), metricName) && isnumeric(s.(groupName).(metricName)) && ...
			isscalar(s.(groupName).(metricName))
		value = s.(groupName).(metricName);
	end
end

function value = getfield_default_local(s, name, defaultValue)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		value = s.(name);
	else
		value = defaultValue;
	end
end
