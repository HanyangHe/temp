function resultSR = make_stage0_sr_ablation_result(resultPhdn)
%MAKE_STAGE0_SR_ABLATION_RESULT Collect the existing PhDN Stage-0 record.
%
% This helper never launches a second SR/SINDy run.  It only converts the
% Stage-0 information already stored in the PhDN result into the common
% baseline-result layout used by the demo summary.
%
% Two routes are represented explicitly:
%   1) native PySR was executed for at least one unresolved output;
%   2) every output passed the fixed single-layer SINDy bypass, so PySR was
%      deliberately skipped.  This route is reported as BYPASS with its
%      finite SINDy metrics instead of N/A/NaN.

	resultSR = empty_result_local();
	if nargin < 1 || ~isstruct(resultPhdn) || isempty(resultPhdn)
		resultSR.reason = 'missing_phdn_result';
		return;
	end

	timeStats = getfield_default_local(resultPhdn, 'timeStats', struct());
	stage0Info = getfield_default_local(resultPhdn, 'stage0', ...
		getfield_default_local(timeStats, 'stage0', struct()));

	% phdnn_identify stores the actual Stage-0 backend result in
	% resultPhdn.stage0.result.  Older versions stored the backend fields
	% directly in resultPhdn.stage0.  Support both layouts.
	stage0Result = stage0Info;
	if isstruct(stage0Info) && isfield(stage0Info, 'result') && ...
			isstruct(stage0Info.result) && ~isempty(fieldnames(stage0Info.result))
		stage0Result = stage0Info.result;
	end

	resultSR.stage0Info = stage0Info;
	resultSR.stage0Result = stage0Result;
	resultSR.usedSingleLayerBypass = logical(first_field_local( ...
		stage0Info, stage0Result, 'usedSingleLayerBypass', false));
	resultSR.usedPerOutputSindyBypass = logical(first_field_local( ...
		stage0Info, stage0Result, 'usedPerOutputSindyBypass', false));
	resultSR.bypassOutputMask = first_field_local( ...
		stage0Info, stage0Result, 'bypassOutputMask', []);

	% Populate metrics before route classification so both PySR and BYPASS
	% records retain the same finite validation/ID/OOD statistics.
	resultSR = populate_common_metrics_local(resultSR, resultPhdn, timeStats, ...
		stage0Info, stage0Result);

	if resultSR.usedSingleLayerBypass
		resultSR.available = true;
		resultSR.searchExecuted = false;
		resultSR.bypassed = true;
		resultSR.status = 'BYPASS';
		resultSR.reason = 'all_outputs_used_stage0_sindy_bypass_no_pysr_search';
		resultSR.method = 'PhDN-Stage0-SR-bypassed-by-SINDy';
		resultSR.reportTitle = 'PhDN Stage-0 SR ablation (SINDy bypass; no extra run)';
		resultSR.structureLabel = format_expression_label_local(resultSR.selectedExpressions);
		return;
	end

	stage0Applied = logical(first_field_local(stage0Info, stage0Result, 'applied', false));
	searchResult = first_field_local(stage0Info, stage0Result, 'searchResult', struct());
	searchExecuted = stage0Applied && isstruct(searchResult) && ...
		~isempty(fieldnames(searchResult));

	if ~searchExecuted
		% Compatibility fallback for older saved results that did not retain
		% searchResult but did retain an explicit PySR method label and finite
		% Stage-0 evaluation metrics.
		methodLabel = lower(char(string(first_field_local( ...
			stage0Info, stage0Result, 'method', ''))));
		searchExecuted = stage0Applied && contains(methodLabel, 'pysr') && ...
			isfinite(getfield_default_local(timeStats, 'stage0IDTestRMSE', NaN));
	end

	if ~searchExecuted
		resultSR.reason = 'stage0_pysr_result_not_available';
		resultSR.status = 'N/A';
		return;
	end

	resultSR.available = true;
	resultSR.searchExecuted = true;
	resultSR.bypassed = false;
	resultSR.status = 'AVAILABLE';
	resultSR.reason = 'collected_from_existing_phdn_stage0_pysr_result';
	resultSR.method = 'PhDN-Stage0-SR';
	resultSR.reportTitle = 'PhDN Stage-0 SR ablation (collected, no extra run)';
	resultSR.structureLabel = format_expression_label_local(resultSR.selectedExpressions);
end

function resultSR = populate_common_metrics_local(resultSR, resultPhdn, timeStats, stage0Info, stage0Result)
	valMSE = getfield_default_local(timeStats, 'stage0ValidationMSE', NaN);
	valRMSE = getfield_default_local(timeStats, 'stage0ValidationRMSE', ...
		sqrt_nonnegative_local(valMSE));
	idMSE = getfield_default_local(timeStats, 'stage0IDTestMSE', NaN);
	idRMSE = getfield_default_local(timeStats, 'stage0IDTestRMSE', ...
		sqrt_nonnegative_local(idMSE));
	oodMSE = getfield_default_local(timeStats, 'stage0OODTestMSE', NaN);
	oodRMSE = getfield_default_local(timeStats, 'stage0OODTestRMSE', ...
		sqrt_nonnegative_local(oodMSE));

	resultSR.valMetrics = struct('mse', valMSE, 'rmse', valRMSE);
	resultSR.testMetrics = struct('mse', idMSE, 'rmse', idRMSE);
	resultSR.oodMetrics = struct('mse', oodMSE, 'rmse', oodRMSE);
	resultSR.trainTime = first_finite_local([ ...
		getfield_default_local(timeStats, 'stage0Time', NaN), ...
		getfield_default_local(stage0Info, 'trainTime', NaN), ...
		getfield_default_local(stage0Result, 'trainTime', NaN)]);
	resultSR.stage0SearchTime = first_finite_local([ ...
		getfield_default_local(stage0Info, 'searchTime', NaN), ...
		getfield_default_local(stage0Result, 'searchTime', NaN)]);
	resultSR.stage0BaseDictionaryTime = first_finite_local([ ...
		getfield_default_local(stage0Info, 'baseDictionaryTime', NaN), ...
		getfield_default_local(stage0Result, 'baseDictionaryTime', NaN)]);
	resultSR.timeStats = struct('total', resultSR.trainTime, ...
		'stage0Total', resultSR.trainTime, ...
		'pysrSearch', resultSR.stage0SearchTime, ...
		'baseDictionary', resultSR.stage0BaseDictionaryTime);

	bestModel = first_field_local(stage0Info, stage0Result, 'bestModel', struct());
	bestValMetrics = getfield_default_local(bestModel,'valMetrics',struct());
	bestTestMetrics = getfield_default_local(bestModel,'testMetrics',struct());
	resultSR.valMetrics.nrmse = getfield_default_local(bestValMetrics,'nrmse',NaN);
	resultSR.testMetrics.nrmse = getfield_default_local(bestTestMetrics,'nrmse',NaN);
	resultSR.nActiveCoefficients = getfield_default_local(bestModel, 'nActiveTerms', ...
		getfield_default_local(bestModel, 'nActiveCoefficients', ...
		getfield_default_local(bestModel, 'complexity', NaN)));
	resultSR.complexity = getfield_default_local(bestModel, 'complexity', NaN);
	resultSR.outputSelections = getfield_default_local(bestModel, 'outputSelections', struct([]));
	searchResult = first_field_local(stage0Info, stage0Result, 'searchResult', struct());
	resultSR.adaptiveRescue = getfield_default_local(bestModel,'adaptiveRescue', ...
		getfield_default_local(searchResult,'adaptiveRescue',struct()));
	resultSR.selectedExpressions = first_field_local(stage0Info, stage0Result, ...
		'bestExpressions', getfield_default_local(resultPhdn, 'stage0Expressions', {}));
	resultSR.selectedModelTrainTime = NaN;
end

function resultSR = empty_result_local()
	resultSR = struct();
	resultSR.available = false;
	resultSR.searchExecuted = false;
	resultSR.bypassed = false;
	resultSR.status = 'N/A';
	resultSR.usedSingleLayerBypass = false;
	resultSR.usedPerOutputSindyBypass = false;
	resultSR.bypassOutputMask = [];
	resultSR.reason = '';
	resultSR.sourceRole = 'phdn_stage0_sr_ablation';
	resultSR.method = 'PhDN-Stage0-SR';
	resultSR.reportRole = 'phdn_stage0_sr_ablation';
	resultSR.reportTitle = 'PhDN Stage-0 SR ablation (collected, no extra run)';
	resultSR.valMetrics = struct('mse', NaN, 'rmse', NaN, 'nrmse', NaN);
	resultSR.testMetrics = struct('mse', NaN, 'rmse', NaN, 'nrmse', NaN);
	resultSR.oodMetrics = struct('mse', NaN, 'rmse', NaN);
	resultSR.trainTime = NaN;
	resultSR.stage0SearchTime = NaN;
	resultSR.stage0BaseDictionaryTime = NaN;
	resultSR.timeStats = struct('total', NaN, 'stage0Total', NaN, ...
		'pysrSearch', NaN, 'baseDictionary', NaN);
	resultSR.nActiveCoefficients = NaN;
	resultSR.complexity = NaN;
	resultSR.outputSelections = struct([]);
	resultSR.adaptiveRescue = struct();
	resultSR.selectedExpressions = {};
	resultSR.structureLabel = '';
	resultSR.selectedModelTrainTime = NaN;
	resultSR.stage0Info = struct();
	resultSR.stage0Result = struct();
end

function value = first_field_local(primary, secondary, name, defaultValue)
	if isstruct(primary) && isfield(primary, name) && ~isempty(primary.(name))
		value = primary.(name);
	elseif isstruct(secondary) && isfield(secondary, name) && ~isempty(secondary.(name))
		value = secondary.(name);
	else
		value = defaultValue;
	end
end

function value = first_finite_local(values)
	value = NaN;
	for i = 1:numel(values)
		if isnumeric(values(i)) && isfinite(values(i))
			value = values(i);
			return;
		end
	end
end

function value = sqrt_nonnegative_local(x)
	if isnumeric(x) && isscalar(x) && isfinite(x) && x >= 0
		value = sqrt(x);
	else
		value = NaN;
	end
end

function label = format_expression_label_local(expressions)
	if isempty(expressions)
		label = '';
		return;
	end
	if ischar(expressions) || isstring(expressions)
		expressions = cellstr(expressions);
	end
	if ~iscell(expressions)
		label = '';
		return;
	end
	parts = cell(1, numel(expressions));
	for i = 1:numel(expressions)
		parts{i} = sprintf('y%d=%s', i, char(string(expressions{i})));
	end
	label = strjoin(parts, '; ');
end

function value = getfield_default_local(s, name, defaultValue)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		value = s.(name);
	else
		value = defaultValue;
	end
end
