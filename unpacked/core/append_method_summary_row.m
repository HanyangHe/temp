function [summaryRows, iSummary] = append_method_summary_row(summaryRows, iSummary, caseName, methodName, methodResult, roundIndex)
%APPEND_METHOD_SUMMARY_ROW Append one method-level row to the demo summary.
%
% This helper gives PhDN and all baselines the same summary status.  Each
% enabled method contributes one row, so baseline-only runs do not end with a
% PhDN-only NaN summary.

	if nargin < 6 || isempty(roundIndex)
		roundIndex = NaN;
	end
	if nargin < 5
		methodResult = struct();
	end
	if nargin < 4 || isempty(methodName)
		methodName = 'unknown';
	end
	if nargin < 3 || isempty(caseName)
		caseName = '';
	end

	iSummary = iSummary + 1;
	summaryRows(iSummary).baseCaseName = caseName;
	summaryRows(iSummary).roundIndex = roundIndex;
	if isnumeric(roundIndex) && isscalar(roundIndex) && isfinite(roundIndex)
		summaryRows(iSummary).caseName = sprintf('%s [R%d]', caseName, round(roundIndex));
	else
		summaryRows(iSummary).caseName = caseName;
	end
	summaryRows(iSummary).method = methodName;
	methodFamily = method_family_local(methodName,methodResult);
	summaryRows(iSummary).methodFamily = methodFamily;
	summaryRows(iSummary).validationMSE = extract_validation_mse_local(methodFamily, methodResult);
	summaryRows(iSummary).activeCoefficients = extract_active_count_local(methodFamily, methodResult);
	summaryRows(iSummary).idRMSE = extract_id_rmse_local(methodFamily, methodResult);
	summaryRows(iSummary).oodRMSE = extract_ood_rmse_local(methodFamily, methodResult);
	summaryRows(iSummary).stage0ValidationMSE = extract_stage0_metric_local(methodFamily, methodResult, 'stage0ValidationMSE');
	summaryRows(iSummary).stage0IDRMSE = extract_stage0_metric_local(methodFamily, methodResult, 'stage0IDTestRMSE');
	summaryRows(iSummary).stage0OODRMSE = extract_stage0_metric_local(methodFamily, methodResult, 'stage0OODTestRMSE');
	summaryRows(iSummary).trainTime = extract_train_time_local(methodFamily, methodResult);
	summaryRows(iSummary).stage0Time = extract_nested_local(methodResult, {'timeStats','stage0Time'}, NaN);
	summaryRows(iSummary).stage1Time = extract_nested_local(methodResult, {'timeStats','stage1Time'}, NaN);
	summaryRows(iSummary).stage2Time = extract_nested_local(methodResult, {'timeStats','lsqTime'}, NaN);
	summaryRows(iSummary).structureLabel = extract_text_local(methodResult, 'structureLabel', '');
	summaryRows(iSummary).available = extract_logical_local(methodResult, 'available', true);
	summaryRows(iSummary).statusReason = extract_text_local(methodResult, 'reason', '');
	summaryRows(iSummary).sourceRole = extract_text_local(methodResult, 'sourceRole', '');
	summaryRows(iSummary).selectedModelTime = extract_nested_local(methodResult, {'selectedModelTrainTime'}, NaN);
end

function val = extract_validation_mse_local(methodName, r)
	if strcmpi(methodName, 'phdn')
		val = first_finite_local(r, { ...
			{'bestValidationMSE'}, {'valMetrics','mse'}, ...
			{'timeStats','stage0ValidationMSE'}});
	else
		val = extract_nested_local(r, {'valMetrics','mse'}, NaN);
	end
end

function val = extract_active_count_local(methodName, r)
	if strcmpi(methodName, 'phdn')
		val = extract_nested_local(r, {'nActiveFinal'}, NaN);
	elseif strcmpi(methodName, 'eql-div')
		% EQL's published Vint-S complexity is connected functional units,
		% not the number of scalar nonzero weights.
		val = extract_nested_local(r, {'nActiveUnits'}, NaN);
		if ~isfinite(val)
			val = extract_nested_local(r, {'nActiveCoefficients'}, NaN);
		end
	else
		val = extract_nested_local(r, {'nActiveCoefficients'}, NaN);
	end
end

function val = extract_id_rmse_local(methodName, r)
	if strcmpi(methodName, 'phdn')
		val = first_finite_local(r, { ...
			{'testMetrics','rmse'}, {'physicalTestMetrics','rmse'}, ...
			{'physicalTestRMSE'}, {'testRMSE'}, ...
			{'timeStats','stage0IDTestRMSE'}});
	else
		val = extract_nested_local(r, {'testMetrics','rmse'}, NaN);
	end
end

function val = extract_ood_rmse_local(methodName, r)
	if strcmpi(methodName, 'phdn')
		val = first_finite_local(r, { ...
			{'oodTestMetrics','rmse'}, {'oodPhysicalTestMetrics','rmse'}, ...
			{'oodPhysicalTestRMSE'}, {'oodTestRMSE'}, ...
			{'timeStats','stage0OODTestRMSE'}});
	else
		val = extract_nested_local(r, {'oodMetrics','rmse'}, NaN);
	end
end

function val = extract_stage0_metric_local(methodName, r, fieldName)
	if strcmpi(methodName, 'phdn')
		val = extract_nested_local(r, {'timeStats', fieldName}, NaN);
	else
		val = NaN;
	end
end

function val = extract_train_time_local(methodName, r)
	if strcmpi(methodName, 'phdn')
		stage0 = extract_nested_local(r, {'timeStats','stage0Time'}, NaN);
		stage1 = extract_nested_local(r, {'timeStats','stage1Time'}, NaN);
		stage2 = extract_nested_local(r, {'timeStats','lsqTime'}, NaN);
		parts = [stage0 stage1 stage2];
		if any(isfinite(parts))
			val = sum(parts(isfinite(parts)));
		else
			val = first_finite_local(r, { ...
				{'timeStats','trainingTimeMaskedLSQ'}, ...
				{'timeStats','trainingWallTime'}, {'trainTime'}});
		end
	else
		val = extract_nested_local(r, {'trainTime'}, NaN);
		if ~isfinite(val)
			val = extract_nested_local(r, {'timeStats','total'}, NaN);
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

function val = first_finite_local(s, paths)
	val = NaN;
	for i = 1:numel(paths)
		candidate = extract_nested_local(s, paths{i}, NaN);
		if isfinite(candidate)
			val = candidate;
			return;
		end
	end
end


function val = extract_logical_local(s, name, defaultVal)
	val = logical(defaultVal);
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name)) && ...
			(islogical(s.(name)) || isnumeric(s.(name))) && isscalar(s.(name))
		val = logical(s.(name));
	end
end

function txt = extract_text_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		txt = char(string(s.(name)));
	else
		txt = defaultVal;
	end
end
function val = extract_nested_local(s, path, defaultVal)
	val = defaultVal;
	cur = s;
	for i = 1:numel(path)
		if ~isstruct(cur) || ~isfield(cur, path{i}) || isempty(cur.(path{i}))
			return;
		end
		cur = cur.(path{i});
	end
	if isnumeric(cur) && isscalar(cur)
		val = cur;
	elseif islogical(cur) && isscalar(cur)
		val = double(cur);
	end
end
