function summaryRows = add_mlp_fields_to_summary_row(summaryRows, iCase, resultMlp)
%ADD_MLP_FIELDS_TO_SUMMARY_ROW Add MLP metrics to an existing summaryRows array.

	summaryRows(iCase).mlpIdNRMSE = get_nested_field_with_default_local(resultMlp, {'testMetrics', 'nrmse'}, NaN);
	summaryRows(iCase).mlpIdNMAE = get_nested_field_with_default_local(resultMlp, {'testMetrics', 'nmae'}, NaN);
	summaryRows(iCase).mlpOodNRMSE = get_nested_field_with_default_local(resultMlp, {'oodMetrics', 'nrmse'}, NaN);
	summaryRows(iCase).mlpOodNMAE = get_nested_field_with_default_local(resultMlp, {'oodMetrics', 'nmae'}, NaN);
	summaryRows(iCase).mlpTrainTime = getfield_with_default_local(resultMlp, 'trainTime', NaN);
end

function val = get_nested_field_with_default_local(s, names, defaultVal)
	val = s;
	for k = 1:numel(names)
		if isstruct(val) && isfield(val, names{k})
			val = val.(names{k});
		else
			val = defaultVal;
			return;
		end
	end
end

function val = getfield_with_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name)
		val = s.(name);
	else
		val = defaultVal;
	end
end
