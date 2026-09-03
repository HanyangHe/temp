function summaryRows=add_kan_fields_to_summary_row(summaryRows,iCase,resultKan)
%ADD_KAN_FIELDS_TO_SUMMARY_ROW Add KAN metrics to a legacy summary row.
    summaryRows(iCase).kanIdNRMSE=get_nested_local(resultKan,{'testMetrics','nrmse'},NaN);
    summaryRows(iCase).kanIdNMAE=get_nested_local(resultKan,{'testMetrics','nmae'},NaN);
    summaryRows(iCase).kanOodNRMSE=get_nested_local(resultKan,{'oodMetrics','nrmse'},NaN);
    summaryRows(iCase).kanOodNMAE=get_nested_local(resultKan,{'oodMetrics','nmae'},NaN);
    summaryRows(iCase).kanTrainTime=getfield_default_local(resultKan,'trainTime',NaN);
end
function v=get_nested_local(s,p,d);v=s;for k=1:numel(p);if isstruct(v)&&isfield(v,p{k});v=v.(p{k});else;v=d;return;end;end;end
function v=getfield_default_local(s,n,d);if isstruct(s)&&isfield(s,n);v=s.(n);else;v=d;end;end
