function resultEql = run_eql_baseline_from_task(task, phdnOpts, eqlOpts)
%RUN_EQL_BASELINE_FROM_TASK Generate one shared split and train EQL.
    if nargin < 2 || isempty(phdnOpts); phdnOpts = phdnn_default_options(); end
    if nargin < 3 || isempty(eqlOpts); eqlOpts = eql_default_options(); end
    baselineDataResult = make_baseline_data_result_from_task(task, phdnOpts);
    resultEql = run_eql_baseline_from_phdn_result(baselineDataResult, eqlOpts);
end
