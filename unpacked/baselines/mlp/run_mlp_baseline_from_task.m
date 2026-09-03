function resultMlp = run_mlp_baseline_from_task(task, opts, mlpOpts)
%RUN_MLP_BASELINE_FROM_TASK Train MLP baseline without running PhDN.
%
% A shared PhDN-like data result is generated first, then the normal MLP
% baseline entry point is reused.  This keeps baseline-only and PhDN-enabled
% runs consistent.

	if nargin < 2 || isempty(opts)
		opts = phdnn_default_options(task);
	end
	if nargin < 3 || isempty(mlpOpts)
		mlpOpts = make_default_mlp_options_for_demo(1);
	end

	dataResult = make_baseline_data_result_from_task(task, opts);
	resultMlp = run_mlp_baseline_from_phdn_result(dataResult, mlpOpts);
	resultMlp.dataSource = 'Task-generated split (PhDN skipped)';
	if ~isfield(resultMlp, 'timeStats') || isempty(resultMlp.timeStats)
		resultMlp.timeStats = struct();
	end
	resultMlp.timeStats.dataGenerationTime = dataResult.timeStats.dataGenerationTime;
	resultMlp.timeStats.splitTime = dataResult.timeStats.splitTime;
	resultMlp.timeStats.oodDataGenerationTime = dataResult.timeStats.oodDataGenerationTime;
end
