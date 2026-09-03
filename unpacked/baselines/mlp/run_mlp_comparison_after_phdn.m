function [resultMlp, allResults, summaryRows] = run_mlp_comparison_after_phdn(task, resultPhdn, allResults, summaryRows, iCase, runFlag, mlpOpts)
%RUN_MLP_COMPARISON_AFTER_PHDN Clean one-line MLP integration for demos.
%
% Usage in loop demos:
%   [resultMlp, allResults, summaryRows] = run_mlp_comparison_after_phdn( ...
%       task, result, allResults, summaryRows, iCase, RunMLPBaseline);
%
% Usage in single-case demos:
%   [resultMlp, allResults, summaryRows] = run_mlp_comparison_after_phdn( ...
%       task, result, struct(), struct([]), 1, RunMLPBaseline);

	if nargin < 6 || isempty(runFlag)
		runFlag = true;
	end
	if nargin < 7 || isempty(mlpOpts)
		mlpOpts = make_default_mlp_options_for_demo(1);
	end
	if nargin < 3 || isempty(allResults)
		allResults = struct();
	end
	if nargin < 4
		summaryRows = struct([]);
	end
	if nargin < 5 || isempty(iCase)
		iCase = 1;
	end

	resultMlp = [];

	if runFlag
		resultMlp = run_mlp_baseline_from_phdn_result(resultPhdn, mlpOpts);

		allResults.(task.name).PhDN = resultPhdn;
		allResults.(task.name).MLP = resultMlp;

		if ~isempty(summaryRows) && numel(summaryRows) >= iCase
			summaryRows = add_mlp_fields_to_summary_row(summaryRows, iCase, resultMlp);
		end
	else
		allResults.(task.name).PhDN = resultPhdn;
	end
end
