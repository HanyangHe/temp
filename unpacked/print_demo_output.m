function print_demo_output(task, result)
%PRINT_DEMO_OUTPUT Standard demo output for true/surrogate PhDN runs.

	displayOptions = struct();
	displayOptions.showNRMSE = false;
	displayOptions.showSelectedTerms = true;
	displayOptions.selectedTermMaxRows = Inf;
	displayOptions.selectedTermSortBy = 'block';

	modelLabel = 'final-operator';
	if isfield(result, 'modelOperatorMode') && ~isempty(result.modelOperatorMode)
		modelLabel = result.modelOperatorMode;
	end

	fprintf('\nDemo finished for task: %s\n', task.name);
	fprintf('Final model operator mode: %s\n', modelLabel);
	fprintf('Best validation MSE = %.6e\n', result.bestValidationMSE);

	if isfield(result, 'symbolic') && ~isempty(result.symbolic)
		fprintf('\nSymbolic expressions:\n');

		if isfield(result.symbolic, 'reference') && ~isempty(result.symbolic.reference)
			fprintf('  Reference system:\n');
			disp(result.symbolic.reference);
		elseif isfield(result.symbolic, 'referenceMessage') && ~isempty(result.symbolic.referenceMessage)
			fprintf('  Reference system: %s\n', result.symbolic.referenceMessage);
		else
			fprintf('  Reference system: not available for this task.\n');
		end

		if isfield(result.symbolic, 'identified') && ~isempty(result.symbolic.identified)
			fprintf('  Identified PhDN model:\n');
			disp(vpa(result.symbolic.identified, 3));
		elseif isfield(result.symbolic, 'identifiedMessage') && ~isempty(result.symbolic.identifiedMessage)
			fprintf('  Identified PhDN model: %s\n', result.symbolic.identifiedMessage);
		elseif isfield(result.symbolic, 'skipReason') && ~isempty(result.symbolic.skipReason)
			fprintf('  Identified PhDN model: skipped (%s).\n', result.symbolic.skipReason);
		else
			fprintf('  Identified PhDN model: not available.\n');
		end
	end

	fprintf('\nIn-distribution test results:\n');
	if isfield(result, 'testMetrics') && ~isempty(result.testMetrics)
		print_metric_line_local(sprintf('%s PhDN Test', modelLabel), result.testMetrics, displayOptions);
	else
		fprintf('  %s PhDN Test MSE = %.6e, RMSE = %.6e\n', ...
			modelLabel, result.testMSE, result.testRMSE);
	end

	if isfield(result, 'oodTestMetrics') && ~isempty(result.oodTestMetrics) && isfinite(result.oodTestMetrics.mse)
		fprintf('\nOut-of-distribution test results:\n');
		print_metric_line_local(sprintf('OOD %s PhDN Test', modelLabel), result.oodTestMetrics, displayOptions);
	elseif isfield(result, 'oodTestMSE') && ~isempty(result.oodTestMSE) && isfinite(result.oodTestMSE)
		fprintf('\nOut-of-distribution test results:\n');
		fprintf('  OOD %s PhDN Test MSE = %.6e, RMSE = %.6e\n', ...
			modelLabel, result.oodTestMSE, result.oodTestRMSE);
	else
		fprintf('\nOOD test results are not available. Check opts.ood.enable and make_ood_domain settings.\n');
	end

	if isfield(displayOptions, 'showSelectedTerms') && displayOptions.showSelectedTerms && ...
			isfield(result, 'selectedTerms') && ~isempty(result.selectedTerms)
		print_selected_terms(result.selectedTerms, ...
			'MaxRows', displayOptions.selectedTermMaxRows, ...
			'SortBy', displayOptions.selectedTermSortBy);
	end
end

function print_metric_line_local(label, metrics, displayOptions)
	if displayOptions.showNRMSE && isfield(metrics, 'nrmse')
		fprintf('  %s MSE = %.6e, RMSE = %.6e, NRMSE = %.6e\n', ...
			label, metrics.mse, metrics.rmse, metrics.nrmse);
	else
		fprintf('  %s MSE = %.6e, RMSE = %.6e\n', ...
			label, metrics.mse, metrics.rmse);
	end
end
