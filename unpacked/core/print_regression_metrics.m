function print_regression_metrics(prefix, metrics)
%PRINT_REGRESSION_METRICS Print regression metrics in a compact format.
%
% NRMSE is still computed and stored in metrics.nrmse, but it is hidden by
% default because std-normalized errors can be misleading for nearly flat
% output distributions. Set displayOptions.showNRMSE = true here if needed.

	displayOptions = struct();
	displayOptions.showNRMSE = false;

	if nargin < 1 || isempty(prefix)
		prefix = 'Test';
	end

	if displayOptions.showNRMSE && isfield(metrics, 'nrmse')
		fprintf('%s MSE/RMSE/NRMSE = %.6e / %.6e / %.6e\n', ...
			prefix, metrics.mse, metrics.rmse, metrics.nrmse);
	else
		fprintf('%s MSE/RMSE = %.6e / %.6e\n', ...
			prefix, metrics.mse, metrics.rmse);
	end

	if isfield(metrics, 'nmae')
		fprintf('%s MAE/NMAE = %.6e / %.6e\n', ...
			prefix, metrics.mae, metrics.nmae);
	else
		fprintf('%s MAE = %.6e\n', prefix, metrics.mae);
	end
end
