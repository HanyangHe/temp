function metrics = compute_regression_metrics(Ypred, Ytrue)
%COMPUTE_REGRESSION_METRICS Compute absolute and normalized regression errors.
%
% The main normalized metric is the mean per-output NRMSE:
%
%   NRMSE_j = RMSE_j / scale_j,
%   metrics.nrmse = mean_j NRMSE_j.
%
% This is more comparable across cases and across multi-output systems than
% raw MSE/RMSE. The output scale is estimated from std(Ytrue(:,j)). If the
% standard deviation is degenerate, the function falls back to range, then
% RMS magnitude, then 1.
%
% Inputs:
%   Ypred, Ytrue: N x ny matrices.
%
% Output:
%   metrics: struct containing MSE/RMSE/MAE and normalized variants.

	if nargin < 2
		error('compute_regression_metrics requires Ypred and Ytrue.');
	end

	if ~isequal(size(Ypred), size(Ytrue))
		error('Ypred and Ytrue must have the same size.');
	end

	if ~isreal(Ypred) || ~isreal(Ytrue) || any(~isfinite(Ypred(:))) || any(~isfinite(Ytrue(:)))
		metrics = invalid_metrics_local(size(Ytrue, 2));
		return;
	end

	E = Ypred - Ytrue;
	if ~isreal(E) || any(~isfinite(E(:)))
		metrics = invalid_metrics_local(size(Ytrue, 2));
		return;
	end

	metrics.mse = mean(E(:).^2);
	if ~isfinite(metrics.mse) || metrics.mse < 0
		metrics = invalid_metrics_local(size(Ytrue, 2));
		return;
	end
	metrics.rmse = sqrt(metrics.mse);
	metrics.mae = mean(abs(E(:)));
	metrics.maxAbsError = max(abs(E(:)));

	metrics.mseByOutput = mean(E.^2, 1);
	metrics.mseByOutput(~isfinite(metrics.mseByOutput) | metrics.mseByOutput < 0) = Inf;
	metrics.rmseByOutput = sqrt(metrics.mseByOutput);
	metrics.maeByOutput = mean(abs(E), 1);
	metrics.maxAbsErrorByOutput = max(abs(E), [], 1);

	scale = std(Ytrue, 0, 1);

	rangeScale = max(Ytrue, [], 1) - min(Ytrue, [], 1);
	rmsScale = sqrt(mean(Ytrue.^2, 1));

	bad = ~isfinite(scale) | scale < 1e-12;
	scale(bad) = rangeScale(bad);

	bad = ~isfinite(scale) | scale < 1e-12;
	scale(bad) = rmsScale(bad);

	bad = ~isfinite(scale) | scale < 1e-12;
	scale(bad) = 1;

	metrics.scaleByOutput = scale;

	metrics.nrmseByOutput = metrics.rmseByOutput ./ scale;
	metrics.nmaeByOutput = metrics.maeByOutput ./ scale;
	metrics.nmaxAbsErrorByOutput = metrics.maxAbsErrorByOutput ./ scale;

	metrics.nrmse = mean(metrics.nrmseByOutput);
	metrics.nmae = mean(metrics.nmaeByOutput);
	metrics.nmaxAbsError = mean(metrics.nmaxAbsErrorByOutput);

	globalScale = std(Ytrue(:));
	if ~isfinite(globalScale) || globalScale < 1e-12
		globalScale = max(Ytrue(:)) - min(Ytrue(:));
	end
	if ~isfinite(globalScale) || globalScale < 1e-12
		globalScale = sqrt(mean(Ytrue(:).^2));
	end
	if ~isfinite(globalScale) || globalScale < 1e-12
		globalScale = 1;
	end

	metrics.globalScale = globalScale;
	metrics.globalNRMSE = metrics.rmse / globalScale;
	metrics.globalNMAE = metrics.mae / globalScale;
end


function metrics = invalid_metrics_local(ny)
	metrics = struct();
	metrics.mse = Inf;
	metrics.rmse = Inf;
	metrics.mae = Inf;
	metrics.maxAbsError = Inf;
	metrics.mseByOutput = Inf(1, ny);
	metrics.rmseByOutput = Inf(1, ny);
	metrics.maeByOutput = Inf(1, ny);
	metrics.maxAbsErrorByOutput = Inf(1, ny);
	metrics.scaleByOutput = ones(1, ny);
	metrics.nrmseByOutput = Inf(1, ny);
	metrics.nmaeByOutput = Inf(1, ny);
	metrics.nmaxAbsErrorByOutput = Inf(1, ny);
	metrics.nrmse = Inf;
	metrics.nmae = Inf;
	metrics.nmaxAbsError = Inf;
	metrics.globalScale = 1;
	metrics.globalNRMSE = Inf;
	metrics.globalNMAE = Inf;
end
