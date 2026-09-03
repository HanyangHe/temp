function normOpt = fit_norm_options(Xtrain, Ytrain, normSettings, useLayerNorm)
%FIT_NORM_OPTIONS Fit normalization statistics from training data only.
%
% Usage:
%   normOpt = fit_norm_options(Xtrain,Ytrain,true,false)
%   normOpt = fit_norm_options(Xtrain,Ytrain,opts.norm,false)
%
% For mapminmax, each input/output column is mapped by training-set min/max
% to [ymin,ymax], default [-1,1], following the standard Neural Network
% Toolbox preprocessing style. New validation/test/OOD data must reuse the
% training-set offsets/gains stored here.

	if nargin < 3 || isempty(normSettings)
		normSettings = false;
	end
	if nargin < 4 || isempty(useLayerNorm)
		useLayerNorm = false;
	end

	normOpt = default_norm_options();

	if isstruct(normSettings)
		useIOnorm = getfield_default_norm_local(normSettings, 'useInputOutputNorm', false) || ...
			getfield_default_norm_local(normSettings, 'applyToMLPSurrogate', false);
		normOpt.style = getfield_default_norm_local(normSettings, 'style', normOpt.style);
		normOpt.ymin = getfield_default_norm_local(normSettings, 'ymin', normOpt.ymin);
		normOpt.ymax = getfield_default_norm_local(normSettings, 'ymax', normOpt.ymax);
		normOpt.epsNorm = getfield_default_norm_local(normSettings, 'epsNorm', normOpt.epsNorm);
		if isfield(normSettings, 'useLayerNorm') && ~isempty(normSettings.useLayerNorm)
			useLayerNorm = normSettings.useLayerNorm;
		end
	else
		useIOnorm = logical(normSettings);
	end

	normOpt.useInputNorm = useIOnorm;
	normOpt.useOutputNorm = useIOnorm;
	normOpt.useLayerNorm = useLayerNorm;

	% z-score statistics are always filled for residual scaling/debugging.
	normOpt.muX = mean(Xtrain, 1);
	normOpt.sigX = std(Xtrain, 0, 1);
	normOpt.sigX(~isfinite(normOpt.sigX) | normOpt.sigX < normOpt.epsNorm) = 1;
	normOpt.muY = mean(Ytrain, 1);
	normOpt.sigY = std(Ytrain, 0, 1);
	normOpt.sigY(~isfinite(normOpt.sigY) | normOpt.sigY < normOpt.epsNorm) = 1;

	% mapminmax statistics.
	xmin = min(Xtrain, [], 1);
	xmax = max(Xtrain, [], 1);
	xRange = xmax - xmin;
	normOpt.xIsConstant = ~isfinite(xRange) | abs(xRange) < normOpt.epsNorm;
	xRange(normOpt.xIsConstant) = 1;
	normOpt.xOffset = xmin;
	normOpt.xGain = (normOpt.ymax - normOpt.ymin) ./ xRange;
	normOpt.xGain(~isfinite(normOpt.xGain)) = 1;

	yminData = min(Ytrain, [], 1);
	ymaxData = max(Ytrain, [], 1);
	yRange = ymaxData - yminData;
	normOpt.yIsConstant = ~isfinite(yRange) | abs(yRange) < normOpt.epsNorm;
	yRange(normOpt.yIsConstant) = 1;
	normOpt.yOffset = yminData;
	normOpt.yGain = (normOpt.ymax - normOpt.ymin) ./ yRange;
	normOpt.yGain(~isfinite(normOpt.yGain)) = 1;
end

function val = getfield_default_norm_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
