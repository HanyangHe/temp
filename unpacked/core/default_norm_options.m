function normOpt = default_norm_options()
%DEFAULT_NORM_OPTIONS Default input/output normalization settings.
%
% The MLP-surrogate route can use MATLAB Neural Network Toolbox style
% mapminmax preprocessing: statistics are fitted on the training set only,
% inputs/targets are mapped to [-1,1], and predictions are mapped back to
% the original output scale for validation/test/OOD metrics.

	normOpt = struct();
	normOpt.useInputNorm = false;
	normOpt.useOutputNorm = false;
	normOpt.useLayerNorm = false;
	normOpt.style = 'mapminmax';       % 'mapminmax' | 'zscore' | 'none'
	normOpt.ymin = -1;
	normOpt.ymax = 1;
	normOpt.epsNorm = 1e-12;

	% z-score fields, kept for backward compatibility.
	normOpt.muX = [];
	normOpt.sigX = [];
	normOpt.muY = [];
	normOpt.sigY = [];

	% mapminmax fields.
	normOpt.xOffset = [];
	normOpt.xGain = [];
	normOpt.xIsConstant = [];
	normOpt.yOffset = [];
	normOpt.yGain = [];
	normOpt.yIsConstant = [];
end
