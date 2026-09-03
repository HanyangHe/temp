function result = train_mlp_regression_baseline(XTrain, YTrain, XVal, YVal, XTest, YTest, XOod, YOod, mlpOpts)
%TRAIN_MLP_REGRESSION_BASELINE Train a MATLAB fitnet MLP regression baseline.
%
% Inputs use row-sample format:
%   XTrain: Ntrain x nx
%   YTrain: Ntrain x ny
%   XVal  : Nval   x nx
%   YVal  : Nval   x ny
%   XTest : Ntest  x nx
%   YTest : Ntest  x ny
%   XOod  : Nood   x nx, optional
%   YOod  : Nood   x ny, optional

	if nargin < 9 || isempty(mlpOpts)
		mlpOpts = mlp_default_options();
	end
	if nargin < 7
		XOod = [];
		YOod = [];
	end

	if exist('fitnet', 'file') ~= 2
		error(['fitnet was not found. Please install or enable MATLAB Neural ', ...
			'Network Toolbox / Deep Learning Toolbox.']);
	end

	validate_data_local(XTrain, YTrain, 'train');
	validate_data_local(XVal, YVal, 'validation');
	validate_data_local(XTest, YTest, 'test');

	hasOod = ~isempty(XOod) && ~isempty(YOod);
	if hasOod
		validate_data_local(XOod, YOod, 'OOD');
	end

	rng(getfield_with_default_local(mlpOpts, 'seed', 1));

	XAll = [XTrain; XVal; XTest];
	YAll = [YTrain; YVal; YTest];

	nTrain = size(XTrain, 1);
	nVal = size(XVal, 1);
	nTest = size(XTest, 1);

	idxTrain = 1:nTrain;
	idxVal = nTrain + (1:nVal);
	idxTest = nTrain + nVal + (1:nTest);

	net = fitnet(mlpOpts.hiddenLayerSizes, mlpOpts.trainFcn);
	net.performFcn = getfield_with_default_local(mlpOpts, 'performFcn', 'mse');

	if isfield(mlpOpts, 'hiddenTransferFcn') && ~isempty(mlpOpts.hiddenTransferFcn)
		for k = 1:numel(net.layers)-1
			net.layers{k}.transferFcn = mlpOpts.hiddenTransferFcn;
		end
	end
	if isfield(mlpOpts, 'outputTransferFcn') && ~isempty(mlpOpts.outputTransferFcn)
		net.layers{end}.transferFcn = mlpOpts.outputTransferFcn;
	end

	if isfield(mlpOpts, 'inputProcessFcns')
		net.inputs{1}.processFcns = mlpOpts.inputProcessFcns;
	end
	if isfield(mlpOpts, 'outputProcessFcns')
		net.outputs{end}.processFcns = mlpOpts.outputProcessFcns;
	end

	net.divideFcn = 'divideind';
	net.divideParam.trainInd = idxTrain;
	net.divideParam.valInd = idxVal;
	net.divideParam.testInd = idxTest;

	if isfield(mlpOpts, 'trainParam')
		fn = fieldnames(mlpOpts.trainParam);
		for k = 1:numel(fn)
			net.trainParam.(fn{k}) = mlpOpts.trainParam.(fn{k});
		end
	end

	tStart = tic;
	[net, tr] = train(net, XAll.', YAll.');
	trainTime = toc(tStart);

	tEval = tic;
	YTrainPred = net(XTrain.').';
	YValPred = net(XVal.').';
	YTestPred = net(XTest.').';
	idEvalTime = toc(tEval);

	result = struct();
	result.method = 'MLP-fitnet';
	result.net = net;
	result.trainingRecord = tr;
	result.trainTime = trainTime;
	result.idEvalTime = idEvalTime;
	result.hiddenLayerSizes = mlpOpts.hiddenLayerSizes;
	result.trainFcn = mlpOpts.trainFcn;

	result.trainMetrics = regression_metrics_local(YTrain, YTrainPred);
	result.valMetrics = regression_metrics_local(YVal, YValPred);
	result.testMetrics = regression_metrics_local(YTest, YTestPred);

	result.YTrainPred = YTrainPred;
	result.YValPred = YValPred;
	result.YTestPred = YTestPred;

	if hasOod
		tOod = tic;
		YOodPred = net(XOod.').';
		result.oodEvalTime = toc(tOod);
		result.oodMetrics = regression_metrics_local(YOod, YOodPred);
		result.YOodPred = YOodPred;
	else
		result.oodEvalTime = NaN;
		result.oodMetrics = empty_metrics_local();
		result.YOodPred = [];
	end
end

function validate_data_local(X, Y, name)
	if isempty(X) || isempty(Y)
		error('%s data is empty.', name);
	end
	if size(X, 1) ~= size(Y, 1)
		error('%s X/Y sample count mismatch.', name);
	end
	if any(~isfinite(X(:))) || any(~isfinite(Y(:)))
		error('%s data contains NaN or Inf values.', name);
	end
end

function metrics = regression_metrics_local(Y, Yhat)
	E = Yhat - Y;

	metrics = struct();
	metrics.mse = mean(E(:).^2);
	metrics.rmse = sqrt(metrics.mse);
	metrics.mae = mean(abs(E(:)));

	yStd = std(Y(:));
	yRange = max(Y(:)) - min(Y(:));
	yMeanAbs = mean(abs(Y(:)));

	if yStd > 0
		metrics.nrmse = metrics.rmse / yStd;
	else
		metrics.nrmse = NaN;
	end

	if yRange > 0
		metrics.nrmseRange = metrics.rmse / yRange;
	else
		metrics.nrmseRange = NaN;
	end

	if yMeanAbs > 0
		metrics.nmae = metrics.mae / yMeanAbs;
	else
		metrics.nmae = NaN;
	end
end

function metrics = empty_metrics_local()
	metrics = struct();
	metrics.mse = NaN;
	metrics.rmse = NaN;
	metrics.mae = NaN;
	metrics.nrmse = NaN;
	metrics.nrmseRange = NaN;
	metrics.nmae = NaN;
end

function val = getfield_with_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end
