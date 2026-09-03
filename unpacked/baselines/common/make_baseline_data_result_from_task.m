function dataResult = make_baseline_data_result_from_task(task, opts)
%MAKE_BASELINE_DATA_RESULT_FROM_TASK Generate one shared data split for baseline-only runs.
%
% The returned struct mimics the subset of a PhDN result needed by baseline
% entry points, so MLP, SINDy, and SR can be run independently from PhDN while
% still using the same train/validation/test/OOD split.

	if nargin < 2 || isempty(opts)
		opts = phdnn_default_options(task);
	end

	if isfield(task, 'domain') && ~isempty(task.domain)
		if isfield(task, 'variableNames')
			task.domain = normalize_task_domain(task.domain, task.nx, task.variableNames);
		else
			task.domain = normalize_task_domain(task.domain, task.nx);
		end
	end

	if isfield(task, 'domain') && ~isempty(task.domain)
		if ~isfield(opts, 'training') || isempty(opts.training)
			opts.training = struct();
		end
		if ~isfield(opts, 'init') || isempty(opts.init)
			opts.init = struct();
		end
		if ~isfield(opts.init, 'domainFilter') || isempty(opts.init.domainFilter)
			opts.init.domainFilter = struct();
		end
		opts.training.inputDomain = task.domain;
		opts.init.domainFilter.inputDomain = task.domain;
	end

	tData = tic;
	[X, Y] = sample_task_data(task, opts.data.nSamples);
	dataGenerationTime = toc(tData);

	tSplit = tic;
	[Xtr, Ytr, Xval, Yval, Xte, Yte] = split_train_val_test( ...
		X, Y, opts.data.ratioTrain, opts.data.ratioVal);
	splitTime = toc(tSplit);

	XOod = [];
	YOod = [];
	oodDomain = [];
	oodDataGenerationTime = 0;
	if isfield(opts, 'ood') && isfield(opts.ood, 'enable') && opts.ood.enable
		tOod = tic;
		oodDomain = make_ood_domain(task, opts);
		[XOod, YOod] = sample_task_data(task, opts.ood.nSamples, oodDomain);
		oodDataGenerationTime = toc(tOod);
	end

	dataResult = struct();
	dataResult.method = 'baseline_shared_task_data';
	dataResult.task = task;
	dataResult.arch = task.arch;
	dataResult.bestOpArgPolyOrder = resolve_op_arg_poly_order_local(task, opts);
	dataResult.data = struct();
	dataResult.data.Xtr = Xtr;
	dataResult.data.Ytr = Ytr;
	dataResult.data.Xval = Xval;
	dataResult.data.Yval = Yval;
	dataResult.data.Xte = Xte;
	dataResult.data.Yte = Yte;
	dataResult.data.Xood = XOod;
	dataResult.data.Yood = YOod;
	dataResult.data.nTrain = size(Xtr, 1);
	dataResult.data.nVal = size(Xval, 1);
	dataResult.data.nTest = size(Xte, 1);
	dataResult.data.nOod = size(XOod, 1);
	if ~isempty(oodDomain)
		dataResult.data.oodDomain = oodDomain;
	end
	dataResult.timeStats = struct();
	dataResult.timeStats.dataGenerationTime = dataGenerationTime;
	dataResult.timeStats.splitTime = splitTime;
	dataResult.timeStats.oodDataGenerationTime = oodDataGenerationTime;
end

function opArgPolyOrder = resolve_op_arg_poly_order_local(task, opts)
	opArgPolyOrder = [];
	if isfield(opts, 'training') && isfield(opts.training, 'opArgPolyOrderList') && ...
			~isempty(opts.training.opArgPolyOrderList)
		opArgPolyOrder = opts.training.opArgPolyOrderList(1);
	elseif isfield(task, 'training') && isfield(task.training, 'opArgPolyOrderList') && ...
			~isempty(task.training.opArgPolyOrderList)
		opArgPolyOrder = task.training.opArgPolyOrderList(1);
	elseif isfield(task, 'arch') && isfield(task.arch, 'interact') && ...
			isfield(task.arch.interact, 'opArgPolyOrder') && ~isempty(task.arch.interact.opArgPolyOrder)
		opArgPolyOrder = task.arch.interact.opArgPolyOrder;
	elseif isfield(task, 'arch') && isfield(task.arch, 'polyOrder') && ~isempty(task.arch.polyOrder)
		opArgPolyOrder = task.arch.polyOrder;
	else
		opArgPolyOrder = 1;
	end
	opArgPolyOrder = max(1, round(opArgPolyOrder));
end
