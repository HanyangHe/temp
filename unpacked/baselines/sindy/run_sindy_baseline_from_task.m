function resultSindy = run_sindy_baseline_from_task(task, opts, sindyOpts)
%RUN_SINDY_BASELINE_FROM_TASK Train the single-layer SINDy baseline without running PhDN.
%
% This helper is used when the demo is configured with RunPhDNMainModel=false.
% It generates the same type of train/validation/test/OOD data split as the
% PhDN workflow and then calls train_sindy_single_layer_baseline.  By default
% the SINDy library is an independent broad/general dictionary built directly
% from raw inputs, not a PhDN/SR support mask.

	if nargin < 2 || isempty(opts)
		opts = phdnn_default_options(task);
	end
	if nargin < 3 || isempty(sindyOpts)
		sindyOpts = sindy_default_options();
	end

	% Fair-prior invariant: mirror any enabled Stage-0 SR initial-expression
	% library into the flat SINDy candidate-function dictionary.
	[sindyOpts, stage0GuessSync] = sync_sindy_stage0_initial_guesses( ...
		sindyOpts, opts, task.nx);

	if isfield(task, 'domain') && ~isempty(task.domain)
		if isfield(task, 'variableNames')
			task.domain = normalize_task_domain(task.domain, task.nx, task.variableNames);
		else
			task.domain = normalize_task_domain(task.domain, task.nx);
		end
	end

	% Keep the declared ID domain available to any downstream helper that expects
	% the same fields as the PhDN workflow.
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

	baseArch = task.arch;
	baseArch.nx = task.nx;
	baseArch.ny = task.ny;
	if isfield(opts, 'arch') && ~isempty(opts.arch)
		if isfield(opts.arch, 'dims') && ~isempty(opts.arch.dims)
			baseArch.dims = opts.arch.dims;
		end
		if isfield(opts.arch, 'hiddenDims') && ~isempty(opts.arch.hiddenDims)
			baseArch.hiddenDims = opts.arch.hiddenDims;
		end
		if isfield(opts.arch, 'hiddenWidth') && ~isempty(opts.arch.hiddenWidth)
			baseArch.hiddenWidth = opts.arch.hiddenWidth;
		end
	end
	baseArch.safety = opts.safety;
	baseArch.feasibility = opts.training;

	opArgPolyOrder = resolve_sindy_op_arg_poly_order_local(task, opts, baseArch);
	dictionaryMode = lower(strtrim(char(getfield_default_local(sindyOpts, 'dictionaryMode', 'general'))));
	if strcmpi(dictionaryMode, 'phdn_phi')
		arch = baseArch;
		if ~isfield(arch, 'interact') || isempty(arch.interact)
			arch.interact = struct();
		end
		arch.interact.opArgPolyOrder = opArgPolyOrder;
		arch.operatorMode = 'true';
		arch.sindyLibrarySupport = make_sindy_support_from_phdn_opts_local(opts, arch);
		arch = append_sindy_stage0_initial_guesses_to_arch( ...
			arch, task.nx, sindyOpts.stage0InitialGuessTerms);
	else
		% Independent SINDy baseline: build its own broad/general flat dictionary.
		arch = make_sindy_general_arch(task, sindyOpts, baseArch);
		arch.sindyLibrarySupport = struct('keepRows', [], 'source', 'independent_general_sindy_dictionary', ...
			'reason', 'SINDy dictionary is generated independently; no PhDN support is used');
		sindyOpts.usePhdnDictionarySupport = false;
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

	resultSindy = train_sindy_single_layer_baseline( ...
		Xtr, Ytr, Xval, Yval, Xte, Yte, XOod, YOod, arch, sindyOpts);

	resultSindy.dataSource = 'Task-generated split (PhDN skipped)';
	resultSindy.operatorMode = 'true-operator';
	resultSindy.stage0InitialGuessSync = stage0GuessSync;
	resultSindy.dictionarySource = get_sindy_dictionary_source_local(arch, sindyOpts);
	resultSindy.bestOpArgPolyOrder = opArgPolyOrder;
	resultSindy.data = struct();
	resultSindy.data.nTrain = size(Xtr, 1);
	resultSindy.data.nVal = size(Xval, 1);
	resultSindy.data.nTest = size(Xte, 1);
	resultSindy.data.nOod = size(XOod, 1);
	if ~isempty(oodDomain)
		resultSindy.data.oodDomain = oodDomain;
	end
	resultSindy.data.Xtr = Xtr;
	resultSindy.data.Ytr = Ytr;
	resultSindy.data.Xval = Xval;
	resultSindy.data.Yval = Yval;
	resultSindy.data.Xte = Xte;
	resultSindy.data.Yte = Yte;
	resultSindy.data.Xood = XOod;
	resultSindy.data.Yood = YOod;

	if ~isfield(resultSindy, 'timeStats') || isempty(resultSindy.timeStats)
		resultSindy.timeStats = struct();
	end
	resultSindy.timeStats.dataGenerationTime = dataGenerationTime;
	resultSindy.timeStats.splitTime = splitTime;
	resultSindy.timeStats.oodDataGenerationTime = oodDataGenerationTime;

	if sindyOpts.verbose
		print_sindy_baseline_result(resultSindy);
		fprintf('SINDy-only data generation / split / OOD time : %.3f / %.3f / %.3f s\n', ...
			dataGenerationTime, splitTime, oodDataGenerationTime);
		fprintf('Best opArgPolyOrder used by SINDy Phi         : %d\n', opArgPolyOrder);
		fprintf('========================================\n');
	end
end


function source = get_sindy_dictionary_source_local(arch, sindyOpts)
	dictionaryMode = lower(strtrim(char(getfield_default_local(sindyOpts, 'dictionaryMode', 'general'))));
	if strcmpi(dictionaryMode, 'phdn_phi') && isfield(arch, 'caseDictionary') && ...
			isfield(arch.caseDictionary, 'source') && ~isempty(arch.caseDictionary.source)
		source = arch.caseDictionary.source;
	elseif strcmpi(dictionaryMode, 'phdn_phi')
		source = 'Legacy single-layer PhDN branch dictionary Phi(x), with mapped PhDN compact support when available';
	elseif isfield(arch, 'caseDictionary') && isfield(arch.caseDictionary, 'source')
		source = arch.caseDictionary.source;
	else
		source = 'Independent general SINDy dictionary';
	end
end

function opArgPolyOrder = resolve_sindy_op_arg_poly_order_local(task, opts, arch)
	opArgPolyOrder = [];
	if isfield(opts, 'training') && isfield(opts.training, 'opArgPolyOrderList') && ...
			~isempty(opts.training.opArgPolyOrderList)
		opArgPolyOrder = opts.training.opArgPolyOrderList(1);
	elseif isfield(task, 'training') && isfield(task.training, 'opArgPolyOrderList') && ...
			~isempty(task.training.opArgPolyOrderList)
		opArgPolyOrder = task.training.opArgPolyOrderList(1);
	elseif isfield(arch, 'interact') && isfield(arch.interact, 'opArgPolyOrder') && ...
			~isempty(arch.interact.opArgPolyOrder)
		opArgPolyOrder = arch.interact.opArgPolyOrder;
	elseif isfield(arch, 'polyOrder') && ~isempty(arch.polyOrder)
		opArgPolyOrder = arch.polyOrder;
	else
		opArgPolyOrder = 1;
	end
	opArgPolyOrder = max(1, round(opArgPolyOrder));
end


function support = make_sindy_support_from_phdn_opts_local(phdnOpts, arch)
%MAKE_SINDY_SUPPORT_FROM_PHDN_OPTS_LOCAL Map PhDN dictionarySupportA to SINDy Phi(x).
	support = struct();
	support.keepRows = [];
	support.source = 'full_sindy_phi';
	support.reason = 'phdnOpts.training.dictionarySupportA unavailable';

	if isempty(phdnOpts) || ~isstruct(phdnOpts) || ~isfield(phdnOpts, 'training') || ...
			~isfield(phdnOpts.training, 'dictionarySupportA') || isempty(phdnOpts.training.dictionarySupportA)
		return;
	end

	supportA = phdnOpts.training.dictionarySupportA;
	if isempty(supportA) || size(supportA, 1) < 1 || size(supportA, 2) < 1 || isempty(supportA{1, 1})
		support.reason = 'dictionarySupportA{1,1} unavailable';
		return;
	end

	M = logical(supportA{1, 1});
	keep = any(M, 1).';
	nTerms = branch_dictionary_size(arch.nx, arch);
	if numel(keep) ~= nTerms
		support.reason = sprintf('dictionarySupportA{1,1} has %d columns but SINDy Phi has %d terms', numel(keep), nTerms);
		return;
	end

	support.keepRows = keep;
	if isfield(phdnOpts.training, 'dictionarySupportMode') && ~isempty(phdnOpts.training.dictionarySupportMode)
		support.source = ['phdn_dictionary_support_A11_column_union:' char(phdnOpts.training.dictionarySupportMode)];
	else
		support.source = 'phdn_dictionary_support_A11_column_union';
	end
	support.reason = sprintf('mapped %d/%d Phi(x) rows from PhDN A{1,1} compact support', nnz(keep), numel(keep));
end
function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end

