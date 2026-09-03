function resultSindy = run_sindy_baseline_from_phdn_result(resultPhdn, task, sindyOpts, phdnOpts)
%RUN_SINDY_BASELINE_FROM_PHDN_RESULT Train a single-layer SINDy baseline.
%
% The baseline uses the exact data split saved in resultPhdn.data.  By default
% it builds an independent broad/general SINDy dictionary from raw inputs, not
% the PhDN/SR dictionary.  Legacy PhDN-Phi mode is still available through
% sindyOpts.dictionaryMode = 'phdn_phi'.

	if nargin < 3 || isempty(sindyOpts)
		sindyOpts = sindy_default_options();
	end
	if nargin < 4
		phdnOpts = [];
	end
	if isempty(phdnOpts) && isstruct(resultPhdn) && isfield(resultPhdn, 'opts') && ...
			~isempty(resultPhdn.opts)
		phdnOpts = resultPhdn.opts;
	end

	if nargin < 2 || isempty(task)
		if isfield(resultPhdn, 'task') && ~isempty(resultPhdn.task)
			task = resultPhdn.task;
		else
			error('Task is missing. Pass task or provide resultPhdn.task.');
		end
	end

	% Stage-0 initial guesses are optionally mirrored into SINDy. The Lorenz--96
	% matched-library comparison disables this channel so its declared 53/25
	% (or dimension-generalized) dictionaries remain exact.
	if logical(getfield_default_local(sindyOpts,'syncStage0InitialGuesses',true))
		[sindyOpts, stage0GuessSync] = sync_sindy_stage0_initial_guesses( ...
			sindyOpts, phdnOpts, task.nx);
	else
		sindyOpts.stage0InitialGuessTerms = {};
		stage0GuessSync = struct('enabled',false, ...
			'source','disabled_by_sindy_option','scope','none', ...
			'requestedTerms',{{}},'normalizedTerms',{{}}, ...
			'nRequested',0,'nNormalizedUnique',0, ...
			'reason','SINDy Stage-0 guess synchronization disabled for an exact matched dictionary.');
		sindyOpts.stage0InitialGuessSyncInfo = stage0GuessSync;
	end

	if ~isfield(resultPhdn, 'data') || isempty(resultPhdn.data)
		error('resultPhdn.data is missing. The PhDN result must save exact data arrays.');
	end

	d = resultPhdn.data;
	requiredFields = {'Xtr', 'Ytr', 'Xval', 'Yval', 'Xte', 'Yte'};
	for k = 1:numel(requiredFields)
		if ~isfield(d, requiredFields{k}) || isempty(d.(requiredFields{k}))
			error('resultPhdn.data.%s is missing or empty.', requiredFields{k});
		end
	end

	XOod = [];
	YOod = [];
	if isfield(d, 'Xood') && isfield(d, 'Yood') && ~isempty(d.Xood) && ~isempty(d.Yood)
		XOod = d.Xood;
		YOod = d.Yood;
	end

	baseArch = struct();
	if isfield(resultPhdn, 'arch') && ~isempty(resultPhdn.arch)
		baseArch = resultPhdn.arch;
	elseif isfield(task, 'arch') && ~isempty(task.arch)
		baseArch = task.arch;
	end
	if isfield(task, 'arch') && isfield(task.arch, 'safety') && ~isfield(baseArch, 'safety')
		baseArch.safety = task.arch.safety;
	end

	dictionaryMode = lower(strtrim(char(getfield_default_local(sindyOpts, 'dictionaryMode', 'general'))));
	if strcmpi(dictionaryMode, 'phdn_phi')
		if isempty(fieldnames(baseArch))
			error('Cannot determine PhDN architecture for SINDy dictionaryMode=phdn_phi.');
		end
		arch = baseArch;
		arch.nx = task.nx;
		arch.ny = task.ny;
		if isfield(resultPhdn, 'bestOpArgPolyOrder') && ~isempty(resultPhdn.bestOpArgPolyOrder) && isfinite(resultPhdn.bestOpArgPolyOrder)
			if ~isfield(arch, 'interact') || isempty(arch.interact)
				arch.interact = struct();
			end
			arch.interact.opArgPolyOrder = resultPhdn.bestOpArgPolyOrder;
		end
		arch.operatorMode = 'true';
		arch.sindyLibrarySupport = make_sindy_support_from_phdn_opts_local(phdnOpts, arch);
		arch = append_sindy_stage0_initial_guesses_to_arch( ...
			arch, task.nx, sindyOpts.stage0InitialGuessTerms);
	elseif any(strcmpi(dictionaryMode,{'neural_general','neural_sindy','neural'}))
		arch = make_sindy_neural_arch(task,sindyOpts,baseArch,d.Xtr);
		arch.sindyLibrarySupport = struct('keepRows', [], ...
			'source', 'independent_neural_sindy_dictionary', ...
			'reason', ['Standalone polynomial terms are replaced by fixed ', ...
			'neural-ridge bases; no PhDN support mask is used']);
		sindyOpts.usePhdnDictionarySupport = false;
	else
		arch = make_sindy_general_arch(task, sindyOpts, baseArch);
		arch.sindyLibrarySupport = struct('keepRows', [], 'source', 'independent_general_sindy_dictionary', ...
			'reason', 'SINDy dictionary is generated independently; no PhDN support is used');
		sindyOpts.usePhdnDictionarySupport = false;
	end

	resultSindy = train_sindy_single_layer_baseline( ...
		d.Xtr,  d.Ytr, ...
		d.Xval, d.Yval, ...
		d.Xte,  d.Yte, ...
		XOod, YOod, arch, sindyOpts);

	resultSindy.dataSource = 'Exact PhDN result.data split';
	resultSindy.operatorMode = 'true-operator';
	resultSindy.methodFamily = 'sindy';
	if any(strcmpi(dictionaryMode,{'neural_general','neural_sindy','neural'}))
		resultSindy.method = 'independent_single_layer_neural_sindy_stlsq';
		resultSindy.baselineVariant = 'neural_sindy';
	else
		resultSindy.baselineVariant = 'standard_sindy';
	end
	resultSindy.stage0InitialGuessSync = stage0GuessSync;
	resultSindy.dictionarySource = get_sindy_dictionary_source_local(arch, sindyOpts);
	resultSindy.data = struct();
	resultSindy.data.nTrain = size(d.Xtr, 1);
	resultSindy.data.nVal = size(d.Xval, 1);
	resultSindy.data.nTest = size(d.Xte, 1);
	resultSindy.data.nOod = size(XOod, 1);
	if isfield(d, 'oodDomain')
		resultSindy.data.oodDomain = d.oodDomain;
	end

	if sindyOpts.verbose
		print_sindy_baseline_result(resultSindy);
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

