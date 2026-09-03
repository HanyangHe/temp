function arch = make_sindy_general_arch(task, opts, baseArch)
%MAKE_SINDY_GENERAL_ARCH Build an independent broad/general SINDy dictionary.
%
% This SINDy baseline is independent from the PhDN model and does not reuse
% PhDN masks, SR-derived DAGs, or compact case dictionaries.  It builds the
% classical flat SINDy library directly on the raw inputs:
%   constant + polynomial monomials up to a given total degree
%   + selected unary operator terms
%   + optional variable-times-unary cross terms.
%
% The resulting architecture is single-layer and true-operator.  It is passed
% to the existing explicit dictionary evaluator through arch.caseDictionary.

	if nargin < 2 || isempty(opts)
		opts = sindy_default_options();
	end
	if nargin < 3 || isempty(baseArch)
		baseArch = struct();
	end

	arch = struct();
	arch.nx = task.nx;
	arch.ny = task.ny;
	arch.layer = 1;
	arch.hiddenDims = [];
	arch.operatorMode = 'true';
	arch.dictionaryMode = 'sindy_independent_general';
	arch.branchActiveMask = true(1, 1);

	if isstruct(baseArch) && isfield(baseArch, 'safety') && ~isempty(baseArch.safety)
		arch.safety = baseArch.safety;
	else
		arch.safety = struct();
	end
	if ~isfield(arch.safety, 'eps') || isempty(arch.safety.eps)
		arch.safety.eps = 1e-8;
	end
	if isstruct(baseArch) && isfield(baseArch, 'feasibility') && ~isempty(baseArch.feasibility)
		arch.feasibility = baseArch.feasibility;
	end

	polyOrder = max(0, round(getfield_default_local(opts, 'polyOrder', 2)));
	unaryOps = normalize_cellstr_local(getfield_default_local(opts, 'unaryOperators', {'inv','sqrt','exp','sin','cos','log'}));
	caseSupportsTypedPrior = is_single_generator_dynamic_case_local(task);
	typedGeneratorPrior = logical(getfield_default_local( ...
		opts,'typedPhysicalPriorEnable',caseSupportsTypedPrior));
	if typedGeneratorPrior
		if ~caseSupportsTypedPrior
			error(['The typed generator SINDy prior was requested for a task ', ...
				'that is not recognized as SingleGeneratorDynamic/SMIB-AVR.']);
		end
		% Same physical prior used by Stage-0 SR: fixed parameters absorb all
		% constant denominators and trigonometric functions act only on rotor angle.
		if logical(getfield_default_local(opts,'forbidStateDependentDivision',true))
			unaryOps = unaryOps(~strcmpi(unaryOps,'inv'));
		end
		if ~any(strcmpi(unaryOps,'sin')); unaryOps{end+1} = 'sin'; end
		if ~any(strcmpi(unaryOps,'cos')); unaryOps{end+1} = 'cos'; end
	end
	trigAllowedVariableIndex = round(getfield_default_local( ...
		opts,'trigAllowedVariableIndex',1));
	includePolynomialTerms = logical(getfield_default_local(opts, 'includePolynomialTerms', true));
	includeUnaryOnMonomials = logical(getfield_default_local(opts, 'includeUnaryOnMonomials', true));
	includeOperatorCrossTerms = logical(getfield_default_local(opts, 'includeOperatorCrossTerms', true));
	includeSinCosPair = logical(getfield_default_local(opts, 'includeSinCosPair', false));
	maxTerms = getfield_default_local(opts, 'maxLibraryTerms', Inf);

	terms = build_sindy_general_terms_local(task.nx, polyOrder, unaryOps, ...
		includePolynomialTerms, includeUnaryOnMonomials, includeOperatorCrossTerms, includeSinCosPair, ...
		maxTerms, typedGeneratorPrior,trigAllowedVariableIndex);

	% Fair-prior union: every expression used to seed Stage-0 PySR is also one
	% candidate function in the flat SINDy library. Exact/canonical duplicates
	% are removed while sparse grouped sums are intentionally retained.
	stage0GuessTerms = normalize_cellstr_local(getfield_default_local( ...
		opts, 'stage0InitialGuessTerms', {}));
	[terms, stage0GuessUnion] = union_stage0_guess_terms_local(terms, stage0GuessTerms);

	expectedSize = getfield_default_local(opts,'expectedLibrarySize',[]);
	if logical(getfield_default_local(opts,'strictLibraryAssertions',false)) && ...
			~isempty(expectedSize) && numel(terms) ~= expectedSize
		error('Standard SINDy dictionary size mismatch: constructed %d, expected %d.', ...
			numel(terms),expectedSize);
	end

	D = struct();
	D.caseId = getfield_default_local(task, 'name', 'SINDy_independent_general');
	D.termsByDim = cell(1, max(1, task.nx));
	D.termsByDim{task.nx} = terms;
	D.noFallback = true;
	D.appendGlobalTerms = false;
	D.source = sprintf('independent SINDy general dictionary: polynomialTerms=%d, polyOrder=%d, unary={%s}, unaryOnMonomials=%d, operatorCross=%d', ...
		includePolynomialTerms, polyOrder, strjoin(unaryOps, ','), includeUnaryOnMonomials, includeOperatorCrossTerms);
	if stage0GuessUnion.enabled
		D.source = sprintf('%s + union(Stage0SRInitialGuesses: requested=%d, added=%d, duplicate=%d)', ...
			D.source, stage0GuessUnion.nRequested, stage0GuessUnion.nAdded, stage0GuessUnion.nDuplicate);
	end
	D.stage0InitialGuessUnion = stage0GuessUnion;
	arch.caseDictionary = D;
	arch.sindyDictionaryReport = struct('nTerms', numel(terms), 'polyOrder', polyOrder, ...
		'includePolynomialTerms',includePolynomialTerms, ...
		'unaryOperators', {unaryOps}, ...
		'includeUnaryOnMonomials', includeUnaryOnMonomials, ...
		'includeOperatorCrossTerms', includeOperatorCrossTerms, ...
		'stage0InitialGuessUnion', stage0GuessUnion, 'source', D.source);
end

function terms = build_sindy_general_terms_local(inputDim, polyOrder, unaryOps, includePolynomialTerms, includeUnaryOnMonomials, includeOperatorCrossTerms, includeSinCosPair, maxTerms, typedGeneratorPrior, trigAllowedVariableIndex)
	terms = {'1'};
	if includePolynomialTerms
		polyTerms = build_polynomial_terms_local(inputDim, polyOrder);
		terms = unique_stable_local([terms(:); polyTerms(:)]);
	end

	% Unary operators are applied to raw variables and, by default, pure powers
	% such as v1^2.  This keeps the baseline independent but still able to build
	% traditional operator terms like inv(v1), sqrt(v2^2), exp(v3), sin(v4).
	unaryArgs = arrayfun(@(k) sprintf('v%d', k), 1:inputDim, 'UniformOutput', false).';
	if includeUnaryOnMonomials
		purePowerTerms = {};
		for k = 1:inputDim
			for p = 2:max(polyOrder, 2)
				purePowerTerms{end+1,1} = sprintf('v%d^%d', k, p); %#ok<AGROW>
			end
		end
		unaryArgs = unique_stable_local([unaryArgs(:); purePowerTerms(:)]);
	end

	unaryTerms = {};
	for i = 1:numel(unaryOps)
		op = char(unaryOps{i});
		if isempty(op); continue; end
		if typedGeneratorPrior && any(strcmpi(op,{'sin','cos'}))
			% Known rotor-angle identity: only the configured angle variable is admissible.
			operatorArgs = {sprintf('v%d',trigAllowedVariableIndex)};
		else
			operatorArgs = unaryArgs;
		end
		for j = 1:numel(operatorArgs)
			arg = operatorArgs{j};
			% square/cube are already covered by polynomial terms when applied to raw variables.
			if any(strcmpi(op, {'square','cube'})) && ~contains(arg, '^')
				continue;
			end
			unaryTerms{end+1,1} = sprintf('%s(%s)', op, arg); %#ok<AGROW>
		end
	end
	terms = unique_stable_local([terms(:); unaryTerms(:)]);


	if includeSinCosPair
		if typedGeneratorPrior
			kList = trigAllowedVariableIndex;
		else
			kList = 1:inputDim;
		end
		for k = kList
			terms{end+1,1} = sprintf('sin(v%d)', k); %#ok<AGROW>
			terms{end+1,1} = sprintf('cos(v%d)', k); %#ok<AGROW>
		end
		terms = unique_stable_local(terms);
	end

	% Optional flat cross terms between raw variables and unary operator terms.
	% This is useful for traditional one-layer SINDy libraries to represent terms
	% like v1*inv(v2), v1*sqrt(v2^2), or v1*cos(v2) without relying on PhDN layers.
	if includeOperatorCrossTerms && ~isempty(unaryTerms)
		varTerms = arrayfun(@(k) sprintf('v%d', k), 1:inputDim, 'UniformOutput', false);
		crossTerms = {};
		for i = 1:numel(varTerms)
			for j = 1:numel(unaryTerms)
				ut = unaryTerms{j};
				% Skip trivial self square/cube duplicates when possible.
				crossTerms{end+1,1} = sprintf('%s*%s', varTerms{i}, ut); %#ok<AGROW>
			end
		end
		terms = unique_stable_local([terms(:); crossTerms(:)]);
	end

	if isfinite(maxTerms) && numel(terms) > maxTerms
		terms = terms(1:maxTerms);
	end
end


function [out, info] = union_stage0_guess_terms_local(baseTerms, guessTerms)
	out = baseTerms(:);
	info = struct();
	info.enabled = ~isempty(guessTerms);
	info.requestedTerms = guessTerms(:);
	info.nRequested = numel(guessTerms);
	info.nAdded = 0;
	info.nDuplicate = 0;
	info.addedTerms = {};
	info.duplicateTerms = {};
	info.libraryIndices = zeros(numel(guessTerms),1);
	info.libraryTerms = cell(numel(guessTerms),1);

	keys = cellfun(@canonical_term_key_local, out, 'UniformOutput', false);
	for k = 1:numel(guessTerms)
		term = strrep(strtrim(char(guessTerms{k})), ' ', '');
		key = canonical_term_key_local(term);
		idx = find(strcmp(keys, key), 1);
		if isempty(idx)
			out{end+1,1} = term; %#ok<AGROW>
			keys{end+1,1} = key; %#ok<AGROW>
			idx = numel(out);
			info.nAdded = info.nAdded + 1;
			info.addedTerms{end+1,1} = term; %#ok<AGROW>
		else
			info.nDuplicate = info.nDuplicate + 1;
			info.duplicateTerms{end+1,1} = term; %#ok<AGROW>
		end
		info.libraryIndices(k) = idx;
		info.libraryTerms{k} = out{idx};
	end
	out = out(:);
end

function key = canonical_term_key_local(term)
	key = lower(strrep(strtrim(char(term)), ' ', ''));
	key = strrep(key, '**', '^');
	key = regexprep(key, 'square\((v[0-9]+)\)', '$1^2');
	key = regexprep(key, 'sqr\((v[0-9]+)\)', '$1^2');
	key = strip_outer_parentheses_local(key);
end

function out = strip_outer_parentheses_local(in)
	out = in;
	changed = true;
	while changed && numel(out) >= 2 && out(1) == '(' && out(end) == ')'
		changed = false;
		depth = 0;
		enclosesAll = true;
		for k = 1:numel(out)
			if out(k) == '('
				depth = depth + 1;
			elseif out(k) == ')'
				depth = depth - 1;
				if depth == 0 && k < numel(out)
					enclosesAll = false;
					break;
				end
			end
		end
		if enclosesAll && depth == 0
			out = out(2:end-1);
			changed = true;
		end
	end
end

function terms = build_polynomial_terms_local(inputDim, polyOrder)
	terms = {};
	if polyOrder <= 0
		return;
	end
	alphas = generate_positive_total_degree_multi_indices_local(inputDim, polyOrder);
	for r = 1:size(alphas, 1)
		terms{end+1,1} = multi_index_to_term_local(alphas(r, :)); %#ok<AGROW>
	end
	terms = unique_stable_local(terms);
end

function alphas = generate_positive_total_degree_multi_indices_local(inputDim, maxOrder)
	rows = zeros(0, inputDim);
	for total = 1:maxOrder
		rows = [rows; generate_fixed_degree_multi_indices_local(inputDim, total)]; %#ok<AGROW>
	end
	alphas = rows;
end

function rows = generate_fixed_degree_multi_indices_local(inputDim, totalDegree)
	if inputDim == 1
		rows = totalDegree;
		return;
	end
	rows = zeros(0, inputDim);
	for a = totalDegree:-1:0
		tail = generate_fixed_degree_multi_indices_local(inputDim - 1, totalDegree - a);
		rows = [rows; [a * ones(size(tail, 1), 1), tail]]; %#ok<AGROW>
	end
end

function term = multi_index_to_term_local(alpha)
	parts = {};
	for k = 1:numel(alpha)
		a = alpha(k);
		if a == 0
			continue;
		elseif a == 1
			parts{end+1} = sprintf('v%d', k); %#ok<AGROW>
		else
			parts{end+1} = sprintf('v%d^%d', k, a); %#ok<AGROW>
		end
	end
	term = strjoin(parts, '*');
end

function c = normalize_cellstr_local(x)
	if isempty(x)
		c = {};
		return;
	end
	if ischar(x) || isstring(x)
		c = cellstr(x);
	elseif iscell(x)
		c = x(:);
	else
		c = {};
	end
	out = {};
	for i = 1:numel(c)
		name = strtrim(char(c{i}));
		if ~isempty(name) && ~any(strcmp(out, name))
			out{end+1,1} = name; %#ok<AGROW>
		end
	end
	c = out;
end

function out = unique_stable_local(in)
	out = {};
	for i = 1:numel(in)
		v = strrep(strtrim(char(in{i})), ' ', '');
		if ~isempty(v) && ~any(strcmp(out, v))
			out{end+1,1} = v; %#ok<AGROW>
		end
	end
end

function val = getfield_default_local(s, name, defaultVal)
	if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
		val = s.(name);
	else
		val = defaultVal;
	end
end

function tf = is_single_generator_dynamic_case_local(task)
	tf = false;
	if ~isstruct(task); return; end
	names = {};
	for fieldName = {'name','caseId','modelVariant'}
		f = fieldName{1};
		if isfield(task,f) && ~isempty(task.(f))
			names{end+1} = lower(char(string(task.(f)))); %#ok<AGROW>
		end
	end
	if isfield(task,'parameters') && isstruct(task.parameters) && ...
			isfield(task.parameters,'modelVariant')
		names{end+1} = lower(char(string(task.parameters.modelVariant))); %#ok<AGROW>
	end
	joined = strjoin(names,' ');
	tf = contains(joined,'singlegeneratordynamic') || ...
		contains(joined,'single_generator_dynamic') || ...
		contains(joined,'smib_avr') || contains(joined,'salient_pole');
end
