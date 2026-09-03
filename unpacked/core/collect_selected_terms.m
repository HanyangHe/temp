function selectedTerms = collect_selected_terms(mask, Coef, arch, varargin)
%COLLECT_SELECTED_TERMS Map the final active mask to readable dictionary terms.
%
% selectedTerms = collect_selected_terms(mask, Coef, arch)
% The returned struct array records every active coefficient selected by the
% final training mask.  It is intended for post-run diagnostics: which terms
% were kept and what coefficient values they learned.

	if nargin < 2 || isempty(Coef)
		Coef = mask;
	end

	dims = [];
	try
		dims = get_arch_dims(arch);
	catch
		dims = [];
	end

	template = make_empty_term_local();
	selectedTerms = repmat(template, 0, 1);
	gidx = 0;

	for ell = 1:size(mask, 2)
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M)
				continue;
			end

			activeLinear = find(logical(M));
			if isempty(activeLinear)
				continue;
			end

			[rowIdx, colIdx] = ind2sub(size(M), activeLinear);
			terms = local_branch_terms_local(arch, dims, src, ell, size(M, 2));

			for p = 1:numel(activeLinear)
				gidx = gidx + 1;
				row = rowIdx(p);
				col = colIdx(p);
				termInfo = local_term_at_local(terms, col);

				coefValue = NaN;
				if iscell(Coef) && src <= size(Coef, 1) && ell <= size(Coef, 2) && ~isempty(Coef{src, ell})
					C = Coef{src, ell};
					if row <= size(C, 1) && col <= size(C, 2)
						coefValue = C(row, col);
					end
				end

				isSelectedReference = false;

				d = template;
				d.globalIndex = gidx;
				d.src = src;
				d.ell = ell;
				d.row = row;
				d.col = col;
				d.block = sprintf('A{%d,%d}', src, ell);
				d.xi = sprintf('Xi^{(%d,%d)}', src, ell);
				d.inputState = ell - src + 1;
				d.outputState = ell + 1;
				d.termName = termInfo.name;
				d.termType = termInfo.type;
				d.opName = termInfo.opName;
				d.coefficient = coefValue;
				d.absCoefficient = abs(coefValue);
				d.isOracle = logical(isSelectedReference);
				d.hasInv = contains_operator_local(termInfo.name, 'inv');
				d.hasExp = contains_operator_local(termInfo.name, 'exp');
				d.hasSqrt = contains_operator_local(termInfo.name, 'sqrt');
				d.hasLog = contains_operator_local(termInfo.name, 'log');
				d.hasTrig = contains_operator_local(termInfo.name, 'sin') || contains_operator_local(termInfo.name, 'cos') || contains_operator_local(termInfo.name, 'asin');
				d.isCross = strcmpi(termInfo.type, 'cross');
				d.isIdentityCancellation = is_identity_cancellation_term_local(termInfo);
				d.usedRows = parse_used_rows_local(termInfo.name);
				d.numUsedRows = numel(d.usedRows);
				d.polyDegree = estimate_poly_degree_local(termInfo.exponent);
				d.opArgDegree = estimate_op_arg_degree_local(termInfo.exponent, termInfo.type);

				selectedTerms(end + 1, 1) = d; %#ok<AGROW>
			end
		end
	end
end

function d = make_empty_term_local()
	d = struct();
	d.globalIndex = NaN;
	d.src = NaN;
	d.ell = NaN;
	d.row = NaN;
	d.col = NaN;
	d.block = '';
	d.xi = '';
	d.inputState = NaN;
	d.outputState = NaN;
	d.termName = '';
	d.termType = '';
	d.opName = '';
	d.coefficient = NaN;
	d.absCoefficient = NaN;
	d.isOracle = false;
	d.hasInv = false;
	d.hasExp = false;
	d.hasSqrt = false;
	d.hasLog = false;
	d.hasTrig = false;
	d.isCross = false;
	d.isIdentityCancellation = false;
	d.usedRows = [];
	d.numUsedRows = 0;
	d.polyDegree = NaN;
	d.opArgDegree = NaN;
end

function terms = local_branch_terms_local(arch, dims, src, ell, nCols)
	try
		k = ell - src + 1;
		if ~isempty(dims) && k <= numel(dims)
			inputDim = dims(k);
		else
			inputDim = nCols;
		end
		if k == 1
			varPrefix = 'x';
		else
			varPrefix = 'h';
		end
		terms = branch_dictionary_terms(inputDim, arch, varPrefix, ell, src);
	catch
		terms = struct('index', {}, 'name', {}, 'type', {}, 'exponent', {}, 'opName', {}, 'leftIndex', {}, 'rightIndex', {});
	end

	if numel(terms) < nCols
		for jj = (numel(terms) + 1):nCols
			terms(jj).index = jj; %#ok<AGROW>
			terms(jj).name = sprintf('term_%d', jj); %#ok<AGROW>
			terms(jj).type = 'unknown'; %#ok<AGROW>
			terms(jj).exponent = []; %#ok<AGROW>
			terms(jj).opName = ''; %#ok<AGROW>
			terms(jj).leftIndex = NaN; %#ok<AGROW>
			terms(jj).rightIndex = NaN; %#ok<AGROW>
		terms(jj).isIdentityCancellation = false; %#ok<AGROW>
		end
	end
end

function t = local_term_at_local(terms, col)
	if col >= 1 && col <= numel(terms)
		t = terms(col);
	else
		t = struct('index', col, 'name', sprintf('term_%d', col), 'type', 'unknown', 'exponent', [], 'opName', '', 'leftIndex', NaN, 'rightIndex', NaN);
	end
	if ~isfield(t, 'opName') || isempty(t.opName)
		t.opName = '';
	end
	if ~isfield(t, 'isIdentityCancellation') || isempty(t.isIdentityCancellation)
		t.isIdentityCancellation = is_identity_cancellation_term_local(t);
	end
end

function tf = is_identity_cancellation_term_local(term)
	tf = false;
	if ~isstruct(term) || ~isfield(term, 'type') || ~strcmpi(term.type, 'cross')
		return;
	end
	if ~isfield(term, 'opName') || ~strcmpi(term.opName, 'inv')
		return;
	end
	if ~isfield(term, 'exponent') || ~isstruct(term.exponent) || ...
			~isfield(term.exponent, 'left') || ~isfield(term.exponent, 'opArg')
		return;
	end
	leftExp = term.exponent.left;
	opArgExp = term.exponent.opArg;
	if isempty(leftExp) || isempty(opArgExp) || ~isnumeric(leftExp) || ~isnumeric(opArgExp)
		return;
	end
	tf = isequal(size(leftExp), size(opArgExp)) && isequal(leftExp, opArgExp) && any(leftExp ~= 0);
end

function tf = contains_operator_local(termName, opName)
	termName = char(termName);
	opName = char(opName);
	tf = ~isempty(regexp(termName, ['(^|[^A-Za-z0-9_])' regexptranslate('escape', opName) '\('], 'once'));
end

function rows = parse_used_rows_local(termName)
	termName = char(termName);
	tok = regexp(termName, '[xh](\d+)', 'tokens');
	if isempty(tok)
		rows = [];
		return;
	end
	rows = zeros(1, numel(tok));
	for ii = 1:numel(tok)
		rows(ii) = str2double(tok{ii}{1});
	end
	rows = unique(rows(isfinite(rows) & rows > 0));
end

function deg = estimate_poly_degree_local(exponent)
	deg = NaN;
	if isempty(exponent)
		return;
	end
	if isnumeric(exponent)
		deg = sum(exponent(:));
	elseif isstruct(exponent) && isfield(exponent, 'left') && isnumeric(exponent.left)
		deg = sum(exponent.left(:));
	end
end

function deg = estimate_op_arg_degree_local(exponent, termType)
	deg = NaN;
	if isempty(exponent)
		return;
	end
	if strcmpi(termType, 'op') && isnumeric(exponent)
		deg = sum(exponent(:));
	elseif strcmpi(termType, 'cross') && isstruct(exponent) && isfield(exponent, 'opArg') && isnumeric(exponent.opArg)
		deg = sum(exponent.opArg(:));
	end
end
