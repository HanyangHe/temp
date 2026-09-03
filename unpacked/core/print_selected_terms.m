function print_selected_terms(selectedTerms, varargin)
%PRINT_SELECTED_TERMS Print readable final selected terms.

	maxRows = Inf;
	sortBy = 'block';
	minAbsCoefficient = 0;

	for k = 1:2:numel(varargin)
		name = lower(strtrim(char(varargin{k})));
		value = varargin{k + 1};
		switch name
			case 'maxrows'
				maxRows = value;
			case 'sortby'
				sortBy = lower(strtrim(char(value)));
			case 'minabscoefficient'
				minAbsCoefficient = value;
			otherwise
				error('print_selected_terms:UnknownOption', 'Unknown option: %s', varargin{k});
		end
	end

	if isempty(selectedTerms)
		fprintf('\nFinal selected term report: no active terms.\n');
		return;
	end

	absCoef = [selectedTerms.absCoefficient];
	keep = isfinite(absCoef) & absCoef >= minAbsCoefficient;
	selectedTerms = selectedTerms(keep);
	if isempty(selectedTerms)
		fprintf('\nFinal selected term report: no active terms above minAbsCoefficient.\n');
		return;
	end

	switch sortBy
		case {'abscoef', 'abscoeff', 'coefficient'}
			[~, order] = sort([selectedTerms.absCoefficient], 'descend');
		case {'layer', 'block'}
			keys = [[selectedTerms.ell].', [selectedTerms.src].', [selectedTerms.row].', [selectedTerms.col].'];
			[~, order] = sortrows(keys, [1 2 3 4]);
		otherwise
			order = 1:numel(selectedTerms);
	end
	selectedTerms = selectedTerms(order);

	if isinf(maxRows)
		maxRowsUse = numel(selectedTerms);
	else
		maxRowsUse = min(max(0, round(maxRows)), numel(selectedTerms));
	end

	nInv = nnz([selectedTerms.hasInv]);
	nExp = nnz([selectedTerms.hasExp]);
	nSqrt = nnz([selectedTerms.hasSqrt]);
	nCross = nnz([selectedTerms.isCross]);
	if isfield(selectedTerms, 'isIdentityCancellation')
		nIdentityCancellation = nnz([selectedTerms.isIdentityCancellation]);
	else
		nIdentityCancellation = 0;
	end
	fprintf('\nFinal selected term report\n');
	fprintf('  active terms             : %d\n', numel(selectedTerms));
	fprintf('  inv / exp / sqrt / cross : %d / %d / %d / %d\n', nInv, nExp, nSqrt, nCross);
	fprintf('  identity-cancellation     : %d\n', nIdentityCancellation);
	fprintf('  %-8s %-3s %-3s %-4s %-5s %-32s %-8s %-8s %-12s\n', ...
		'block','src','ell','row','col','term','type','op','coef');

	for ii = 1:maxRowsUse
		d = selectedTerms(ii);
		fprintf('  %-8s %3d %3d %4d %5d %-32s %-8s %-8s %+12.4e\n', ...
			d.block, d.src, d.ell, d.row, d.col, truncate_string_local(d.termName, 32), ...
			d.termType, d.opName, d.coefficient);
	end

	if maxRowsUse < numel(selectedTerms)
		fprintf('  ... %d more selected terms omitted by MaxRows.\n', numel(selectedTerms) - maxRowsUse);
	end
end

function out = truncate_string_local(in, maxLen)
	in = char(in);
	if numel(in) <= maxLen
		out = in;
	else
		out = [in(1:max(1, maxLen - 3)) '...'];
	end
end
