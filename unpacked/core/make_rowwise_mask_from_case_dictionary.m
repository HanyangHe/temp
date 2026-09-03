function [maskA, info] = make_rowwise_mask_from_case_dictionary(Coef_template, arch)
%MAKE_ROWWISE_MASK_FROM_CASE_DICTIONARY Build a hard row-wise mask from caseDictionary.rowTerms.
%
% This utility is used by prior-level strong-prior cases.  termsByBlock defines
% the compact branch dictionary, while rowTerms defines which terms are required
% in each output row.  The returned mask has the same shape as Coef_template and
% can be used as an admissible hard support mask.

	maskA = false_mask_like(Coef_template);
	info = struct('applied', false, 'active', 0, 'message', '');

	if ~isfield(arch, 'caseDictionary') || ~isstruct(arch.caseDictionary) || ...
			~isfield(arch.caseDictionary, 'rowTerms') || isempty(arch.caseDictionary.rowTerms)
		info.message = 'No caseDictionary.rowTerms field.';
		return;
	end

	R = arch.caseDictionary.rowTerms;
	if ~iscell(R)
		info.message = 'caseDictionary.rowTerms is not a cell array.';
		return;
	end

	dims = get_arch_dims(arch);
	L = arch.layer;
	for ell = 1:L
		for src = 1:ell
			if src > size(Coef_template,1) || ell > size(Coef_template,2) || isempty(Coef_template{src,ell})
				continue;
			end
			[rowDim, nTerms] = size(Coef_template{src,ell});
			if ~is_branch_active_local(arch, src, ell)
				maskA{src,ell} = false(rowDim, nTerms);
				continue;
			end
			if size(R,1) < src || size(R,2) < ell || isempty(R{src,ell})
				maskA{src,ell} = false(rowDim, nTerms);
				continue;
			end
			rowSpec = R{src,ell};
			if ~iscell(rowSpec)
				maskA{src,ell} = false(rowDim, nTerms);
				continue;
			end
			k = ell - src + 1;
			inputDim = dims(k);
			if k == 1
				prefix = 'x';
			else
				prefix = 'h';
			end
			terms = branch_dictionary_terms(inputDim, arch, prefix, ell, src);
			termNames = {terms.name};
			M = false(rowDim, nTerms);
			for r = 1:min(rowDim, numel(rowSpec))
				termsR = normalize_row_terms_local(rowSpec{r}, inputDim, prefix);
				for q = 1:numel(termsR)
					idx = find(strcmp(termNames, termsR{q}), 1);
					if isempty(idx)
						warning('Row-wise prior term %s was not found in A{%d,%d} row %d dictionary.', termsR{q}, src, ell, r);
					else
						M(r, idx) = true;
					end
				end
			end
			maskA{src,ell} = M;
		end
	end
	info.applied = true;
	info.active = count_active_mask(maskA);
	info.message = sprintf('row-wise caseDictionary hard mask, active=%d', info.active);
end

function tf = is_branch_active_local(arch, src, ell)
	tf = true;
	if ~isfield(arch, 'branchActiveMask') || isempty(arch.branchActiveMask)
		return;
	end
	M = arch.branchActiveMask;
	try
		if iscell(M)
			if size(M,1) >= src && size(M,2) >= ell && ~isempty(M{src,ell})
				tf = logical(M{src,ell});
			end
		elseif isnumeric(M) || islogical(M)
			if size(M,1) >= src && size(M,2) >= ell
				tf = logical(M(src,ell));
			end
		end
	catch
		tf = true;
	end
end

function terms = normalize_row_terms_local(terms, inputDim, prefix)
	if isempty(terms)
		terms = {};
		return;
	end
	if ischar(terms) || isstring(terms)
		terms = cellstr(terms);
	end
	terms = terms(:);
	out = {};
	for i = 1:numel(terms)
		name = strrep(strtrim(char(terms{i})), ' ', '');
		if isempty(name); continue; end
		for k = inputDim:-1:1
			name = regexprep(name, sprintf('(?<![A-Za-z0-9_])v%d(?![A-Za-z0-9_])', k), sprintf('%s%d', prefix, k));
			name = regexprep(name, sprintf('(?<![A-Za-z0-9_])x%d(?![A-Za-z0-9_])', k), sprintf('%s%d', prefix, k));
			name = regexprep(name, sprintf('(?<![A-Za-z0-9_])h%d(?![A-Za-z0-9_])', k), sprintf('%s%d', prefix, k));
		end
		out{end+1,1} = name; %#ok<AGROW>
	end
	terms = unique_stable_local(out);
end

function out = unique_stable_local(in)
	out = {};
	for i = 1:numel(in)
		v = char(in{i});
		if ~any(strcmp(out, v))
			out{end+1,1} = v; %#ok<AGROW>
		end
	end
end
