function idx = find_branch_dictionary_terms(inputDim, arch, names, varPrefix, layerIndex)
%FIND_BRANCH_DICTIONARY_TERMS Find exact dictionary-term indices by name.
%
% names can be a char, string, or cell array of chars. The search is exact
% after removing spaces. Use branch_dictionary_terms(...) to inspect names.

	if nargin < 4 || isempty(varPrefix)
		varPrefix = 'h';
	end
	if nargin < 5
		layerIndex = [];
	end
	if ischar(names) || isstring(names)
		names = cellstr(names);
	end

	terms = branch_dictionary_terms(inputDim, arch, varPrefix, layerIndex);
	allNames = {terms.name};
	canonAll = cellfun(@canonical_local, allNames, 'UniformOutput', false);

	idx = zeros(1, numel(names));
	for k = 1:numel(names)
		q = canonical_local(names{k});
		match = find(strcmp(canonAll, q), 1, 'first');
		if isempty(match)
			available = strjoin(allNames, ', ');
			error('Dictionary term "%s" was not found for inputDim=%d. Available terms: %s', ...
				char(names{k}), inputDim, available);
		end
		idx(k) = terms(match).index;
	end
end

function s = canonical_local(s)
	s = char(s);
	s = regexprep(s, '\s+', '');
end
