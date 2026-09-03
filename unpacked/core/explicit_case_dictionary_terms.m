function termNames = explicit_case_dictionary_terms(inputDim, arch, layerIndex, branchIndex)
%EXPLICIT_CASE_DICTIONARY_TERMS Return the numerical branch feature terms.
%
% For the v72 SR-to-PhDN route, SR-required nonlinear terms are structural DAG
% channels and are stored separately from the augmentation dictionary. The
% augmentation dictionary is a uniform dimension-dependent family (constant +
% total-degree Poly_k). This function concatenates the two channels only for
% low-level numerical evaluation and coefficient-matrix compatibility. Legacy
% case dictionaries using termsByBlock/termsByLayer/termsByDim remain supported.

	if nargin < 3
		layerIndex = [];
	end
	if nargin < 4
		branchIndex = [];
	end

	if is_branch_disabled_local(arch, branchIndex, layerIndex)
		termNames = {};
		return;
	end

	termNames = {};
	D = struct();
	if isfield(arch, 'caseDictionary') && isstruct(arch.caseDictionary)
		D = arch.caseDictionary;
		% v72 SR route: keep structural SR channels separate from the weak
		% augmentation dictionary. They are concatenated here only because the
		% existing PhDN coefficient matrices use one branch feature matrix.
		if ~isempty(branchIndex) && ~isempty(layerIndex) && ...
				(isfield(D, 'structuralTermsByBlock') || isfield(D, 'augmentationTermsByBlock'))
			structural = get_block_terms_local(D, 'structuralTermsByBlock', branchIndex, layerIndex);
			augmentation = get_block_terms_local(D, 'augmentationTermsByBlock', branchIndex, layerIndex);
			termNames = unique_stable_local([structural(:); augmentation(:)]);
		% Legacy priority: branch/block dictionary > layer dictionary > dimension dictionary.
		elseif ~isempty(branchIndex) && ~isempty(layerIndex) && isfield(D, 'termsByBlock') && ...
				iscell(D.termsByBlock) && size(D.termsByBlock,1) >= branchIndex && ...
				size(D.termsByBlock,2) >= layerIndex && ~isempty(D.termsByBlock{branchIndex, layerIndex})
			termNames = D.termsByBlock{branchIndex, layerIndex};
		elseif ~isempty(layerIndex) && isfield(D, 'termsByLayer') && ...
				numel(D.termsByLayer) >= layerIndex && ~isempty(D.termsByLayer{layerIndex})
			termNames = D.termsByLayer{layerIndex};
		elseif isfield(D, 'termsByDim') && numel(D.termsByDim) >= inputDim && ~isempty(D.termsByDim{inputDim})
			termNames = D.termsByDim{inputDim};
		end
		% Optional global terms are appended only when explicitly requested.
		if isfield(D, 'appendGlobalTerms') && D.appendGlobalTerms && isfield(D, 'globalTerms') && ~isempty(D.globalTerms)
			termNames = unique_stable_local([termNames(:); D.globalTerms(:)]);
		end
	end

	if isempty(termNames)
		if isstruct(D) && isfield(D, 'noFallback') && D.noFallback
			termNames = {};
			return;
		end
		termNames = [{'1'}, arrayfun(@(k) sprintf('v%d', k), 1:inputDim, 'UniformOutput', false)];
	end

	termNames = normalize_term_list_local(termNames, inputDim);
end

function terms = get_block_terms_local(D, fieldName, branchIndex, layerIndex)
	terms = {};
	if ~isfield(D, fieldName); return; end
	B = D.(fieldName);
	if ~iscell(B) || size(B,1) < branchIndex || size(B,2) < layerIndex || isempty(B{branchIndex,layerIndex})
		return;
	end
	terms = B{branchIndex,layerIndex};
	if ischar(terms) || isstring(terms); terms = cellstr(terms); end
	terms = terms(:);
end

function tf = is_branch_disabled_local(arch, branchIndex, layerIndex)
	tf = false;
	if isempty(branchIndex) || isempty(layerIndex) || ~isfield(arch, 'branchActiveMask') || isempty(arch.branchActiveMask)
		return;
	end
	M = arch.branchActiveMask;
	try
		if iscell(M)
			if size(M,1) >= branchIndex && size(M,2) >= layerIndex && ~isempty(M{branchIndex, layerIndex})
				tf = ~logical(M{branchIndex, layerIndex});
			end
		elseif isnumeric(M) || islogical(M)
			if size(M,1) >= branchIndex && size(M,2) >= layerIndex
				tf = ~logical(M(branchIndex, layerIndex));
			end
		end
	catch
		tf = false;
	end
end

function names = normalize_term_list_local(names, inputDim)
	if isstring(names); names = cellstr(names); end
	if ischar(names); names = {names}; end
	names = names(:);
	out = {};
	for i = 1:numel(names)
		name = char(names{i});
		name = strtrim(name);
		if isempty(name); continue; end
		name = strrep(name, ' ', '');
		% Accept x/h names in task definitions but convert them to canonical v.
		for k = inputDim:-1:1
			name = regexprep(name, sprintf('(?<![A-Za-z0-9_])x%d(?![A-Za-z0-9_])', k), sprintf('v%d', k));
			name = regexprep(name, sprintf('(?<![A-Za-z0-9_])h%d(?![A-Za-z0-9_])', k), sprintf('v%d', k));
		end
		out{end+1,1} = name; %#ok<AGROW>
	end
	names = unique_stable_local(out);
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
