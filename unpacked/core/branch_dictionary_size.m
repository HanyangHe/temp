function nPhi = branch_dictionary_size(inputDim, arch, layerIndex, branchIndex)
%BRANCH_DICTIONARY_SIZE Number of explicit case-specific dictionary terms.
	if nargin < 3
		layerIndex = [];
	end
	if nargin < 4
		branchIndex = [];
	end
	nPhi = numel(explicit_case_dictionary_terms(inputDim, arch, layerIndex, branchIndex));
end
