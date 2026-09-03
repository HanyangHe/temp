function print_phdn_branch_dictionaries(arch, maxTermsPerBranch)
%PRINT_PHDN_BRANCH_DICTIONARIES Print every active branch dictionary.

	if nargin < 2 || isempty(maxTermsPerBranch)
		maxTermsPerBranch = Inf;
	end
	dims = get_arch_dims(arch);
	fprintf('\n============================================================\n');
	fprintf('Active PhDN branch dictionaries\n');
	fprintf('Layer dimension vector = [%s]\n', num2str(dims));
	fprintf('============================================================\n');

	for ell = 1:arch.layer
		for src = 1:ell
			if ~is_branch_active_local(arch, src, ell)
				continue;
			end
			inputState = ell - src + 1;
			inputDim = dims(inputState);
			if inputState == 1
				prefix = 'x';
				sourceLabel = 'x';
			else
				prefix = 'h';
				sourceLabel = sprintf('h^{%d}', inputState);
			end
			terms = branch_dictionary_terms(inputDim, arch, prefix, ell, src);
			fprintf('\nBranch Coef_M{%d,%d}: %s -> h^{%d}; input dim=%d, output dim=%d, dictionary size=%d\n', ...
				src, ell, sourceLabel, ell+1, inputDim, dims(ell+1), numel(terms));
			nPrint = min(numel(terms), maxTermsPerBranch);
			for k = 1:nPrint
				fprintf('  phi_%d = %s\n', k, terms(k).name);
			end
			if nPrint < numel(terms)
				fprintf('  ... %d additional term(s) omitted.\n', numel(terms)-nPrint);
			end
		end
	end
end

function tf = is_branch_active_local(arch, src, ell)
	tf = true;
	if ~isfield(arch, 'branchActiveMask') || isempty(arch.branchActiveMask)
		return;
	end
	M = arch.branchActiveMask;
	if iscell(M)
		if size(M,1) >= src && size(M,2) >= ell && ~isempty(M{src,ell})
			tf = logical(M{src,ell});
		end
	elseif (isnumeric(M) || islogical(M)) && size(M,1) >= src && size(M,2) >= ell
		tf = logical(M(src,ell));
	end
end
