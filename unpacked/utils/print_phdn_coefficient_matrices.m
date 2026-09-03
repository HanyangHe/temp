function print_phdn_coefficient_matrices(Coef_M, arch, label, precision, onlyActiveBranches)
%PRINT_PHDN_COEFFICIENT_MATRICES Print branch coefficient matrices with labels.

	if nargin < 3 || isempty(label)
		label = 'PhDN coefficient matrices';
	end
	if nargin < 4 || isempty(precision)
		precision = 6;
	end
	if nargin < 5 || isempty(onlyActiveBranches)
		onlyActiveBranches = true;
	end

	dims = get_arch_dims(arch);
	fmt = sprintf('%% .%dg', precision);
	fprintf('\n============================================================\n');
	fprintf('%s\n', label);
	fprintf('============================================================\n');
	for ell = 1:arch.layer
		for src = 1:ell
			if src > size(Coef_M,1) || ell > size(Coef_M,2) || isempty(Coef_M{src,ell})
				continue;
			end
			if onlyActiveBranches && ~is_branch_active_local(arch, src, ell)
				continue;
			end
			inputState = ell - src + 1;
			if inputState == 1
				sourceLabel = 'x';
			else
				sourceLabel = sprintf('h^{%d}', inputState);
			end
			A = Coef_M{src,ell};
			fprintf('\nCoef_M{%d,%d}: %s -> h^{%d}, size %dx%d\n', ...
				src, ell, sourceLabel, ell+1, size(A,1), size(A,2));
			if isempty(A)
				fprintf('  []\n');
			else
				for r = 1:size(A,1)
					fprintf('  [');
					for c = 1:size(A,2)
						fprintf(fmt, A(r,c));
						if c < size(A,2); fprintf('  '); end
					end
					fprintf(']\n');
				end
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
