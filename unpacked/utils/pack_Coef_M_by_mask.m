function theta = pack_Coef_M_by_mask(Coef_M, mask)
%PACK_COEF_M_BY_MASK Pack active entries of coefficient cells into a column vector.
%
% The packing order is intentionally identical to unpack_Coef_M_by_mask:
% layer ell first, then source src within that layer.  Entries inside each
% coefficient block use MATLAB column-major logical indexing A(mask{i,j}).

	theta = [];
	nLayer = size(mask, 2);

	for ell = 1:nLayer
		for src = 1:ell
			if src > size(Coef_M, 1) || ell > size(Coef_M, 2) || isempty(Coef_M{src, ell})
				continue;
			end

			A = Coef_M{src, ell};
			M = mask{src, ell};
			if isempty(M)
				continue;
			end

			if ~isequal(size(A), size(M))
				error('Mask size mismatch at cell (%d,%d): Coef size [%s], mask size [%s].', ...
					src, ell, num2str(size(A)), num2str(size(M)));
			end

			vals = A(M);
			theta = [theta; vals(:)]; %#ok<AGROW>
		end
	end
end
