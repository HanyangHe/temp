function Coef_M = unpack_Coef_M_by_mask(theta, mask, Coef_template)
%UNPACK_COEF_M_BY_MASK Unpack theta into Coef_M according to mask.

	Coef_M = Coef_template;
	ptr = 1;
	[layer, ~] = size(mask);

	for ell = 1:layer
		for src = 1:ell
			M = mask{src, ell};
			if isempty(M)
				continue;
			end

			nActive = nnz(M);
			if nActive > 0
				A = Coef_M{src, ell};
				A(M) = theta(ptr : ptr + nActive - 1);
				Coef_M{src, ell} = A;
				ptr = ptr + nActive;
			end
		end
	end
end
