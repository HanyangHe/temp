function mask = unpack_mask_vector(mask_vec, Coef_M)
%UNPACK_MASK_VECTOR Map a global logical vector back to Coef_M cells.

	mask = cell(size(Coef_M));
	ptr = 1;
	[layer, ncol] = size(Coef_M);

	for ell = 1:layer
		for src = 1:ncol
			if src <= ell
				sz = size(Coef_M{src, ell});
				nElem = prod(sz);
				mask{src, ell} = reshape(mask_vec(ptr : ptr + nElem - 1), sz);
				ptr = ptr + nElem;
			else
				mask{src, ell} = [];
			end
		end
	end
end
