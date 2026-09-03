function mask = false_mask_like(Coef_M)
%FALSE_MASK_LIKE Create a false mask with the same active cell structure.

	mask = cell(size(Coef_M));
	[layer, ~] = size(Coef_M);

	for ell = 1:layer
		for src = 1:ell
			mask{src, ell} = false(size(Coef_M{src, ell}));
		end
	end
end
