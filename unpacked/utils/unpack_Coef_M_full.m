function Coef_M = unpack_Coef_M_full(theta, Coef_template)
%UNPACK_COEF_M_FULL Unpack full theta into Coef_M.

	Coef_M = Coef_template;
	ptr = 1;
	[layer, ~] = size(Coef_template);

	for ell = 1:layer
		for src = 1:ell
			sz = size(Coef_template{src, ell});
			nElem = prod(sz);
			Coef_M{src, ell} = reshape(theta(ptr : ptr + nElem - 1), sz);
			ptr = ptr + nElem;
		end
	end
end
