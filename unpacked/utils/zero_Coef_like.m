function Coef_zero = zero_Coef_like(Coef_template)
%ZERO_COEF_LIKE Create a zero Coef_M with the same cell sizes.

	Coef_zero = cell(size(Coef_template));
	[layer, ~] = size(Coef_template);

	for ell = 1:layer
		for src = 1:ell
			Coef_zero{src, ell} = zeros(size(Coef_template{src, ell}));
		end
	end
end
