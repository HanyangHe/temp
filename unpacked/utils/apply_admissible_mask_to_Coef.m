function CoefOut = apply_admissible_mask_to_Coef(CoefIn, A)
%APPLY_ADMISSIBLE_MASK_TO_COEF Force all inadmissible coefficients to zero.

	CoefOut = CoefIn;
	if nargin < 2 || isempty(A)
		return;
	end

	for i = 1:size(CoefOut, 1)
		for j = 1:size(CoefOut, 2)
			if isempty(CoefOut{i, j})
				continue;
			end
			if i > size(A, 1) || j > size(A, 2) || isempty(A{i, j})
				continue;
			end
			if ~isequal(size(CoefOut{i, j}), size(A{i, j}))
				error('Admissible mask size mismatch at block {%d,%d}.', i, j);
			end
			CoefOut{i, j} = CoefOut{i, j} .* double(A{i, j});
		end
	end
end
