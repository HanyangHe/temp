function A = make_admissible_mask_like(Coef_template, defaultAllowed)
%MAKE_ADMISSIBLE_MASK_LIKE Create a hard admissibility mask like Coef_template.
%
% A{i,j}(r,k)=true means coefficient Coef_M{i,j}(r,k) is allowed.
% A{i,j}(r,k)=false means this coefficient is forbidden and is forced to zero.

	if nargin < 2 || isempty(defaultAllowed)
		defaultAllowed = true;
	end

	A = cell(size(Coef_template));
	for i = 1:size(Coef_template, 1)
		for j = 1:size(Coef_template, 2)
			B = Coef_template{i, j};
			if isempty(B)
				A{i, j} = [];
			else
				A{i, j} = repmat(logical(defaultAllowed), size(B));
			end
		end
	end
end
