function nTheta = count_active_mask(trainMask)
%COUNT_ACTIVE_MASK Count active coefficients in a cell-array or numeric mask.
%
% For masked-LSQ, this number is exactly the optimization dimension when
% trainMask is the final mask passed to pack/unpack_Coef_M_by_mask.

	nTheta = 0;
	if isempty(trainMask)
		return;
	end

	if iscell(trainMask)
		for k = 1:numel(trainMask)
			if isempty(trainMask{k})
				continue;
			end
			nTheta = nTheta + nnz(trainMask{k});
		end
	else
		nTheta = nnz(trainMask);
	end
end
