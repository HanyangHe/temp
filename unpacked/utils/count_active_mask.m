function nActive = count_active_mask(mask)
%COUNT_ACTIVE_MASK Count true entries in a coefficient mask cell array.

	nActive = 0;

	for i = 1:size(mask, 1)
		for j = 1:size(mask, 2)
			M = mask{i, j};

			if isempty(M)
				continue;
			end

			nActive = nActive + nnz(M);
		end
	end
end
