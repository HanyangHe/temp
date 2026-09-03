function theta = pack_Coef_M_full(Coef_M)
%PACK_COEF_M_FULL Pack all coefficient cells into one column vector.
%
% This robust version always converts each coefficient cell to a column
% vector. It is backward-compatible with previous multi-output cases.

	theta = [];

	nRow = size(Coef_M, 1);
	nCol = size(Coef_M, 2);

	for i = 1:nRow
		for j = 1:nCol
			if isempty(Coef_M{i, j})
				continue;
			end

			A = Coef_M{i, j};
			theta = [theta; A(:)]; %#ok<AGROW>
		end
	end
end
