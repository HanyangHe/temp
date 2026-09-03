function print_admissible_mask_summary(A, titleStr)
%PRINT_ADMISSIBLE_MASK_SUMMARY Print allowed coefficient counts per block.

	if nargin < 2 || isempty(titleStr)
		titleStr = 'Admissible mask summary';
	end
	fprintf('\n%s\n', titleStr);
	for j = 1:size(A, 2)
		for i = 1:size(A, 1)
			if isempty(A{i, j})
				continue;
			end
			rowCounts = sum(A{i, j}, 2);
			fprintf('  A{%d,%d}: allowed %d / %d, row counts = [%s]\n', ...
				i, j, nnz(A{i, j}), numel(A{i, j}), num2str(rowCounts(:).'));
		end
	end
end
