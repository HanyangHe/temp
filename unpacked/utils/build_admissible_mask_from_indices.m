function A = build_admissible_mask_from_indices(Coef_template, spec, defaultAllowed)
%BUILD_ADMISSIBLE_MASK_FROM_INDICES Build a row-wise hard admissibility mask.
%
% A{i,j}(r,k)=true means coefficient Coef_M{i,j}(r,k) is admissible.
% The input spec has the same cell layout as Coef_template.
%
% Supported spec{i,j} formats:
%   []                 : use defaultAllowed for the whole block;
%   false or 0          : disable the whole block;
%   true or 1           : allow the whole block;
%   numeric vector      : allowed column indices shared by all rows;
%   cell array          : row-wise allowed column indices, length = nRows;
%   logical matrix      : direct row-wise admissible matrix.
%
% Example:
%   spec{1,1} = {[2 5], [1 3 8]};  % row 1 allows columns 2,5; row 2 allows 1,3,8.
%   spec{2,2} = 0;                  % disable this branch block.

	if nargin < 3 || isempty(defaultAllowed)
		defaultAllowed = true;
	end
	if nargin < 2 || isempty(spec)
		spec = cell(size(Coef_template));
	end

	if ~iscell(spec)
		error('The admissibility specification must be a cell array.');
	end

	A = make_admissible_mask_like(Coef_template, defaultAllowed);

	for i = 1:size(Coef_template, 1)
		for j = 1:size(Coef_template, 2)
			B = Coef_template{i, j};
			if isempty(B)
				A{i, j} = [];
				continue;
			end

			if i > size(spec, 1) || j > size(spec, 2) || isempty(spec{i, j})
				continue;
			end

			A{i, j} = parse_block_spec_local(spec{i, j}, size(B), i, j);
		end
	end
end

function M = parse_block_spec_local(S, blockSize, i, j)
	nRows = blockSize(1);
	nCols = blockSize(2);

	if islogical(S)
		if isscalar(S)
			M = repmat(S, nRows, nCols);
		elseif isequal(size(S), blockSize)
			M = S;
		else
			error('Logical admissible spec{%d,%d} has size [%s], expected scalar or [%d %d].', ...
				i, j, num2str(size(S)), nRows, nCols);
		end
		return;
	end

	if isnumeric(S)
		if isscalar(S) && (S == 0 || S == 1)
			M = repmat(logical(S), nRows, nCols);
			return;
		end
		M = false(nRows, nCols);
		idx = validate_indices_local(S(:).', nCols, i, j);
		M(:, idx) = true;
		return;
	end

	if iscell(S)
		if numel(S) ~= nRows
			error('Cell admissible spec{%d,%d} must have one entry per row. Expected %d, got %d.', ...
				i, j, nRows, numel(S));
		end
		M = false(nRows, nCols);
		for r = 1:nRows
			idx = S{r};
			if isempty(idx)
				continue;
			end
			if islogical(idx)
				if numel(idx) ~= nCols
					error('Logical row spec{%d,%d}{%d} length must be %d.', i, j, r, nCols);
				end
				M(r, :) = reshape(idx, 1, []);
			else
				idx = validate_indices_local(idx(:).', nCols, i, j);
				M(r, idx) = true;
			end
		end
		return;
	end

	error('Unsupported admissible spec{%d,%d} type: %s.', i, j, class(S));
end

function idx = validate_indices_local(idx, nCols, i, j)
	idx = unique(round(idx));
	idx = idx(~isnan(idx));
	if any(idx < 1) || any(idx > nCols)
		error('Admissible index out of range in spec{%d,%d}. Valid column range is 1:%d.', i, j, nCols);
	end
end
