function txt = format_pykan_shape(shape)
%FORMAT_PYKAN_SHAPE Convert pykan [sum,mult] widths to readable text.
%
% jsondecode may represent a pykan width as an N-by-2 numeric matrix,
% a cell array of two-element vectors, or an ordinary numeric vector. When
% every multiplicative-node count is zero, report the familiar compact shape
% [n0 n1 ... nL]. Otherwise retain each (sum,mult) pair explicitly.

    if isempty(shape)
        txt = '[]';
        return;
    end

    pairs = numeric_pairs_local(shape);
    if ~isempty(pairs)
        if size(pairs,2) == 2 && all(pairs(:,2) == 0)
            txt = ['[', strtrim(num2str(pairs(:,1).', '%g ')), ']'];
        elseif size(pairs,2) == 2
            parts = arrayfun(@(i) sprintf('(%g,%g)', pairs(i,1), pairs(i,2)), ...
                1:size(pairs,1), 'UniformOutput', false);
            txt = ['[', strjoin(parts, ' '), ']'];
        else
            txt = ['[', strtrim(num2str(pairs(:).', '%g ')), ']'];
        end
        return;
    end

    if iscell(shape)
        parts = cellfun(@format_pykan_shape, shape, 'UniformOutput', false);
        txt = ['[', strjoin(parts, ' '), ']'];
    else
        txt = char(string(shape));
    end
end

function pairs = numeric_pairs_local(shape)
    pairs = [];
    if isnumeric(shape)
        if isvector(shape)
            pairs = shape(:);
        else
            pairs = shape;
        end
        return;
    end
    if ~iscell(shape) || isempty(shape)
        return;
    end
    try
        rows = cell(numel(shape),1);
        for i = 1:numel(shape)
            item = shape{i};
            if ~isnumeric(item) || isempty(item)
                return;
            end
            rows{i} = double(item(:).');
        end
        widths = cellfun(@numel, rows);
        if all(widths == widths(1))
            pairs = vertcat(rows{:});
        end
    catch
        pairs = [];
    end
end
