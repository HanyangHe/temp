function merged = merge_soft_saturated_lorenz96_noise_rows(existingRows,newRows)
%MERGE_SOFT_SATURATED_LORENZ96_NOISE_ROWS Merge by exact (method,round,rho).

    if isempty(existingRows); merged = newRows; return; end
    if isempty(newRows); merged = existingRows; return; end
    merged = existingRows;
    tol = 1e-14;
    for i = 1:numel(newRows)
        replace = false(1,numel(merged));
        if isfield(merged,'method') && isfield(newRows,'method')
            replace = strcmpi({merged.method},newRows(i).method);
            if isfield(merged,'roundIndex') && isfield(newRows,'roundIndex')
                replace = replace & [merged.roundIndex] == newRows(i).roundIndex;
            end
            if isfield(merged,'noiseLevel') && isfield(newRows,'noiseLevel')
                replace = replace & abs([merged.noiseLevel]-newRows(i).noiseLevel) <= tol;
            end
        end
        merged = merged(~replace);
        if isempty(merged)
            merged = newRows(i);
        else
            merged(end+1) = newRows(i); %#ok<AGROW>
        end
    end
end
