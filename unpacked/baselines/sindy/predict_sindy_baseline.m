function Y = predict_sindy_baseline(resultSindy, X)
%PREDICT_SINDY_BASELINE Evaluate a trained single-layer SINDy result on new X.

    validateattributes(X, {'numeric'}, {'2d','real','finite'}, mfilename, 'X');
    required = {'arch','Xi','rowKeep'};
    for k = 1:numel(required)
        if ~isfield(resultSindy, required{k}) || isempty(resultSindy.(required{k}))
            error('SINDy result.%s is missing.', required{k});
        end
    end

    H = X.';
    branch = build_branch_cache(H, resultSindy.arch, 1, {});
    PhiRaw = branch.Phi.';
    rowKeep = logical(resultSindy.rowKeep(:));
    if numel(rowKeep) ~= size(PhiRaw,2)
        error('SINDy rowKeep size mismatch: %d versus %d dictionary columns.', ...
            numel(rowKeep), size(PhiRaw,2));
    end
    Phi = PhiRaw(:,rowKeep);

    useScaled = false;
    if isfield(resultSindy, 'libraryNormalization') && ...
            isstruct(resultSindy.libraryNormalization)
        useScaled = logical(getfield_default_local( ...
            resultSindy.libraryNormalization, 'useScaledPrediction', false));
    end
    if useScaled
        mu = resultSindy.libraryNormalization.mean;
        sigma = resultSindy.libraryNormalization.scale;
        Y = ((Phi-mu)./sigma)*resultSindy.Xi;
    else
        Y = Phi*resultSindy.Xi;
    end
    if any(~isfinite(Y(:)))
        error('SINDy prediction produced NaN or Inf values.');
    end
end

function value = getfield_default_local(s, name, defaultValue)
    if isstruct(s) && isfield(s,name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = defaultValue;
    end
end
