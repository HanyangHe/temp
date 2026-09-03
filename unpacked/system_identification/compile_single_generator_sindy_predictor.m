function compiled = compile_single_generator_sindy_predictor(resultSindy)
%COMPILE_SINGLE_GENERATOR_SINDY_PREDICTOR Compile active SINDy terms once.
%
% The full general dictionary is not reconstructed during rollout. Only rows
% retained by ROWKEEP and used by at least one nonzero Xi coefficient are
% parsed and evaluated. The compiled result is checked against the original
% PREDICT_SINDY_BASELINE before use.

    required = {'arch', 'Xi', 'rowKeep'};
    for k = 1:numel(required)
        if ~isfield(resultSindy, required{k}) || isempty(resultSindy.(required{k}))
            error('SINDy result.%s is missing.', required{k});
        end
    end

    nx = resultSindy.arch.nx;
    namesRaw = explicit_case_dictionary_terms(nx, resultSindy.arch, 1, 1);
    rowKeep = logical(resultSindy.rowKeep(:));
    if numel(rowKeep) ~= numel(namesRaw)
        error('SINDy rowKeep size mismatch: %d versus %d terms.', ...
            numel(rowKeep), numel(namesRaw));
    end

    namesKept = namesRaw(rowKeep);
    Xi = resultSindy.Xi;
    if ~isnumeric(Xi) || ndims(Xi) ~= 2
        error('SINDy Xi must be a numeric matrix.');
    end
    if size(Xi, 1) ~= numel(namesKept)
        error('SINDy Xi row mismatch: %d versus %d kept terms.', ...
            size(Xi, 1), numel(namesKept));
    end

    active = any(Xi ~= 0, 2);
    compiled = struct();
    compiled.kind = 'compiled_sindy_active_value_only_ast';
    compiled.termPlan = compile_explicit_value_terms( ...
        namesKept(active), resultSindy.arch, 1, 1);
    compiled.Xi = Xi(active, :);
    compiled.activeTermCount = nnz(active);
    compiled.activeCoefficientCount = nnz(compiled.Xi);
    compiled.fallbackTermCount = compiled.termPlan.nFallback;
    compiled.useScaled = false;
    compiled.mu = zeros(1, compiled.activeTermCount);
    compiled.sigma = ones(1, compiled.activeTermCount);

    if isfield(resultSindy, 'libraryNormalization') && ...
            isstruct(resultSindy.libraryNormalization)
        compiled.useScaled = logical(getfield_default_local( ...
            resultSindy.libraryNormalization, 'useScaledPrediction', false));
        if compiled.useScaled
            mu = resultSindy.libraryNormalization.mean;
            sigma = resultSindy.libraryNormalization.scale;
            if numel(mu) ~= numel(namesKept) || numel(sigma) ~= numel(namesKept)
                error('SINDy library-normalization size does not match kept terms.');
            end
            compiled.mu = reshape(mu(active), 1, []);
            compiled.sigma = reshape(sigma(active), 1, []);
            compiled.sigma(~isfinite(compiled.sigma) | compiled.sigma == 0) = 1;
        end
    end

    [maxAbsError, maxRelError, nProbe] = validate_compiled_sindy_local( ...
        compiled, resultSindy, nx);
    compiled.validation = struct('nProbe', nProbe, ...
        'maxAbsError', maxAbsError, 'maxRelError', maxRelError);
end

function [maxAbsError, maxRelError, nProbe] = validate_compiled_sindy_local(compiled, resultSindy, nx)
    Xprobe = zeros(0, nx);
    if isfield(resultSindy, 'data') && isstruct(resultSindy.data)
        fields = {'Xtr', 'Xval', 'Xte'};
        for k = 1:numel(fields)
            name = fields{k};
            if isfield(resultSindy.data, name) && ...
                    isnumeric(resultSindy.data.(name)) && ...
                    size(resultSindy.data.(name), 2) == nx && ...
                    ~isempty(resultSindy.data.(name))
                values = resultSindy.data.(name);
                count = min(3, size(values, 1));
                Xprobe = [Xprobe; values(1:count, :)]; %#ok<AGROW>
            end
        end
    end

    % Older saved SINDy results may not store raw X. Use the architecture
    % domain center only when available; otherwise validation is deferred to
    % the rollout warm-up call.
    if isempty(Xprobe)
        maxAbsError = NaN;
        maxRelError = NaN;
        nProbe = 0;
        return;
    end
    Xprobe = Xprobe(all(isfinite(Xprobe), 2), :);
    nProbe = size(Xprobe, 1);
    if nProbe == 0
        maxAbsError = NaN;
        maxRelError = NaN;
        return;
    end

    Yreference = predict_sindy_baseline(resultSindy, Xprobe);
    Ycompiled = predict_compiled_single_generator_sindy(compiled, Xprobe);
    delta = abs(Ycompiled - Yreference);
    maxAbsError = max(delta(:));
    scale = max(1, abs(Yreference));
    relative = delta ./ scale;
    maxRelError = max(relative(:));

    absTolerance = 1e-9;
    relTolerance = 1e-8;
    if ~isfinite(maxAbsError) || ~isfinite(maxRelError) || ...
            (maxAbsError > absTolerance && maxRelError > relTolerance)
        error(['Compiled SINDy validation failed: max absolute/relative ', ...
            'difference = %.6e / %.6e on %d probe points.'], ...
            maxAbsError, maxRelError, nProbe);
    end
end

function value = getfield_default_local(s, name, defaultValue)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = defaultValue;
    end
end
