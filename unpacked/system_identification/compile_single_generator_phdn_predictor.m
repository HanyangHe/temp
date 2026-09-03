function compiled = compile_single_generator_phdn_predictor(methodResult)
%COMPILE_SINGLE_GENERATOR_PHDN_PREDICTOR Compile fixed value-only PhDN inference.
%
% Called once before rollout. Only exact nonzero coefficient columns are kept.
% Explicit dictionary terms are parsed once into value-only AST plans. The
% compiled model is checked against the original MODEL_FORWARD before it is
% accepted for numerical integration.

    required = {'Coef_M_est', 'arch', 'data'};
    for k = 1:numel(required)
        if ~isfield(methodResult, required{k}) || isempty(methodResult.(required{k}))
            error('Incomplete PhDN result: methodResult.%s is missing.', required{k});
        end
    end
    if ~isfield(methodResult.data, 'Xtr') || ~isfield(methodResult.data, 'Ytr')
        error('Incomplete PhDN result: training data required for normalization is missing.');
    end

    useIONorm = false;
    useLayerNorm = false;
    if isfield(methodResult, 'opts') && isfield(methodResult.opts, 'norm')
        useIONorm = logical(getfield_default_local( ...
            methodResult.opts.norm, 'useInputOutputNorm', false));
        useLayerNorm = logical(getfield_default_local( ...
            methodResult.opts.norm, 'useLayerNorm', false));
    end
    if useLayerNorm
        error('Compiled rollout predictor does not support hidden-layer normalization.');
    end

    compiled = struct();
    compiled.kind = 'compiled_phdn_value_only_ast';
    compiled.arch = methodResult.arch;
    compiled.dims = get_arch_dims(methodResult.arch);
    compiled.normOpt = fit_norm_options(methodResult.data.Xtr, ...
        methodResult.data.Ytr, useIONorm, useLayerNorm);
    compiled.layer = methodResult.arch.layer;
    compiled.blocks = cell(compiled.layer, compiled.layer);
    compiled.activeCoefficientCount = 0;
    compiled.activeTermCount = 0;
    compiled.fallbackTermCount = 0;

    coefTol = 0; % Preserve the trained model exactly; drop only exact zeros.
    for ell = 1:compiled.layer
        for src = 1:ell
            A = methodResult.Coef_M_est{src, ell};
            if ~isnumeric(A) || ndims(A) ~= 2
                error('PhDN coefficient block {%d,%d} must be a numeric matrix.', src, ell);
            end

            inputStateIndex = ell - src + 1;
            inputDim = compiled.dims(inputStateIndex);
            names = explicit_case_dictionary_terms(inputDim, ...
                methodResult.arch, ell, src);
            if numel(names) ~= size(A, 2)
                error(['PhDN branch {%d,%d} dictionary size mismatch: ', ...
                    '%d names versus %d coefficient columns.'], ...
                    src, ell, numel(names), size(A, 2));
            end

            keep = any(abs(A) > coefTol, 1);
            block = struct();
            block.A = A(:, keep);
            block.keep = find(keep);
            block.inputStateIndex = inputStateIndex;
            block.termPlan = compile_explicit_value_terms( ...
                names(keep), methodResult.arch, ell, src);
            compiled.blocks{src, ell} = block;

            compiled.activeCoefficientCount = compiled.activeCoefficientCount + ...
                nnz(block.A);
            compiled.activeTermCount = compiled.activeTermCount + nnz(keep);
            compiled.fallbackTermCount = compiled.fallbackTermCount + ...
                block.termPlan.nFallback;
        end
    end

    [maxAbsError, maxRelError, nProbe] = validate_compiled_phdn_local( ...
        compiled, methodResult);
    compiled.validation = struct('nProbe', nProbe, ...
        'maxAbsError', maxAbsError, 'maxRelError', maxRelError);
end

function [maxAbsError, maxRelError, nProbe] = validate_compiled_phdn_local(compiled, methodResult)
%VALIDATE_COMPILED_PHDN_LOCAL Compare against the original fixed model once.
    Xprobe = make_probe_points_local(methodResult.data, compiled.dims(1));
    nProbe = size(Xprobe, 1);
    if nProbe == 0
        maxAbsError = NaN;
        maxRelError = NaN;
        return;
    end

    Yreference = model_forward(Xprobe, methodResult.Coef_M_est, ...
        methodResult.arch, compiled.normOpt);
    Ycompiled = predict_compiled_single_generator_phdn(compiled, Xprobe);

    delta = abs(Ycompiled - Yreference);
    maxAbsError = max(delta(:));
    scale = max(1, abs(Yreference));
    relative = delta ./ scale;
    maxRelError = max(relative(:));

    absTolerance = 1e-9;
    relTolerance = 1e-8;
    if ~isfinite(maxAbsError) || ~isfinite(maxRelError) || ...
            (maxAbsError > absTolerance && maxRelError > relTolerance)
        error(['Compiled PhDN validation failed: max absolute/relative ', ...
            'difference = %.6e / %.6e on %d probe points.'], ...
            maxAbsError, maxRelError, nProbe);
    end
end

function Xprobe = make_probe_points_local(data, nx)
    Xprobe = zeros(0, nx);
    fields = {'Xtr', 'Xval', 'Xte'};
    for k = 1:numel(fields)
        name = fields{k};
        if isfield(data, name) && isnumeric(data.(name)) && ...
                size(data.(name), 2) == nx && ~isempty(data.(name))
            values = data.(name);
            count = min(3, size(values, 1));
            Xprobe = [Xprobe; values(1:count, :)]; %#ok<AGROW>
        end
    end
    if ~isempty(Xprobe)
        finiteRows = all(isfinite(Xprobe), 2);
        Xprobe = Xprobe(finiteRows, :);
    end
end

function value = getfield_default_local(s, name, defaultValue)
    if isstruct(s) && isfield(s, name) && ~isempty(s.(name))
        value = s.(name);
    else
        value = defaultValue;
    end
end
