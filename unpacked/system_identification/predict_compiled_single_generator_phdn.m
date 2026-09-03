function Y = predict_compiled_single_generator_phdn(compiled, X)
%PREDICT_COMPILED_SINGLE_GENERATOR_PHDN Fixed value-only PhDN inference.
    if ~isnumeric(X) || ndims(X) ~= 2 || any(~isfinite(X(:)))
        error('Compiled PhDN input must be a finite two-dimensional numeric matrix.');
    end
    if size(X, 2) ~= compiled.dims(1)
        error('Input dimension mismatch: got %d, expected %d.', ...
            size(X, 2), compiled.dims(1));
    end

    nSamples = size(X, 1);
    Xnormalized = apply_input_norm(X, compiled.normOpt);
    Xnormalized(~isfinite(Xnormalized)) = 0;
    h = cell(1, compiled.layer + 1);
    h{1} = Xnormalized.';

    for ell = 1:compiled.layer
        rowDim = compiled.dims(ell + 1);
        layerValue = zeros(rowDim, nSamples);
        for src = 1:ell
            block = compiled.blocks{src, ell};
            if isempty(block) || isempty(block.A) || size(block.A, 2) == 0
                continue;
            end
            H = h{block.inputStateIndex};
            Phi = evaluate_compiled_value_terms(block.termPlan, H);
            layerValue = layerValue + block.A * Phi;
        end
        layerValue(~isfinite(layerValue)) = 0;

        if ell == compiled.layer
            clipBound = get_clip_bound_local( ...
                compiled.arch.safety, 'finalOutputClip', Inf);
        else
            clipBound = get_clip_bound_local(compiled.arch.safety, ...
                'hiddenLayerOutputClip', get_clip_bound_local( ...
                compiled.arch.safety, 'layerOutputClip', Inf));
        end
        if ~isinf(clipBound)
            layerValue = min(max(layerValue, -clipBound), clipBound);
        end
        layerValue(~isfinite(layerValue)) = 0;
        h{ell + 1} = layerValue;
    end

    Y = reverse_output_norm(h{end}.', compiled.normOpt);
end

function value = get_clip_bound_local(safety, name, defaultValue)
    if isstruct(safety) && isfield(safety, name) && ~isempty(safety.(name))
        value = safety.(name);
    else
        value = defaultValue;
    end
end
