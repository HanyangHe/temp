function Y = predict_compiled_soft_saturated_lorenz96_sindy(compiled, X)
%PREDICT_COMPILED_SOFT_SATURATED_LORENZ96_SINDY Evaluate final active terms only.
    if ~isnumeric(X) || ndims(X) ~= 2 || any(~isfinite(X(:)))
        error('Compiled SINDy input must be a finite two-dimensional numeric matrix.');
    end

    Phi = evaluate_compiled_value_terms(compiled.termPlan, X.').';
    if compiled.useScaled
        Phi = (Phi - compiled.mu) ./ compiled.sigma;
    end
    Y = Phi * compiled.Xi;
    if any(~isfinite(Y(:)))
        error('Compiled SINDy prediction produced NaN or Inf values.');
    end
end
